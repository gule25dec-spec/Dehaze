"""Load the official networks with their released weights and run inference.

Pre/post-processing follows each repository's own test script:
  DEA-Net   : [0,1] input, reflect-padded to a multiple of 4                             (code/eval.py)
  ConvIR    : [0,1] input, reflect-padded to a multiple of 32, full-scale output [2]    (Dehazing/ITS/eval.py)
  CoA       : CLIP Normalize, bicubic resize to a multiple of 16 and back; output[0]      (Eval.py)
  BiLaLoRA  : same as CoA (base network + LoRA layers)                                  (BiLaLoRA_Test.py)
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from . import config as C

sys.path.insert(0, str(C.ROOT))  # so "third_party" is importable


def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _strip_module(sd):
    return {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}


def _torch_load(path, **kw):
    try:
        return torch.load(path, map_location="cpu", weights_only=False, **kw)
    except TypeError:  # very old torch without weights_only
        return torch.load(path, map_location="cpu", **kw)


# ----------------------------------------------------------------------------------
# builders: return an nn.Module with weights loaded
# ----------------------------------------------------------------------------------
def _build_convir(path):
    from third_party.convir import build_net
    net = build_net("base", "ITS")
    ckp = _torch_load(path)
    net.load_state_dict(_strip_module(ckp["model"] if "model" in ckp else ckp))
    return net


def _build_dea(path):
    from third_party.dea import Backbone
    net = Backbone()
    net.load_state_dict(_strip_module(_torch_load(path)))
    return net


def _build_coa(path):
    from third_party.coa import Student_x
    net = Student_x()
    net.load_state_dict(_strip_module(_torch_load(path)))
    return net


def _build_bilalora(path):
    from third_party.bilalora import build_bilalora
    return build_bilalora(Path(path).parent / "Base.pth", path)


BUILDERS = {"DEA-Net": _build_dea, "ConvIR": _build_convir, "CoA": _build_coa, "BiLaLoRA": _build_bilalora}


def load_model(model, tag):
    path = C.weight_path(model, tag)
    if not path.exists():
        raise FileNotFoundError(f"missing weights {path} (run the download step)")
    net = BUILDERS[model](path).eval().to(device())
    for p in net.parameters():
        p.requires_grad_(False)
    return net


# ----------------------------------------------------------------------------------
# forward passes (x: 1x3xHxW float in [0,1] on device) -> 1x3xHxW in [0,1]
# ----------------------------------------------------------------------------------
def _pad_to(x, m, mode="reflect"):
    h, w = x.shape[-2:]
    ph, pw = (m - h % m) % m, (m - w % m) % m
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph), mode=mode if min(h, w) > max(ph, pw) else "replicate")
    return x, h, w


_CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
_CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)


def _fwd_pad(m, pick=0):
    def f(net, x):
        xp, h, w = _pad_to(x, m)
        y = net(xp)
        y = y[pick] if isinstance(y, (tuple, list)) else y
        return y[..., :h, :w]
    return f


def _fwd_clip_resize16(net, x):
    """CoA / BiLaLoRA official test: CLIP normalisation, bicubic resize to a multiple of 16, take output[0],
    resize back to the input size."""
    from torchvision.transforms.functional import resize, InterpolationMode
    h, w = x.shape[-2:]
    xn = (x - _CLIP_MEAN.to(x)) / _CLIP_STD.to(x)
    xr = resize(xn, [max(16, h // 16 * 16), max(16, w // 16 * 16)],
                interpolation=InterpolationMode.BICUBIC, antialias=True)
    y = net(xr)
    y = y[0] if isinstance(y, (tuple, list)) else y
    return resize(y, [h, w], interpolation=InterpolationMode.BICUBIC, antialias=True)


FORWARD = {"DEA-Net": _fwd_pad(4), "ConvIR": _fwd_pad(32, pick=2),
           "CoA": _fwd_clip_resize16, "BiLaLoRA": _fwd_clip_resize16}


def _tiled(fn, net, x, tile=512, overlap=64):
    """Run `fn` on overlapping tiles and blend them (used only when an image is too big for memory)."""
    _, _, H, W = x.shape
    out = torch.zeros_like(x)
    weight = torch.zeros_like(x[:, :1])
    step = tile - overlap
    ys = list(range(0, max(H - tile, 0) + 1, step)) or [0]
    xs = list(range(0, max(W - tile, 0) + 1, step)) or [0]
    if ys[-1] + tile < H:
        ys.append(H - tile)
    if xs[-1] + tile < W:
        xs.append(W - tile)
    ramp = torch.ones(1, 1, min(tile, H), min(tile, W), device=x.device)
    for y0 in ys:
        for x0 in xs:
            y1, x1 = min(y0 + tile, H), min(x0 + tile, W)
            y0c, x0c = max(0, y1 - tile), max(0, x1 - tile)
            patch = fn(net, x[..., y0c:y1, x0c:x1]).clamp(0, 1)
            wgt = ramp[..., : y1 - y0c, : x1 - x0c]
            out[..., y0c:y1, x0c:x1] += patch * wgt
            weight[..., y0c:y1, x0c:x1] += wgt
    return out / weight.clamp_min(1e-8)


def _is_oom(e):
    s = str(e).lower()
    return "out of memory" in s or "defaultcpuallocator" in s or "not enough memory" in s


@torch.inference_mode()
def dehaze_tensor(model, net, x, tile=None):
    fn = FORWARD[model]
    if tile:
        return _tiled(fn, net, x, tile)
    try:
        return fn(net, x).clamp(0, 1)
    except RuntimeError as e:
        if not _is_oom(e):
            raise
        if x.is_cuda:
            torch.cuda.empty_cache()
        return _tiled(fn, net, x, 512)


def load_image(path):
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0


def to_tensor(arr):
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device())


def save_tensor(t, path):
    arr = (t.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0 + 0.5).clip(0, 255).astype(np.uint8)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)


def count_params(net):
    return sum(p.numel() for p in net.parameters()) / 1e6


def run_inference(models=None, datasets=None, limit=None, tile=None, overwrite=False):
    """Dehaze every hazy test image with every model; writes predictions/<dataset>/<model>/<id>.png.
    Returns a list of dicts with timing info (also saved to results/inference_log.csv)."""
    import pandas as pd
    from tqdm.auto import tqdm
    from .data import get_pairs

    log = []
    dev = device()
    print("Device:", dev, "|", torch.cuda.get_device_name(0) if dev.type == "cuda" else "CPU (slow: use --limit for a quick test)")
    for model in models or C.MODELS:
        cache = {}
        for ds in datasets or C.DATASETS:
            pairs, msg = get_pairs(ds)
            if not pairs:
                print(f"skip {ds}: no data ({msg})")
                continue
            if limit:
                pairs = pairs[:limit]
            tag = C.CKPT_FOR[model][ds]
            try:
                if tag not in cache:
                    cache.clear()
                    if dev.type == "cuda":
                        torch.cuda.empty_cache()
                    cache[tag] = load_model(model, tag)
                net = cache[tag]
            except Exception as e:
                print(f"skip {model} on {ds}: {e}")
                continue
            out_dir = C.PRED / ds / model
            times = []
            prev_shape = None
            for sid, hz, _gt in tqdm(pairs, desc=f"{model:9s} {ds}"):
                dst = out_dir / f"{sid}.png"
                if dst.exists() and not overwrite:
                    continue
                x = to_tensor(load_image(hz))
                if dev.type == "cuda":
                    # Images of different sizes fragment the CUDA cache; on Windows the overflow
                    # silently spills into (very slow) shared system memory. Release it when the size changes.
                    if x.shape != prev_shape:
                        torch.cuda.empty_cache()
                    prev_shape = x.shape
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                y = dehaze_tensor(model, net, x, tile)
                if dev.type == "cuda":
                    torch.cuda.synchronize()
                times.append(time.perf_counter() - t0)
                save_tensor(y, dst)
            log.append({"Model": model, "Dataset": ds, "Checkpoint": C.checkpoint_name(model, tag),
                        "Images_run": len(times), "Sec_per_image": float(np.mean(times)) if times else np.nan,
                        "Params_M": round(count_params(net), 3), "Device": str(dev)})
    C.RESULTS.mkdir(parents=True, exist_ok=True)
    if log:
        # keep timings from earlier runs unless this run actually processed images
        f = C.RESULTS / "inference_log.csv"
        rows = {}
        if f.exists():
            for r in pd.read_csv(f).to_dict("records"):
                rows[(r["Model"], r["Dataset"])] = r
        for r in log:
            k = (r["Model"], r["Dataset"])
            if r["Images_run"] > 0 or k not in rows:
                rows[k] = r
        pd.DataFrame(list(rows.values())).to_csv(f, index=False)
    return log
