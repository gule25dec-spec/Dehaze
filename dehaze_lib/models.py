"""Load the four official networks with their released weights and run inference.

Pre/post-processing follows each repository's own test script:
  FFA-Net   : ImageNet-style Normalize(mean=[0.64,0.6,0.58], std=[0.14,0.15,0.152]) input   (net/test.py)
  MSBDN-DFF : [0,1] input, size padded to a multiple of 16                               (test.py)
  AECR-Net  : [0,1] input, size padded to a multiple of 4                                (train_aecrnet.py)
  DEA-Net   : [0,1] input, reflect-padded to a multiple of 4                             (code/eval.py)
"""
import io
import pickle
import sys
import time
import types
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
def _build_ffa(path):
    from third_party.ffa import FFA
    net = FFA(gps=3, blocks=19)
    ckp = _torch_load(path)
    net.load_state_dict(_strip_module(ckp["model"] if "model" in ckp else ckp))
    return net


def _msbdn_pickle_module():
    """MSBDN-DFF released its weights with torch.save(model): a pickled *module object*
    whose classes live in 'networks.<file>'. Map those names onto the bundled code."""
    from third_party.msbdn import MSBDN_DFF_v1_1 as v11, MSBDN_RDFF as rdff, base_networks as base
    lookup = [v11, base, rdff]

    class Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module.startswith("networks") or "MSBDN" in module or module.startswith("base_networks"):
                for m in lookup:
                    if hasattr(m, name):
                        return getattr(m, name)
            return super().find_class(module, name)

    mod = types.ModuleType("msbdn_pickle")
    mod.__dict__.update({k: getattr(pickle, k) for k in dir(pickle) if not k.startswith("__")})
    mod.Unpickler = Unpickler
    mod.load = lambda f, **kw: Unpickler(f, **kw).load()
    mod.loads = lambda b, **kw: Unpickler(io.BytesIO(b), **kw).load()
    return mod


def _build_msbdn(path):
    from third_party.msbdn.MSBDN_DFF_v1_1 import Net as NetDFF
    from third_party.msbdn.MSBDN_RDFF import Net as NetRDFF
    obj = _torch_load(path, pickle_module=_msbdn_pickle_module())
    if isinstance(obj, dict):
        sd = obj.get("model", obj.get("state_dict", obj))
        if isinstance(sd, torch.nn.Module):
            sd = sd.state_dict()
    else:
        sd = obj.state_dict()
    sd = _strip_module(sd)
    for cls in (NetDFF, NetRDFF):
        net = cls()
        try:
            net.load_state_dict(sd, strict=True)
            return net
        except RuntimeError:
            continue
    if isinstance(obj, torch.nn.Module):   # architecture differs from both files: use the pickled model itself
        print("  note: MSBDN checkpoint does not match the bundled Net exactly; using the pickled module")
        return obj
    raise RuntimeError("MSBDN-DFF checkpoint could not be loaded")


def _build_aecr(path):
    from third_party.aecr import Dehaze
    net = Dehaze(3, 3)
    ckp = _torch_load(path)
    net.load_state_dict(_strip_module(ckp["model"] if "model" in ckp else ckp))
    return net


def _build_dea(path):
    from third_party.dea import Backbone
    net = Backbone()
    net.load_state_dict(_strip_module(_torch_load(path)))
    return net


BUILDERS = {"FFA-Net": _build_ffa, "MSBDN-DFF": _build_msbdn, "AECR-Net": _build_aecr,
            "DEA-Net": _build_dea}


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


_FFA_MEAN = torch.tensor([0.64, 0.6, 0.58]).view(1, 3, 1, 1)
_FFA_STD = torch.tensor([0.14, 0.15, 0.152]).view(1, 3, 1, 1)


def _fwd_ffa(net, x):
    xn = (x - _FFA_MEAN.to(x)) / _FFA_STD.to(x)
    return net(xn)


def _fwd_pad(m):
    def f(net, x):
        xp, h, w = _pad_to(x, m)
        y = net(xp)
        y = y[0] if isinstance(y, (tuple, list)) else y
        return y[..., :h, :w]
    return f


FORWARD = {"FFA-Net": _fwd_ffa, "MSBDN-DFF": _fwd_pad(16), "AECR-Net": _fwd_pad(4),
           "DEA-Net": _fwd_pad(4)}


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
            log.append({"Model": model, "Dataset": ds, "Checkpoint": C.WEIGHT_FILES[model][tag].get("inner", C.WEIGHT_FILES[model][tag]["file"]),
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
