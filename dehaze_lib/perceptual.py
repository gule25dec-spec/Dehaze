"""Perceptual / no-reference quality metrics, as used by the real-world dehazing papers.

MUSIQ  (Ke et al., ICCV 2021): no-reference image quality, KonIQ-10k model, 0-100, higher is better.
       Used by BiLaLoRA (CVPR 2026) to evaluate real-haze results. Needs no ground truth.
LPIPS  (Zhang et al., CVPR 2018): learned perceptual distance to the ground truth, AlexNet
       features, lower is better. Less sensitive than PSNR to global brightness/colour shifts.

Implementations: pyiqa (MUSIQ, weights from Hugging Face) and the official `lpips` package.
Results: results/per_image_perceptual/<dataset>__<model>.csv, merged into summary_all.csv by
evaluate.build_summary().
"""
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from . import config as C
from .data import get_pairs
from .models import load_image

REFERENCE_ROWS = ["Hazy input (no dehazing)", "Ground truth"]


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _tensor(arr, dev):
    return torch.from_numpy(np.ascontiguousarray(arr)).permute(2, 0, 1).unsqueeze(0).to(dev)


def build_metrics():
    import pyiqa
    import lpips
    dev = _device()
    musiq = pyiqa.create_metric("musiq", device=dev, as_loss=False)   # KonIQ-10k checkpoint
    lp = lpips.LPIPS(net="alex", verbose=False).to(dev).eval()
    return musiq, lp, dev


@torch.no_grad()
def score(musiq, lp, dev, pred, gt=None):
    x = _tensor(pred, dev)
    out = {"MUSIQ": float(musiq(x).item())}
    if gt is not None:
        g = _tensor(gt, dev)
        out["LPIPS"] = float(lp(x * 2 - 1, g * 2 - 1).item())
    return out


def evaluate_perceptual(models=None, datasets=None, overwrite=False):
    musiq, lp, dev = build_metrics()
    print("Perceptual metrics on", dev, "| MUSIQ (KonIQ, higher better), LPIPS-Alex (lower better)")
    out_dir = C.RESULTS / "per_image_perceptual"
    out_dir.mkdir(parents=True, exist_ok=True)
    for ds in datasets or C.DATASETS:
        pairs, msg = get_pairs(ds)
        if not pairs:
            print(f"skip {ds}: {msg}")
            continue
        for model in REFERENCE_ROWS + list(models or C.MODELS):
            f = out_dir / f"{ds}__{model.split(' ')[0]}.csv"
            if f.exists() and not overwrite:
                continue
            recs, seen_gt = [], set()
            for sid, hz, gt_path in tqdm(pairs, desc=f"perc {model[:12]:12s} {ds}", leave=False):
                if model == "Ground truth":           # score each clear image once
                    if gt_path in seen_gt:
                        continue
                    seen_gt.add(gt_path)
                    recs.append({"Image": gt_path.stem, **score(musiq, lp, dev, load_image(gt_path))})
                    continue
                p = hz if model.startswith("Hazy") else C.PRED / ds / model / f"{sid}.png"
                if not p.exists():
                    continue
                recs.append({"Image": sid, **score(musiq, lp, dev, load_image(p), load_image(gt_path))})
            if recs:
                pd.DataFrame(recs).assign(Dataset=ds, Model=model).to_csv(f, index=False)
                if dev.type == "cuda":
                    torch.cuda.empty_cache()
    from .evaluate import build_summary
    return build_summary()
