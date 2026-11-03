"""Common evaluation of all predictions with one metric implementation.

PSNR / SSIM: scikit-image, RGB, data_range = 1, SSIM gaussian window (sigma 1.5) as in Wang et al. 2004.
MAE: mean absolute error in [0,1].  Optional LPIPS (AlexNet) if the `lpips` package is installed.
"""
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from . import config as C
from .data import get_pairs
from .models import load_image


def metrics(gt, pred):
    if gt.shape != pred.shape:
        raise ValueError(f"size mismatch GT {gt.shape} vs prediction {pred.shape}")
    return {
        "PSNR": float(peak_signal_noise_ratio(gt, pred, data_range=1.0)),
        "SSIM": float(structural_similarity(gt, pred, data_range=1.0, channel_axis=-1,
                                            gaussian_weights=True, sigma=1.5, use_sample_covariance=False)),
        "MAE": float(np.abs(gt - pred).mean()),
    }


def _lpips_fn():
    try:
        import lpips, torch
        net = lpips.LPIPS(net="alex", verbose=False)
        def f(gt, pred):
            t = lambda a: torch.from_numpy(a).permute(2, 0, 1)[None] * 2 - 1
            with torch.no_grad():
                return float(net(t(gt), t(pred)))
        return f
    except Exception:
        return None


def evaluate_all(models=None, datasets=None, use_lpips=False, include_hazy_baseline=True):
    lp = _lpips_fn() if use_lpips else None
    per_dir = C.RESULTS / "per_image"
    per_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for ds in datasets or C.DATASETS:
        pairs, msg = get_pairs(ds)
        if not pairs:
            print(f"skip {ds}: {msg}")
            continue
        candidates = (["Hazy input (no dehazing)"] if include_hazy_baseline else []) + list(models or C.MODELS)
        for model in candidates:
            recs = []
            for sid, hz, gt_path in tqdm(pairs, desc=f"eval {model[:12]:12s} {ds}", leave=False):
                pred_path = hz if model.startswith("Hazy") else C.PRED / ds / model / f"{sid}.png"
                if not pred_path.exists():
                    continue
                gt, pr = load_image(gt_path), load_image(pred_path)
                m = metrics(gt, pr)
                if lp:
                    m["LPIPS"] = lp(gt, pr)
                recs.append({"Dataset": ds, "Model": model, "Image": sid, **m})
            if not recs:
                continue
            df = pd.DataFrame(recs)
            df.to_csv(per_dir / f"{ds}__{model.split(' ')[0]}.csv", index=False)
            rows.append(len(df))
    if not rows:
        print("Nothing to evaluate yet - run inference first.")
        return pd.DataFrame()
    return build_summary()


def _summ(df, name, n_total):
    row = {"Dataset": name, "Model": df.Model.iloc[0], "N": len(df), "N_total": n_total,
           "PSNR": df.PSNR.mean(), "PSNR_std": df.PSNR.std(),
           "SSIM": df.SSIM.mean(), "SSIM_std": df.SSIM.std(), "MAE": df.MAE.mean()}
    if "LPIPS" in df:
        row["LPIPS"] = df.LPIPS.mean()
    return row


def build_summary():
    """Aggregate results/per_image/*.csv into summary_all.csv, table_*.csv and summary.md.
    Also adds the official test subsets defined in config.TEST_SUBSETS (e.g. NH-HAZE 51-55)."""
    per_dir = C.RESULTS / "per_image"
    order = ["Hazy input (no dehazing)"] + C.MODELS
    rows = []
    for ds in C.DATASETS:
        for model in order:
            f = per_dir / f"{ds}__{model.split(' ')[0]}.csv"
            if not f.exists():
                continue
            df = pd.read_csv(f, dtype={"Image": str})
            df["Model"] = model
            rows.append(_summ(df, ds, len(df)))
            for sub_name, (base, ids) in C.TEST_SUBSETS.items():
                if base == ds:
                    sub = df[df.Image.str.split("_").str[0].isin(ids)]
                    if len(sub):
                        rows.append(_summ(sub, sub_name, len(ids)))
    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    log = C.RESULTS / "inference_log.csv"
    if log.exists():
        lg = pd.read_csv(log)[["Model", "Dataset", "Checkpoint", "Sec_per_image", "Params_M", "Device"]]
        summary["_base"] = summary.Dataset.map(lambda d: C.TEST_SUBSETS.get(d, (d,))[0])
        summary = summary.merge(lg.rename(columns={"Dataset": "_base"}), on=["Model", "_base"], how="left").drop(columns="_base")
    summary = _merge_perceptual(summary)
    summary.to_csv(C.RESULTS / "summary_all.csv", index=False)
    cols = [d for d in C.DATASETS + list(C.TEST_SUBSETS) if d in set(summary.Dataset)]
    order_gt = order + ["Ground truth"]
    for metric in ("PSNR", "SSIM", "MAE", "MUSIQ", "LPIPS"):
        if metric not in summary or summary[metric].isna().all():
            continue
        summary.pivot(index="Model", columns="Dataset", values=metric).reindex(
            index=[m for m in order_gt if m in set(summary.Model)], columns=cols
        ).round(4).to_csv(C.RESULTS / f"table_{metric}.csv")
    _write_markdown(summary)
    return summary


def _merge_perceptual(summary):
    """Add mean MUSIQ / LPIPS from results/per_image_perceptual/ (see perceptual.py), incl. test subsets.
    A 'Ground truth' row carries the MUSIQ of the clear images as an upper reference."""
    pdir = C.RESULTS / "per_image_perceptual"
    files = sorted(pdir.glob("*.csv")) if pdir.exists() else []
    if not files:
        return summary
    recs = []
    for f in files:
        df = pd.read_csv(f, dtype={"Image": str})
        ds, model = df.Dataset.iloc[0], df.Model.iloc[0]
        if model not in C.MODELS and model not in ("Hazy input (no dehazing)", "Ground truth"):
            continue
        groups = [(ds, df)]
        for sub_name, (base, ids) in C.TEST_SUBSETS.items():
            if base == ds:
                groups.append((sub_name, df[df.Image.str.split("_").str[0].isin(ids)]))
        for name, g in groups:
            if len(g):
                recs.append({"Dataset": name, "Model": model, "MUSIQ": g.MUSIQ.mean(),
                             "LPIPS": g.LPIPS.mean() if "LPIPS" in g else np.nan,
                             **({"N": len(g), "N_total": len(g)} if model == "Ground truth" else {})})
    perc = pd.DataFrame(recs)
    gt = perc[perc.Model == "Ground truth"]
    out = summary.merge(perc[perc.Model != "Ground truth"].drop(columns=[c for c in ("N", "N_total") if c in perc]),
                        on=["Dataset", "Model"], how="left")
    return pd.concat([out, gt], ignore_index=True)


def _write_markdown(summary):
    lines = ["# Dehazing results (same images, same metric code for every model)", ""]
    for ds, g in summary.groupby("Dataset", sort=False):
        lines += [f"## {ds}", "", "| Model | Checkpoint | N | PSNR (dB) | SSIM | MAE | MUSIQ | LPIPS | s/img |",
                  "|---|---|---|---|---|---|---|---|---|"]
        f = lambda v, d: "" if pd.isna(v) else f"{v:.{d}f}"
        for _, r in g.iterrows():
            partial = "" if r.N == r.N_total else f" (of {int(r.N_total)})"
            lines.append(f"| {r.Model} | {r.get('Checkpoint', '') if isinstance(r.get('Checkpoint', ''), str) else ''} | "
                         f"{int(r.N)}{partial} | {f(r.get('PSNR'), 2)} | {f(r.get('SSIM'), 4)} | {f(r.get('MAE'), 4)} | "
                         f"{f(r.get('MUSIQ', np.nan), 2)} | {f(r.get('LPIPS', np.nan), 4)} | {f(r.get('Sec_per_image', np.nan), 3)} |")
        lines.append("")
    (C.RESULTS / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def plot_results(summary=None, save=True):
    import matplotlib.pyplot as plt
    if summary is None:
        summary = pd.read_csv(C.RESULTS / "summary_all.csv")
    s = summary[summary.Model.isin(C.MODELS)]
    dsets = [d for d in C.DATASETS + list(C.TEST_SUBSETS) if d in set(s.Dataset)]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    w = 0.8 / max(1, len(C.MODELS))
    for ax, metric in zip(axes, ("PSNR", "SSIM")):
        for i, m in enumerate(C.MODELS):
            vals = [s[(s.Model == m) & (s.Dataset == d)][metric].mean() for d in dsets]
            ax.bar(np.arange(len(dsets)) + i * w, vals, w, label=m)
        ax.set_xticks(np.arange(len(dsets)) + w * (len(C.MODELS) - 1) / 2)
        ax.set_xticklabels([d.replace("RESIDE_", "").replace("_", " ") for d in dsets])
        ax.set_ylabel(metric + (" (dB)" if metric == "PSNR" else ""))
        ax.grid(axis="y", alpha=.3)
    axes[0].legend(fontsize=8)
    plt.tight_layout()
    if save:
        (C.RESULTS / "figures").mkdir(parents=True, exist_ok=True)
        fig.savefig(C.RESULTS / "figures" / "psnr_ssim_bars.png", dpi=150)
    return fig


def plot_gain(summary=None, save=True):
    """PSNR gain of each model over the unprocessed hazy input, per test set."""
    import matplotlib.pyplot as plt
    if summary is None:
        summary = pd.read_csv(C.RESULTS / "summary_all.csv")
    base = summary[summary.Model.str.startswith("Hazy")].set_index("Dataset").PSNR
    dsets = [d for d in [C.DATASETS[0]] + [x for x in C.DATASETS[1:] if x not in {b for b, _ in C.TEST_SUBSETS.values()}]
             + list(C.TEST_SUBSETS) if d in base.index]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    w = 0.8 / len(C.MODELS)
    for i, m in enumerate(C.MODELS):
        s = summary[summary.Model == m].set_index("Dataset").PSNR
        vals = [s.get(d, np.nan) - base[d] for d in dsets]
        bars = ax.bar(np.arange(len(dsets)) + i * w, vals, w, label=m,
                      color=["#6f6f6f", "#2f6fd6", "#e0a458", "#c97b2a"][i % 4],
                      hatch="//" if m in ("CoA", "BiLaLoRA") else None, edgecolor="white")
        ax.bar_label(bars, fmt="%+.1f", fontsize=8)
    ax.set_xticks(np.arange(len(dsets)) + w * (len(C.MODELS) - 1) / 2)
    ax.set_xticklabels([d.replace("RESIDE_", "").replace("_", " ") for d in dsets])
    ax.set_ylabel("PSNR gain over hazy input (dB)")
    ax.set_title("PSNR gain of each model over the hazy input (hatched = real-world models, 2025-26)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=8, frameon=False)
    plt.tight_layout()
    if save:
        (C.RESULTS / "figures").mkdir(parents=True, exist_ok=True)
        fig.savefig(C.RESULTS / "figures" / "psnr_gain.png", dpi=150)
    return fig


def show_examples(dataset, idx=0, save=True):
    import matplotlib.pyplot as plt
    pairs, _ = get_pairs(dataset)
    if not pairs:
        print("no data for", dataset)
        return None
    sid, hz, gt = pairs[min(idx, len(pairs) - 1)]
    panels = [("Hazy", hz)] + [(m, C.PRED / dataset / m / f"{sid}.png") for m in C.MODELS] + [("Ground truth", gt)]
    panels = [(t, p) for t, p in panels if p.exists()]
    fig, axes = plt.subplots(1, len(panels), figsize=(3.2 * len(panels), 3.2))
    gtim = load_image(gt)
    for ax, (t, p) in zip(np.atleast_1d(axes), panels):
        im = load_image(p)
        title = t
        if t not in ("Hazy", "Ground truth") or t == "Hazy":
            m = metrics(gtim, im)
            title += f"\n{m['PSNR']:.2f} dB / {m['SSIM']:.3f}"
        ax.imshow(im)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    plt.suptitle(f"{dataset} - {sid}", fontsize=10)
    plt.tight_layout()
    if save:
        (C.RESULTS / "figures").mkdir(parents=True, exist_ok=True)
        fig.savefig(C.RESULTS / "figures" / f"examples_{dataset}_{sid}.png", dpi=150)
    return fig


def plot_perceptual(summary=None, save=True):
    """MUSIQ (no-reference, higher better; dashed line = ground-truth images) and LPIPS (lower better)."""
    import matplotlib.pyplot as plt
    if summary is None:
        summary = pd.read_csv(C.RESULTS / "summary_all.csv")
    if "MUSIQ" not in summary or summary.MUSIQ.isna().all():
        return None
    bases = {b for b, _ in C.TEST_SUBSETS.values()}
    dsets = [d for d in C.DATASETS if d not in bases] + list(C.TEST_SUBSETS)
    dsets = [d for d in dsets if d in set(summary.Dataset)]
    order = ["Hazy input (no dehazing)"] + C.MODELS
    cols = ["#bdbdbd", "#6f6f6f", "#2f6fd6", "#e0a458", "#c97b2a"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    w = 0.8 / len(order)
    for ax, metric in zip(axes, ("MUSIQ", "LPIPS")):
        for i, m in enumerate(order):
            s = summary[summary.Model == m].set_index("Dataset")[metric]
            vals = [s.get(d, np.nan) for d in dsets]
            bars = ax.bar(np.arange(len(dsets)) + i * w, vals, w, label=m.replace(" (no dehazing)", ""),
                          color=cols[i % len(cols)], hatch="//" if m in ("CoA", "BiLaLoRA") else None, edgecolor="white")
            ax.bar_label(bars, fmt="%.1f" if metric == "MUSIQ" else "%.3f", fontsize=7)
        if metric == "MUSIQ":
            gt = summary[summary.Model == "Ground truth"].set_index("Dataset").MUSIQ
            for j, d in enumerate(dsets):
                if d in gt.index:
                    ax.hlines(gt[d], j - 0.05, j + w * len(order) - 0.05, colors="black", linestyles="--", linewidth=1)
            ax.set_title("MUSIQ, no-reference quality (higher is better; dashed = ground-truth images)")
            ax.set_ylim(40, 70)
        else:
            ax.set_title("LPIPS distance to ground truth (lower is better)")
        ax.set_xticks(np.arange(len(dsets)) + w * (len(order) - 1) / 2)
        ax.set_xticklabels([d.replace("RESIDE_", "").replace("_", " ") for d in dsets])
        ax.spines[["top", "right"]].set_visible(False)
    axes[1].legend(fontsize=8, frameon=False)
    plt.tight_layout()
    if save:
        (C.RESULTS / "figures").mkdir(parents=True, exist_ok=True)
        fig.savefig(C.RESULTS / "figures" / "musiq_lpips_bars.png", dpi=150)
    return fig
