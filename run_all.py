"""One-command pipeline for Assignment 2 (image dehazing on RESIDE-SOTS, O-HAZE, NH-HAZE).

  python run_all.py all                 # everything: download, organize, run 4 models, evaluate
  python run_all.py all --limit 5       # quick test with 5 images per dataset
  python run_all.py download            # datasets + official weights
  python run_all.py organize            # build datasets/<name>/{hazy,gt}
  python run_all.py infer  [--models FFA-Net DEA-Net] [--datasets O_HAZE] [--limit N] [--tile 512]
  python run_all.py evaluate            # PSNR / SSIM / MAE tables + figures in results/
  python run_all.py summary             # rebuild tables/figures from results/per_image/*.csv
  python run_all.py status              # what is downloaded / predicted so far

Console output of every run is also appended to results/run_log.txt.
"""
import argparse
import datetime
import sys
import warnings

warnings.filterwarnings("ignore")

from dehaze_lib import config as C


def status():
    from dehaze_lib.download import weights_status
    from dehaze_lib.data import get_pairs
    import torch
    print("Project folder :", C.ROOT)
    print("PyTorch        :", torch.__version__, "| CUDA:", torch.cuda.is_available(),
          f"({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else "")
    print("\nWeights:")
    for k, ok in weights_status().items():
        print(f"  {'OK ' if ok else '-- '} {k}")
    print("\nDatasets / predictions:")
    for ds in C.DATASETS:
        pairs, msg = get_pairs(ds)
        preds = {m: len(list((C.PRED / ds / m).glob('*.png'))) for m in C.MODELS}
        print(f"  {ds:22s} {msg}")
        print("     predictions:", ", ".join(f"{m} {n}" for m, n in preds.items()))


class _Tee:
    """Write console output to the screen and to results/run_log.txt."""
    def __init__(self, stream, path):
        self.stream, self.f = stream, open(path, "a", encoding="utf-8")
    def write(self, s):
        self.stream.write(s)
        self.f.write(s)
    def flush(self):
        self.stream.flush()
        self.f.flush()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["all", "download", "organize", "infer", "evaluate", "summary", "status"])
    ap.add_argument("--models", nargs="*", default=None, choices=C.MODELS)
    ap.add_argument("--datasets", nargs="*", default=None, choices=C.DATASETS)
    ap.add_argument("--limit", type=int, default=None, help="only the first N hazy images per dataset")
    ap.add_argument("--tile", type=int, default=None, help="force tiled inference with this tile size (low memory)")
    ap.add_argument("--overwrite", action="store_true", help="recompute existing predictions")
    ap.add_argument("--lpips", action="store_true", help="also compute LPIPS (pip install lpips)")
    a = ap.parse_args()
    C.RESULTS.mkdir(parents=True, exist_ok=True)
    sys.stdout = _Tee(sys.stdout, C.RESULTS / "run_log.txt")
    print(f"\n##### {datetime.datetime.now():%Y-%m-%d %H:%M:%S}  run_all.py {' '.join(sys.argv[1:])}")

    if a.step in ("all", "download"):
        from dehaze_lib.download import download_datasets, download_weights
        print("=== 1/4 Downloading datasets ===")
        download_datasets()
        print("\n=== Downloading official pretrained weights ===")
        download_weights(a.models)
    if a.step in ("all", "organize"):
        from dehaze_lib.data import organize_all
        print("\n=== 2/4 Organizing datasets ===")
        organize_all()
    if a.step in ("all", "infer"):
        from dehaze_lib.models import run_inference
        print("\n=== 3/4 Running the four models ===")
        run_inference(a.models, a.datasets, a.limit, a.tile, a.overwrite)
    if a.step in ("all", "evaluate"):
        from dehaze_lib.evaluate import evaluate_all, plot_results, plot_gain, show_examples
        import matplotlib
        matplotlib.use("Agg")
        print("\n=== 4/4 Evaluating ===")
        s = evaluate_all(a.models, a.datasets, use_lpips=a.lpips)
        if not s.empty:
            cols = [c for c in ["Dataset", "Model", "N", "PSNR", "SSIM", "MAE", "Sec_per_image"] if c in s]
            print(s[cols].round(4).to_string(index=False))
            plot_results(s)
            plot_gain(s)
            for ds in C.DATASETS:
                show_examples(ds, 0)
            print("\nSaved: results/summary_all.csv, results/summary.md, results/table_*.csv, results/figures/")
    if a.step == "summary":
        from dehaze_lib.evaluate import build_summary, plot_results, plot_gain
        import matplotlib
        matplotlib.use("Agg")
        s = build_summary()
        print(s[["Dataset", "Model", "N", "PSNR", "SSIM", "MAE"]].round(4).to_string(index=False))
        plot_results(s)
        plot_gain(s)
    if a.step == "status":
        status()


if __name__ == "__main__":
    sys.exit(main())
