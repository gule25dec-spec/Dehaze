"""All paths, download sources and per-dataset checkpoint choices in one place."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # the project folder (…\Desktop\dehaze)
DOWNLOADS = ROOT / "downloads"
WEIGHTS = ROOT / "weights"
DATA = ROOT / "datasets"
PRED = ROOT / "predictions"
RESULTS = ROOT / "results"

MODELS = ["DEA-Net", "ConvIR", "CoA", "BiLaLoRA"]
DATASETS = ["RESIDE_SOTS_indoor", "O_HAZE", "NH_HAZE"]

# O-HAZE photos are up to ~5000 px wide. Both the hazy input and the ground truth are
# downscaled by the SAME factor so the longest side is at most this many pixels.
# Set to None to keep full resolution (needs a lot of GPU memory / time).
MAX_SIDE = {"RESIDE_SOTS_indoor": None, "O_HAZE": 1600, "NH_HAZE": None}

# ----------------------------------------------------------------------------------
# Dataset sources
# ----------------------------------------------------------------------------------
HF = "https://huggingface.co"
SOTS_ZIP_URL = f"{HF}/datasets/desonglll/reside-SOTS/resolve/main/SOTS.zip"        # 435 MB, official SOTS (indoor + outdoor)
NTIRE_REPO = "dronefreak/NTIRE-Haze"                                                  # mirror of O-HAZE / NH-HAZE
NH_HAZE_ZIP_URL = "https://data.vision.ee.ethz.ch/cvl/ntire20/nh-haze/files/NH-HAZE.zip"  # official, 330 MB

DATASET_PAGES = {
    "RESIDE (official page)": "https://sites.google.com/view/reside-dehaze-datasets/reside-standard",
    "O-HAZE (official page)": "https://data.vision.ee.ethz.ch/cvl/ntire18/o-haze/",
    "NH-HAZE (official page)": "https://data.vision.ee.ethz.ch/cvl/ntire20/nh-haze/",
    "SOTS mirror (Hugging Face)": "https://huggingface.co/datasets/desonglll/reside-SOTS",
    "O-HAZE / NH-HAZE mirror (Hugging Face)": "https://huggingface.co/datasets/dronefreak/NTIRE-Haze",
}

# ----------------------------------------------------------------------------------
# Pretrained weights (all official releases from the authors)
#   kind = "gdrive"  -> Google Drive file id (downloaded with gdown)
# ----------------------------------------------------------------------------------

WEIGHT_FILES = {
    "DEA-Net": {
        "its": dict(kind="gdrive", id="18gUXqal36Y8c5MVSt0uywDGIUGa6RqK_", file="PSNR4131_SSIM9945.pth"),
        "ots": dict(kind="gdrive", id="15MLmKtBEsULClWorzU2iz0JFjF3xQ42J", file="PSNR3659_SSIM9897.pth"),
    },
    "ConvIR": {   # ConvIR-B, one official checkpoint per benchmark
        "its": dict(kind="gdrive", id="1tyBKoX0yj5ohx1io2CroEU0NKNmIRs9p", file="its-base.pkl"),
        "ohaze": dict(kind="gdrive", id="1wABarIU30p9HwouC2qSIkeJjkTplxszz", file="ohaze-base.pkl"),
        "nhhaze": dict(kind="gdrive", id="18X-qsACAi5fFUxJUazduvnjEVJFZYAJD", file="nhhaze-base.pkl"),
    },
    # Real-world dehazing models (CVPR 2025 / CVPR 2026). Weights come as .rar archives; `inner`
    # lists the files taken out of the archive (the first one is the checkpoint path used).
    "CoA": {
        "ema_r": dict(kind="gdrive", id="1xSNdoYlAKsg2ayeRCZEBt1lzQvxMREy8", file="model.rar", inner=["EMA_r.pth"]),
    },
    "BiLaLoRA": {
        "lora": dict(kind="gdrive", id="14h3VyeaA6SxatmJ8_2uPIvvgq3k_2zok", file="weight.rar", inner=["LoRA.pth", "Base.pth"]),
    },
}

WEIGHT_PAGES = {
    "DEA-Net": "https://drive.google.com/drive/folders/1Rjb8dpyNnvvr0XLvIX9fg8Hdru_MhMCj",
    "ConvIR": "https://drive.google.com/drive/folders/1_5fO2p5xoWO5cUEVoXJ7x3Uhg1AP18FQ",
    "CoA": "https://drive.google.com/drive/folders/1GPUtjvlASVokIjaXJlt8j-QsFxLT9Zgg",
    "BiLaLoRA": "https://drive.google.com/drive/folders/1qEmYEONhX8e0b7PLikKxMCKNu_j7pu6b",
}

# Which released checkpoint is used on which test set.
# DEA-Net has no O-HAZE / NH-HAZE weights, so its outdoor (OTS) checkpoint is used there
# (cross-dataset generalisation); ConvIR uses its official in-domain checkpoints.
CKPT_FOR = {
    "DEA-Net":   {"RESIDE_SOTS_indoor": "its", "O_HAZE": "ots", "NH_HAZE": "ots"},
    # ConvIR released a dedicated checkpoint for every benchmark (trained on its training split).
    "ConvIR":    {"RESIDE_SOTS_indoor": "its", "O_HAZE": "ohaze", "NH_HAZE": "nhhaze"},
    # CoA and BiLaLoRA each release ONE real-world model (trained on synthetic RESIDE/Haze4K/RIDCP data
    # + unpaired real hazy photos); the same checkpoint is used on every test set.
    "CoA":       {"RESIDE_SOTS_indoor": "ema_r", "O_HAZE": "ema_r", "NH_HAZE": "ema_r"},
    "BiLaLoRA":  {"RESIDE_SOTS_indoor": "lora", "O_HAZE": "lora", "NH_HAZE": "lora"},
}

# Official held-out test subsets reported separately (scene ids).
# O-HAZE (NTIRE 2018): 01-35 train, 36-40 validation, 41-45 test.
# NH-HAZE (NTIRE 2020): 01-45 train, 46-50 validation, 51-55 test.
# ConvIR's O-HAZE / NH-HAZE checkpoints were trained on those training splits, so the fair
# comparison on the real-haze sets is the official test images.
TEST_SUBSETS = {"O_HAZE_test41-45": ("O_HAZE", ["41", "42", "43", "44", "45"]),
                "NH_HAZE_test51-55": ("NH_HAZE", ["51", "52", "53", "54", "55"])}

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def weight_path(model, tag):
    spec = WEIGHT_FILES[model][tag]
    inner = spec.get("inner")
    return WEIGHTS / model / (inner[0] if inner else spec["file"])


def checkpoint_name(model, tag):
    return weight_path(model, tag).name
