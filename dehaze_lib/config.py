"""All paths, download sources and per-dataset checkpoint choices in one place."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # the project folder (…\Desktop\dehaze)
DOWNLOADS = ROOT / "downloads"
WEIGHTS = ROOT / "weights"
DATA = ROOT / "datasets"
PRED = ROOT / "predictions"
RESULTS = ROOT / "results"

MODELS = ["FFA-Net", "MSBDN-DFF", "AECR-Net", "DEA-Net"]
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
#   kind = "gitee"   -> git blob of the AECR-Net mirror (the official link is Baidu-only)
# ----------------------------------------------------------------------------------
GITEE_AECR = "https://gitee.com/api/v5/repos/damon_one/AECR-Net1/git/blobs/"

WEIGHT_FILES = {
    "FFA-Net": {
        "its": dict(kind="gdrive", id="1xVbgsabnI2-fVw3vmsra690YmBwiK_su", file="its_train_ffa_3_19.pk"),
        "ots": dict(kind="gdrive", id="1EGIJADl24Om3Hv7_bJVMPiqU1Dd0kg3b", file="ots_train_ffa_3_19.pk"),
    },
    "MSBDN-DFF": {
        "reside": dict(kind="gdrive", id="1da13IOlJ3FQfH6Duj_u1exmZzgXPaYXe", file="model.pkl"),
    },
    "AECR-Net": {
        "its": dict(kind="gitee", sha="793e6e0d523d31390fc636dd3b5b506bab0100c6", file="ITS_train.pk"),
        "nh": dict(kind="gitee", sha="ac9c0a3f7dbe045d658e87c2c9d8eef840e6d72a", file="NH_train.pk"),
    },
    "DEA-Net": {
        "its": dict(kind="gdrive", id="18gUXqal36Y8c5MVSt0uywDGIUGa6RqK_", file="PSNR4131_SSIM9945.pth"),
        "ots": dict(kind="gdrive", id="15MLmKtBEsULClWorzU2iz0JFjF3xQ42J", file="PSNR3659_SSIM9897.pth"),
    },
}

WEIGHT_PAGES = {
    "FFA-Net": "https://drive.google.com/drive/folders/19_lSUPrpLDZl9AyewhHBsHidZEpTMIV5",
    "MSBDN-DFF": "https://drive.google.com/open?id=1da13IOlJ3FQfH6Duj_u1exmZzgXPaYXe",
    "AECR-Net": "https://gitee.com/damon_one/AECR-Net1/tree/main/trained_models  (official: https://pan.baidu.com/s/13crsXwwhkI5A3MlHtPihuA pwd xhyi)",
    "DEA-Net": "https://drive.google.com/drive/folders/1Rjb8dpyNnvvr0XLvIX9fg8Hdru_MhMCj",
}

# Which released checkpoint is used on which test set.
# No author released O-HAZE weights, and only AECR-Net released NH-HAZE weights, so for the
# real-haze sets we use the closest outdoor/real-haze checkpoint (cross-dataset generalisation).
CKPT_FOR = {
    "FFA-Net":   {"RESIDE_SOTS_indoor": "its", "O_HAZE": "ots", "NH_HAZE": "ots"},
    "MSBDN-DFF": {"RESIDE_SOTS_indoor": "reside", "O_HAZE": "reside", "NH_HAZE": "reside"},
    "AECR-Net":  {"RESIDE_SOTS_indoor": "its", "O_HAZE": "nh", "NH_HAZE": "nh"},
    "DEA-Net":   {"RESIDE_SOTS_indoor": "its", "O_HAZE": "ots", "NH_HAZE": "ots"},
}

# Official held-out test subsets reported separately (scene ids).
# NH-HAZE official split: 01-45 train, 46-50 validation, 51-55 test. AECR-Net's NH checkpoint was
# trained on 01-45, so the fair comparison on NH-HAZE is the 5 test images.
TEST_SUBSETS = {"NH_HAZE_test51-55": ("NH_HAZE", ["51", "52", "53", "54", "55"])}

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def weight_path(model, tag):
    spec = WEIGHT_FILES[model][tag]
    return WEIGHTS / model / spec.get("inner", spec["file"])
