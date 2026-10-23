"""Organise the downloaded datasets into  datasets/<name>/{hazy,gt}/  and pair the images.

Pairing rule (works for all four test sets):
  * RESIDE-SOTS indoor : hazy '1400_1.png' ... '1400_10.png' -> scene id before the first '_'
                  -> gt '1400.png'   (10 hazy images share one GT, 500 pairs in total)
  * O-HAZE      : '01_outdoor_hazy.jpg'  <-> '01_outdoor_GT.jpg'
  * NH-HAZE     : '01_hazy.png'          <-> '01_GT.png'
Each *hazy* image is one evaluation sample; predictions are saved under the hazy file's stem.
"""
import json
import re
import shutil
from pathlib import Path

from PIL import Image
from tqdm.auto import tqdm

from . import config as C

Image.MAX_IMAGE_PIXELS = None


def list_images(folder):
    folder = Path(folder)
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in C.IMAGE_EXT)


def scene_key(path, dataset):
    s = Path(path).stem
    if dataset.startswith("RESIDE"):
        return s.split("_")[0]
    return re.sub(r"(?i)[_\- ](hazy|haze|gt|clear|groundtruth)$", "", s)


def get_pairs(dataset):
    """Return [(sample_id, hazy_path, gt_path), ...] and a short report string."""
    hazy = list_images(C.DATA / dataset / "hazy")
    gts = {}
    for g in list_images(C.DATA / dataset / "gt"):
        k = scene_key(g, dataset)
        if k in gts:
            raise ValueError(f"two GT images for scene {k}: {gts[k].name}, {g.name}")
        gts[k] = g
    pairs, missing = [], []
    for h in hazy:
        k = scene_key(h, dataset)
        (pairs.append((h.stem, h, gts[k])) if k in gts else missing.append(h.name))
    msg = f"{len(pairs)} hazy/GT pairs ({len(gts)} GT scenes)"
    if missing:
        msg += f"; {len(missing)} hazy images without GT skipped"
    return pairs, msg


def _copy(src, dst_dir, max_side):
    """Copy (and optionally downscale) one image; downscaled copies are saved as PNG."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    if max_side is None:
        dst = dst_dir / src.name
        if not dst.exists():
            shutil.copy2(src, dst)
        return dst
    dst = dst_dir / (src.stem + ".png")
    if dst.exists():
        return dst
    with Image.open(src) as im:
        im = im.convert("RGB")
        w, h = im.size
        s = min(1.0, max_side / max(w, h))
        if s < 1.0:
            im = im.resize((round(w * s), round(h * s)), Image.BICUBIC)
        im.save(dst)
    return dst


def _fill(dataset, hazy_files, gt_files):
    ms = C.MAX_SIDE.get(dataset)
    for kind, files in (("hazy", hazy_files), ("gt", gt_files)):
        for f in tqdm(files, desc=f"{dataset}/{kind}"):
            _copy(f, C.DATA / dataset / kind, ms)
    note = {"source_hazy": len(hazy_files), "source_gt": len(gt_files),
            "max_side": ms, "resize": "bicubic, same factor for hazy and GT" if ms else "none"}
    (C.DATA / dataset / "PREPROCESSING.json").write_text(json.dumps(note, indent=2))


def _find_dir(root, *names):
    """Find a sub-folder by (case-insensitive) name anywhere under root."""
    wanted = [n.lower() for n in names]
    for p in sorted(Path(root).rglob("*")):
        if p.is_dir() and p.name.lower() in wanted:
            return p
    return None


def organize_all():
    ex = C.DOWNLOADS / "extracted"
    report = {}

    # RESIDE-SOTS
    sots = ex / "SOTS"
    for split in ("indoor",):
        ds = f"RESIDE_SOTS_{split}"
        base = _find_dir(sots, split) if sots.exists() else None
        hz = _find_dir(base, "hazy") if base else None
        gt = _find_dir(base, "gt", "clear", "GT") if base else None
        if hz and gt:
            _fill(ds, list_images(hz), list_images(gt))
        else:
            print(f"{ds}: source not found under {sots} (run the download step)")

    # O-HAZE
    oh = ex / "O-HAZE"
    if oh.exists():
        imgs = list_images(oh)
        _fill("O_HAZE", [p for p in imgs if re.search(r"(?i)hazy", p.stem)],
              [p for p in imgs if re.search(r"(?i)_gt$", p.stem)])
    else:
        print("O_HAZE: source not found (run the download step)")

    # NH-HAZE
    nh = ex / "NH-HAZE"
    if nh.exists():
        imgs = list_images(nh)
        _fill("NH_HAZE", [p for p in imgs if p.stem.lower().endswith("_hazy")],
              [p for p in imgs if p.stem.lower().endswith("_gt")])
    else:
        print("NH_HAZE: source not found (run the download step)")

    for ds in C.DATASETS:
        pairs, msg = get_pairs(ds)
        report[ds] = msg
        print(f"{ds:22s}: {msg}")
    return report
