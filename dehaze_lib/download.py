"""Download the three datasets and the official pretrained weights of the five models.

Everything is resumable: files that already exist are skipped, so you can re-run safely.
"""
import shutil
import subprocess
import zipfile
from pathlib import Path
from urllib.parse import quote

import requests
from tqdm.auto import tqdm

from . import config as C

UA = {"User-Agent": "Mozilla/5.0 (dehaze-assignment)"}


# ----------------------------------------------------------------------------------
# generic helpers
# ----------------------------------------------------------------------------------
def http_download(url, dest, desc=None, min_bytes=1):
    """Stream a URL to `dest` with a progress bar and resume support."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size >= min_bytes:
        print(f"  [skip] {dest.name} already downloaded")
        return dest
    part = dest.with_suffix(dest.suffix + ".part")
    have = part.stat().st_size if part.exists() else 0
    headers = dict(UA)
    if have:
        headers["Range"] = f"bytes={have}-"
    with requests.get(url, headers=headers, stream=True, timeout=60, allow_redirects=True) as r:
        if r.status_code == 416:      # already complete
            part.rename(dest)
            return dest
        r.raise_for_status()
        if have and r.status_code != 206:   # server ignored Range -> restart
            have = 0
        total = int(r.headers.get("Content-Length", 0)) + have
        mode = "ab" if have else "wb"
        with open(part, mode) as f, tqdm(total=total or None, initial=have, unit="B", unit_scale=True,
                                         desc=desc or dest.name) as bar:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
                bar.update(len(chunk))
    part.replace(dest)
    return dest


def safe_extract_zip(zip_path, target):
    target = Path(target)
    target.mkdir(parents=True, exist_ok=True)
    base = target.resolve()
    with zipfile.ZipFile(zip_path) as z:
        for item in z.infolist():
            if not (target / item.filename).resolve().is_relative_to(base):
                raise ValueError(f"Unsafe path in archive: {item.filename}")
        z.extractall(target)
    return target


def extract_rar(rar_path, target):
    """Extract a .rar on Windows/Linux using whatever tool is available."""
    rar_path, target = Path(rar_path), Path(target)
    target.mkdir(parents=True, exist_ok=True)
    candidates = [
        ["tar", "-xf", str(rar_path), "-C", str(target)],                       # Windows 10/11 bsdtar
        [r"C:\Program Files\7-Zip\7z.exe", "x", "-y", f"-o{target}", str(rar_path)],
        [r"C:\Program Files (x86)\7-Zip\7z.exe", "x", "-y", f"-o{target}", str(rar_path)],
        ["7z", "x", "-y", f"-o{target}", str(rar_path)],
        [r"C:\Program Files\WinRAR\UnRAR.exe", "x", "-y", str(rar_path), str(target) + "\\"],
        ["unrar", "x", "-y", str(rar_path), str(target) + "/"],
    ]
    for cmd in candidates:
        exe = cmd[0]
        if Path(exe).is_absolute() and not Path(exe).exists():
            continue
        if not Path(exe).is_absolute() and shutil.which(exe) is None:
            continue
        try:
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode == 0 and any(target.rglob("*.pth")):
                print(f"  extracted {rar_path.name} with {Path(exe).name}")
                return True
        except Exception:
            pass
    try:  # last resort: python 'rarfile' (still needs unrar/bsdtar on PATH)
        import rarfile
        with rarfile.RarFile(rar_path) as rf:
            rf.extractall(target)
        return any(target.rglob("*.pth"))
    except Exception:
        return False


# ----------------------------------------------------------------------------------
# datasets
# ----------------------------------------------------------------------------------
def download_sots():
    print("RESIDE-SOTS (indoor + outdoor test set, 435 MB)")
    z = http_download(C.SOTS_ZIP_URL, C.DOWNLOADS / "SOTS.zip", min_bytes=400_000_000)
    out = C.DOWNLOADS / "extracted" / "SOTS"
    if not any(out.rglob("*.png")):
        print("  extracting ...")
        safe_extract_zip(z, out)
    return out


def _hf_list(repo, path):
    url = f"{C.HF}/api/datasets/{repo}/tree/main/{quote(path)}"
    r = requests.get(url, headers=UA, timeout=60)
    r.raise_for_status()
    return [x for x in r.json() if x.get("type") == "file"]


def download_ohaze():
    print("O-HAZE (45 real outdoor hazy/clear pairs, ~0.5 GB)")
    out = C.DOWNLOADS / "extracted" / "O-HAZE"
    for sub in ("GT", "hazy"):
        files = _hf_list(C.NTIRE_REPO, f"O-HAZE/{sub}")
        if len(files) != 45:
            print(f"  warning: expected 45 files in O-HAZE/{sub}, found {len(files)}")
        for f in tqdm(files, desc=f"O-HAZE/{sub}"):
            url = f"{C.HF}/datasets/{C.NTIRE_REPO}/resolve/main/{quote(f['path'])}"
            dest = out / sub / Path(f["path"]).name
            if dest.exists() and dest.stat().st_size == f.get("size", -1):
                continue
            http_download(url, dest, desc=dest.name, min_bytes=f.get("size", 1))
    return out


def download_nhhaze():
    print("NH-HAZE (55 real non-homogeneous hazy/clear pairs, 330 MB)")
    out = C.DOWNLOADS / "extracted" / "NH-HAZE"
    if len(list(out.rglob("*_hazy.png"))) >= 55:
        print("  [skip] already extracted")
        return out
    z = C.DOWNLOADS / "NH-HAZE.zip"
    try:
        http_download(C.NH_HAZE_ZIP_URL, z, min_bytes=300_000_000)
        safe_extract_zip(z, out)
    except Exception as e:  # official server down -> Hugging Face mirror
        print("  official server failed (", e, ") -> using Hugging Face mirror")
        for f in tqdm(_hf_list(C.NTIRE_REPO, "NH-HAZE"), desc="NH-HAZE"):
            url = f"{C.HF}/datasets/{C.NTIRE_REPO}/resolve/main/{quote(f['path'])}"
            http_download(url, out / "NH-HAZE" / Path(f["path"]).name, min_bytes=f.get("size", 1))
    return out


def download_datasets():
    C.DOWNLOADS.mkdir(parents=True, exist_ok=True)
    ok = {}
    for name, fn in [("SOTS", download_sots), ("O-HAZE", download_ohaze), ("NH-HAZE", download_nhhaze)]:
        try:
            fn()
            ok[name] = True
        except Exception as e:
            ok[name] = False
            print(f"  FAILED {name}: {e}")
            print("  Manual option: download from", C.DATASET_PAGES)
    return ok


# ----------------------------------------------------------------------------------
# weights
# ----------------------------------------------------------------------------------
def _gdrive(file_id, dest):
    import gdown
    dest.parent.mkdir(parents=True, exist_ok=True)
    out = gdown.download(id=file_id, output=str(dest), quiet=False)
    if out is None or not dest.exists() or dest.stat().st_size < 10_000:
        raise RuntimeError("Google Drive download failed (quota or network). "
                           f"Open https://drive.google.com/uc?id={file_id} in a browser and save it as {dest}")


def download_weights(models=None):
    status = {}
    for model in models or C.MODELS:
        for tag, spec in C.WEIGHT_FILES[model].items():
            final = C.weight_path(model, tag)
            key = f"{model}/{tag}"
            if final.exists() and final.stat().st_size > 10_000:
                print(f"  [skip] {key}: {final.name}")
                status[key] = True
                continue
            raw = C.WEIGHTS / model / spec["file"]
            try:
                print(f"{key}:")
                if not raw.exists():
                    if spec["kind"] == "gdrive":
                        _gdrive(spec["id"], raw)
                if raw.suffix == ".rar":
                    names = spec["inner"]
                    if not extract_rar(raw, raw.parent / "extracted"):
                        raise RuntimeError(
                            f"Could not extract {raw}. Right-click it -> Extract All (Windows 11) or use 7-Zip/WinRAR, "
                            f"then copy {', '.join(names)} into {final.parent}")
                    for name in names:
                        hits = sorted((raw.parent / "extracted").rglob(name))
                        if not hits:
                            raise RuntimeError(f"{name} not found inside {raw.name}")
                        shutil.copy2(hits[0], final.parent / name)
                status[key] = final.exists()
            except Exception as e:
                status[key] = False
                print(f"  FAILED {key}: {e}\n  Manual page: {C.WEIGHT_PAGES[model]}")
    return status


def weights_status():
    return {f"{m}/{t}": C.weight_path(m, t).exists() for m in C.MODELS for t in C.WEIGHT_FILES[m]}
