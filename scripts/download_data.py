"""Download datasets used in the project.

  DIV2K train HR (800 images, ~3.5 GB)  -> training
  DIV2K valid HR (100 images, ~450 MB)  -> validation
  Set5 / Set14 HR                       -> standard SR benchmarks

Usage:
  python scripts/download_data.py # everything
  python scripts/download_data.py --only set5 set14 div2k_valid
"""
import argparse
import shutil
import tarfile
import urllib.request
import zipfile
from pathlib import Path

SOURCES = {
    "div2k_train": ("https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip", "DIV2K_train_HR"),
    "div2k_valid": ("https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip", "DIV2K_valid_HR"),
    "set5": ("https://huggingface.co/datasets/eugenesiow/Set5/resolve/main/data/Set5_HR.tar.gz", "Set5_HR"),
    "set14": ("https://huggingface.co/datasets/eugenesiow/Set14/resolve/main/data/Set14_HR.tar.gz", "Set14_HR"),
}


def download(url: str, dst: Path) -> None:
    print(f"Downloading {url}")
    tmp = dst.with_suffix(dst.suffix + ".part")
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length", 0))
        done = 0
        while chunk := r.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done / 2**20:.0f}/{total / 2**20:.0f} MB", end="", flush=True)
    print()
    tmp.rename(dst)


def extract(archive: Path, out_dir: Path) -> None:
    print(f"Extracting {archive.name}")
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as z:
            z.extractall(out_dir)
    else:
        with tarfile.open(archive) as t:
            t.extractall(out_dir, filter="data")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="data")
    p.add_argument("--only", nargs="*", choices=list(SOURCES), default=list(SOURCES))
    p.add_argument("--keep-archives", action="store_true")
    args = p.parse_args()

    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    for name in args.only:
        url, folder = SOURCES[name]
        if (root / folder).is_dir():
            print(f"{folder} already present, skipping")
            continue
        archive = root / url.rsplit("/", 1)[1]
        if not archive.exists():
            download(url, archive)
        extract(archive, root)
        if not args.keep_archives:
            archive.unlink()
        # Some archives may nest the folder; flatten to root/folder.
        if not (root / folder).is_dir():
            hits = list(root.rglob(folder))
            if hits:
                shutil.move(str(hits[0]), root / folder)
        print(f"  -> {root / folder}")


if __name__ == "__main__":
    main()
