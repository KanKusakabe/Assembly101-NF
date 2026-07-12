"""Fetch the open Assembly101 mistake-detection annotations (CSV only, no video).

Clones https://github.com/assembly-101/assembly101-mistake-detection into
data/raw/md.  The annotations are CC BY-NC 4.0 (attribution, non-commercial);
they are NOT redistributed in this repo — run this to obtain them.
"""
from __future__ import annotations

import subprocess

from . import config as C

REPO = "https://github.com/assembly-101/assembly101-mistake-detection.git"


def main() -> None:
    dst = C.RAW / "md"
    if (dst / "annots").exists():
        print("already present:", dst)
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "--depth", "1", REPO, str(dst)], check=True)
    print("cloned to", dst)


if __name__ == "__main__":
    main()
