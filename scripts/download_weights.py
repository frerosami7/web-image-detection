#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path
from urllib.request import urlopen

DEFAULT_DIR = Path(__file__).resolve().parents[1] / "checkpoints"


def download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url) as resp, open(dest, "wb") as out:
        out.write(resp.read())
    return dest


def main():
    p = argparse.ArgumentParser(description="Download model weights (.pt) to checkpoints/")
    p.add_argument("--url", help="URL to .pt file; if omitted, uses MODEL_URL env var", default=None)
    p.add_argument("--out", help="Output filename (under checkpoints/)", default=None)
    args = p.parse_args()

    url = args.url or os.environ.get("MODEL_URL")
    if not url:
        print("ERROR: Provide --url or set MODEL_URL env var.", file=sys.stderr)
        sys.exit(2)

    fname = args.out or Path(url).name or "model.pt"
    dest = DEFAULT_DIR / fname
    try:
        path = download(url, dest)
        print(str(path))
    except Exception as e:
        print(f"Download failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
