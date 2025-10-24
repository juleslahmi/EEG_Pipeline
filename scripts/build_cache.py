# scripts/build_cache.py
import argparse, sys
from pathlib import Path

# ensure project root on sys.path if needed
from pathlib import Path as _P
import sys as _S
ROOT = _P(__file__).resolve().parents[1]
if str(ROOT) not in _S.path: _S.path.insert(0, str(ROOT))

from src.data_load import load_dataset, save_dataset_cache

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    bundle = load_dataset(args.data_root, extension=".set")
    save_dataset_cache(bundle, args.out)
    print(f"Wrote dataset cache: {args.out}")

if __name__ == "__main__":
    main()
