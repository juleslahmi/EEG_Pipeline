# scripts/build_cache.py
import argparse, sys
from pathlib import Path
import json
from src import data_load

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
    ap.add_argument("--target", default="diagnosis", choices=["diagnosis", "event"])
    ap.add_argument("--event-map", default=None)
    args = ap.parse_args()

    event_map = json.loads(args.event_map) if args.event_map else None
    patients = data_load.discover_subjects(args.data_root)
    dataset = data_load.PatientEpochsDataset(patients, target=args.target, event_map=event_map)
    bundle = {
        "dataset": dataset,
        "n_chans": dataset.n_chans,
        "n_times": dataset.n_times,
        "patients": patients,
        "target": args.target,
    }
    data_load.save_dataset_cache(bundle, args.out)

if __name__ == "__main__":
    main()
