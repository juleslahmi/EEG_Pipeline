# scripts/build_cache.py
import argparse, sys
from pathlib import Path
import json
from src import data_load

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
    ap.add_argument(
        "--drop-region",
        nargs="+",
        default=None,
        help="One or more regions to drop (e.g., frontal temporal). See src.channel_groups.py",
    )
    ap.add_argument(
        "--drop-channels",
        nargs="+",
        default=None,
        help="Specific channels to drop (e.g., Fp1 Fp2)",
    )
    ap.add_argument("--freq", type=int, default=128, help="Resampling frequency")
    args = ap.parse_args()

    event_map = json.loads(args.event_map) if args.event_map else None
    drop_channels = []

    if args.drop_region:
        from src.channel_groups import get_channels_to_drop

        for region in args.drop_region:
            region_channels = get_channels_to_drop(region)
            drop_channels.extend(region_channels)
            
    if args.drop_channels:
        drop_channels.extend(args.drop_channels)
        
    if drop_channels:
        drop_channels = list(sorted(set(drop_channels)))
    else:
        drop_channels = None

    patients = data_load.discover_subjects(args.data_root)
    dataset = data_load.PatientEpochsDataset(patients, target=args.target, event_map=event_map, freq=args.freq, drop_channels=drop_channels)
    bundle = {
        "dataset": dataset,
        "n_chans": dataset.n_chans,
        "n_times": dataset.n_times,
        "patients": patients,
        "target": args.target,
        "event_map": event_map,
        "freq": args.freq,
        "drop_region": args.drop_region,
        "drop_channels": args.drop_channels,
    }
    data_load.save_dataset_cache(bundle, args.out)

if __name__ == "__main__":
    main()
