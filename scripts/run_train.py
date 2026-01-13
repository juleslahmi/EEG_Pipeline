# scripts/run_train.py

import argparse
import json
from pathlib import Path
from datetime import datetime
import numpy as np

# Import project modules
from src import data_load, split, train, tcn_supress_warning
from src.build_model import build_model

def main():
    # -------------------------------------------------
    # 1) Parse CLI arguments
    # -------------------------------------------------
    parser = argparse.ArgumentParser(description="Train EEG classifier with chosen model and CV scheme")

    parser.add_argument("--data-root", type=str, required=True, help="Path to data root (containing Control/ and Dyslexic/)")

    parser.add_argument("--model", type=str, default="shallow", choices=["shallow", "deep4", "eegnet", "tcn", "hybridnet", "biot"])
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)

    parser.add_argument("--cv-scheme", type=str, default="groupkfold", choices=["groupkfold", "loso"])
    parser.add_argument("--cv-n-splits", type=int, default=10)
    parser.add_argument("--cv-fold", type=str, default="all", help="'all' or integer index of fold")
    parser.add_argument("--cv-seed", type=int, default=42)

    parser.add_argument("--outdir", type=str, default="runs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--dataset-cache", type=str, default=None,
                    help="Path to .npz cache produced by data_load.save_dataset_cache")
    
    parser.add_argument("--target", type=str, default="diagnosis",
                choices=["diagnosis", "event"],
                help="What to predict: patient group (diagnosis) or event code.")
    parser.add_argument("--event-map", type=str, default=None,
                    help="Optional JSON mapping from raw event codes to class ids. "
                        "Example: '{\"201\":0,\"202\":1,\"203\":2,\"204\":3,\"205\":4}'")
    parser.add_argument("--exclude-events", type=int, nargs="*", default=None, help="Event codes to exclude from training")
    parser.add_argument("--freq", type=int, default=40)
    parser.add_argument("--tmin", type=float, default=-0.1)
    parser.add_argument("--tmax", type=float, default=1)
    parser.add_argument("--drop-channels", type=str, nargs='*', default=None)

    args = parser.parse_args()

    # -------------------------------------------------
    # 2) Load dataset
    # -------------------------------------------------
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Loading dataset from {args.data_root} ...")
    
    if args.dataset_cache:
        print(f"Loading dataset cache: {args.dataset_cache}")
        bundle = data_load.load_dataset_from_cache(args.dataset_cache)
    else:
        print(f"Loading dataset from {args.data_root} ...")
        bundle = data_load.load_dataset(args.data_root, extension=".set", freq=args.freq, tmin=args.tmin, tmax=args.tmax)

    #target = args.target  # Either 'diagnosis' or 'event'
    event_map = json.loads(args.event_map) if args.event_map else None

    #dataset = PatientEpochsDataset(patients, target=target, event_map=event_map)

    ds = bundle["dataset"]
    n_chans, n_times = bundle["n_chans"], bundle["n_times"]
    print(f"  -> Loaded {bundle['n_subjects']} subjects, shape ({n_chans} chans, {n_times} samples)")

    if args.exclude_events:
        print(f"Excluding events: {args.exclude_events}")
        # Need to filter ds.X, ds.y, ds.groups, ds.patients, ds.events
        # ds.events is np array
        mask = ~np.isin(ds.events, args.exclude_events)
        print(f"  -> Excluding {np.sum(~mask)} samples, keeping {np.sum(mask)}")
        
        ds.X = ds.X[mask]
        ds.y = ds.y[mask]
        ds.events = ds.events[mask]
        
        # groups and patients are lists, we need to filter them
        ds.groups = [g for i, g in enumerate(ds.groups) if mask[i]]
        ds.patients = [p for i, p in enumerate(ds.patients) if mask[i]]
        
        print(f"  -> New dataset size: {len(ds.X)}")

    # -------------------------------------------------
    # 3) Make splits
    # -------------------------------------------------
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Creating {args.cv_scheme} splits ...")
    splits = split.make_splits(ds, scheme=args.cv_scheme, n_splits=args.cv_n_splits, seed=args.cv_seed)
    print(f"  -> {len(splits)} folds generated")

    # -------------------------------------------------
    # 4) Prepare configs
    # -------------------------------------------------
    n_classes = 5 if args.target == "event" else 2

    labels = np.asarray(ds.y, dtype=np.int64)
    uniq = np.unique(labels)

    if uniq.min() < 0 or uniq.max() >= n_classes:
        print(f"[WARN] Remapping labels; got uniques {uniq.tolist()} with n_classes={n_classes}")
        remap = {int(v): i for i, v in enumerate(sorted(uniq))}
        ds.y = np.array([remap[int(v)] for v in labels], dtype=np.int64)
        uniq = np.unique(ds.y)
        n_classes = int(len(uniq))

    print(f"-> Target={args.target}, n_classes={n_classes}, classes={uniq.tolist()}")
        
    model_cfg = {
        "name": args.model,
        "n_chans": n_chans,
        "n_times": n_times,
        "n_classes": n_classes,
    }
    train_cfg = {
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "max_epochs": args.epochs,
        "scheduler": "plateau",
        "plateau_factor": 0.5,
        "plateau_patience": 3,
        "early_stopping": True,
        "early_patience": 16,
    }

    # -------------------------------------------------
    # 5) Choose folds
    # -------------------------------------------------
    if args.cv_fold.lower() == "all":
        fold_ids = range(len(splits))
    else:
        fold_ids = [int(args.cv_fold)]

    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    all_results = []

    # -------------------------------------------------
    # 6) Train folds
    # -------------------------------------------------
    for k in fold_ids:
        fold = splits[k]
        valid_idx = fold["valid_idx"]
        valid_subjects = np.unique([ds.patients[i] for i in valid_idx])
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] === Training fold {k}/{len(splits)-1} ({args.model}) ===")
        print(f"    LOSO held-out subject(s): {valid_subjects.tolist()}")

        # Build model
        model = build_model(model_cfg)

        if args.model=='tcn':
            tcn_supress_warning.replace_dropout2d(model)

        # Train
        result = train.fit_one_fold(
            dataset=ds,
            train_idx=fold["train_idx"],
            valid_idx=fold["valid_idx"],
            model=model,
            outdir=args.outdir,
            train_cfg=train_cfg,
            device=args.device,
            seed=args.seed,
            run_name=f"{args.model}_{args.cv_scheme}_fold{k}",
            fold_id=k,
            target=args.target,
            event_map=event_map 
        )
        all_results.append(result)
    # -------------------------------------------------
    # 7) Summarize
    # -------------------------------------------------
    summary_path = Path(args.outdir) / f"summary_{args.model}_{args.cv_scheme}.json"
    summary = {
        "model": args.model,
        "cv_scheme": args.cv_scheme,
        "n_folds": len(fold_ids),
        "folds": [
            {
                "fold": i,
                "run_dir": res["run_dir"],
                "metrics_path": res["metrics_path"],
            }
            for i, res in zip(fold_ids, all_results)
        ],
    }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n Training complete. Summary written to {summary_path}")
    return 0


if __name__ == "__main__":
    main()
