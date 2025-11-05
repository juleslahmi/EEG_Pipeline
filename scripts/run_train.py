# scripts/run_train.py

import argparse
import json
from pathlib import Path
from datetime import datetime

# Import project modules
from src import data_load, split, train, tcn_supress_warning
from src.build_model import build_model

def main():
    # -------------------------------------------------
    # 1) Parse CLI arguments
    # -------------------------------------------------
    parser = argparse.ArgumentParser(description="Train EEG classifier with chosen model and CV scheme")

    parser.add_argument("--data-root", type=str, required=True, help="Path to data root (containing Control/ and Dyslexic/)")
    parser.add_argument("--model", type=str, default="shallow", choices=["shallow", "deep4", "eegnet", "tcn", "hybridnet"])
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
        bundle = data_load.load_dataset(args.data_root, extension=".set")
    ds = bundle["dataset"]
    n_chans, n_times = bundle["n_chans"], bundle["n_times"]
    print(f"  -> Loaded {bundle['n_subjects']} subjects, shape ({n_chans} chans, {n_times} samples)")

  

    # -------------------------------------------------
    # 3) Make splits
    # -------------------------------------------------
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Creating {args.cv_scheme} splits ...")
    splits = split.make_splits(ds, scheme=args.cv_scheme, n_splits=args.cv_n_splits, seed=args.cv_seed)
    print(f"  -> {len(splits)} folds generated")

    # -------------------------------------------------
    # 4) Prepare configs
    # -------------------------------------------------
    model_cfg = {
        "name": args.model,
        "n_chans": n_chans,
        "n_times": n_times,
        "n_classes": 2,
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
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] === Training fold {k}/{len(splits)-1} ({args.model}) ===")

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
            fold_id=k
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
