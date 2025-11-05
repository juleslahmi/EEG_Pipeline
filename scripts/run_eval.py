# scripts/run_eval.py
"""
Evaluate trained runs by reloading checkpoints and computing:
- epoch-level accuracy
- patient-level accuracy

Usage:
  python -m scripts.run_eval --data-root /path/to/Data --model shallow --cv-scheme loso --outdir runs
  python -m scripts.run_eval --data-root /path/to/Data --model eegnet --cv-scheme groupkfold --cv-n-splits 10 --fold all
"""

import argparse
import json
from pathlib import Path
import numpy as np
from datetime import datetime

# Ensure project root on path if needed
import sys
from pathlib import Path as _Path
ROOT = _Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import data_load, split
from src.eval import evaluate_fold


def _find_checkpoint(run_dir: Path) -> Path | None:
    cands = sorted(run_dir.glob("best_*.pt"))
    return cands[-1] if cands else None


def main():
    ap = argparse.ArgumentParser(description="Evaluate EEG runs and build a leaderboard.")
    ap.add_argument("--data-root", type=str, required=True)
    ap.add_argument("--model", type=str, required=True,
                    choices=["shallow", "deep4", "eegnet", "tcn", "hybridnet"])
    ap.add_argument("--cv-scheme", type=str, default="loso", choices=["loso", "groupkfold"])
    ap.add_argument("--cv-n-splits", type=int, default=10)
    ap.add_argument("--cv-fold", type=str, default="all", help="'all' or integer fold index")
    ap.add_argument("--outdir", type=str, default="runs")
    ap.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--dataset-cache", type=str, default=None,
                help="Path to .npz cache produced by data_load.save_dataset_cache")
    args = ap.parse_args()

    # 1) Load dataset
    if args.dataset_cache:
        print(f"Loading dataset cache: {args.dataset_cache}")
        bundle = data_load.load_dataset_from_cache(args.dataset_cache)
    else:
        print(f"Loading dataset from {args.data_root} ...")
        bundle = data_load.load_dataset(args.data_root, extension=".set")
    ds = bundle["dataset"]
    n_chans, n_times = bundle["n_chans"], bundle["n_times"]

    # 2) Rebuild splits in the same way training did
    splits = split.make_splits(ds, scheme=args.cv_scheme, n_splits=args.cv_n_splits, seed=42)

    # 3) Model cfg for reconstruction
    model_cfg = {"name": args.model, "n_chans": n_chans, "n_times": n_times, "n_classes": 2}

    # 4) Determine folds to evaluate
    if args.cv_fold.lower() == "all":
        fold_ids = range(len(splits))
    else:
        fold_ids = [int(args.cv_fold)]

    # 5) For each fold, locate the run dir and checkpoint, evaluate
    outdir = Path(args.outdir)
    leaderboard = []
    missing = []

    # Expected run name pattern from training: "{model}_{cv_scheme}_fold{k}"
    for k in fold_ids:
        pattern = f"{args.model}_{args.cv_scheme}_fold{k}"
        # Find the newest directory with that prefix (in case multiple launches)
        candidates = sorted([p for p in outdir.glob(f"{pattern}*") if p.is_dir()], key=lambda p: p.stat().st_mtime)
        if not candidates:
            missing.append({"fold": k, "reason": "no run_dir found", "pattern": pattern})
            continue
        run_dir = candidates[-1]
        ckpt = _find_checkpoint(run_dir)
        if ckpt is None:
            missing.append({"fold": k, "reason": "no checkpoint", "run_dir": str(run_dir)})
            continue

        res = evaluate_fold(
            dataset=ds,
            train_idx=splits[k]["train_idx"],
            valid_idx=splits[k]["valid_idx"],
            model_cfg=model_cfg,
            checkpoint_path=ckpt,
            device=args.device,
        )
        leaderboard.append({
            "fold": k,
            "run_dir": str(run_dir),
            "checkpoint": str(ckpt),
            **res
        })
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Fold {k} -> "
              f"epoch_acc={res['epoch_acc']:.4f}, patient_acc={res['patient_acc']:.4f}")

    # 6) Save leaderboard CSV + summary JSON
    if leaderboard:
        import pandas as pd
        df = pd.DataFrame(leaderboard).sort_values("fold")
        lb_path = outdir / f"leaderboard_{args.model}_{args.cv_scheme}.csv"
        df.to_csv(lb_path, index=False)
        print(f"\n Leaderboard written to: {lb_path}")

        summary = {
            "model": args.model,
            "cv_scheme": args.cv_scheme,
            "n_evaluated_folds": len(df),
            "mean_epoch_acc": float(df["epoch_acc"].mean()),
            "std_epoch_acc": float(df["epoch_acc"].std(ddof=0)),
            "mean_patient_acc": float(df["patient_acc"].mean()),
            "std_patient_acc": float(df["patient_acc"].std(ddof=0)),
        }
        summ_path = outdir / f"summary.json"
        summ_path.write_text(json.dumps(summary, indent=2))
        print(f"Summary: {summary}")
        print(f"Summary JSON: {summ_path}")

    if missing:
        miss_path = outdir / f"missing_{args.model}_{args.cv_scheme}.json"
        miss_path.write_text(json.dumps(missing, indent=2))
        print(f"\nMissing folds info saved to: {miss_path}")

    return 0


if __name__ == "__main__":
    main()
