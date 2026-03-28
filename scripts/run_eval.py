# scripts/run_eval.py

import argparse
import json
from pathlib import Path
import numpy as np
from datetime import datetime
from sklearn.metrics import f1_score

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
                    choices=["shallow", "deep4", "eegnet", "tcn", "hybridnet", "biot"])
    ap.add_argument("--cv-scheme", type=str, default="loso", choices=["loso", "groupkfold"])
    ap.add_argument("--cv-n-splits", type=int, default=10)
    ap.add_argument("--cv-fold", type=str, default="all", help="'all' or integer fold index")
    ap.add_argument("--outdir", type=str, default="runs")
    ap.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--dataset-cache", type=str, default=None,
                help="Path to .npz cache produced by data_load.save_dataset_cache")
    ap.add_argument("--exclude-events", type=int, nargs="*", default=None, help="Event codes to exclude to match training data")
    ap.add_argument("--tmin", type=float, default=0.1)
    ap.add_argument("--tmax", type=float, default=0.2)
    args = ap.parse_args()

    if args.dataset_cache:
        print(f"Loading dataset cache: {args.dataset_cache}")
        bundle = data_load.load_dataset_from_cache(args.dataset_cache)
    else:
        print(f"Loading dataset from {args.data_root} ...")
        bundle = data_load.load_dataset(args.data_root, extension=".set", tmin=args.tmin, tmax=args.tmax)
    ds = bundle["dataset"]
    n_chans, n_times = bundle["n_chans"], bundle["n_times"]

    if args.exclude_events:
        print(f"Excluding events: {args.exclude_events}")
        mask = ~np.isin(ds.events, args.exclude_events)
        print(f"  -> Excluding {np.sum(~mask)} samples, keeping {np.sum(mask)}")
        ds.X = ds.X[mask]
        ds.y = ds.y[mask]
        ds.events = ds.events[mask]
        ds.groups = [g for i, g in enumerate(ds.groups) if mask[i]]
        ds.patients = [p for i, p in enumerate(ds.patients) if mask[i]]
        print(f"  -> New dataset size: {len(ds.X)}")

    splits = split.make_splits(ds, scheme=args.cv_scheme, n_splits=args.cv_n_splits, seed=42)

    labels = np.asarray(ds.y, dtype=np.int64)
    uniq = np.unique(labels)
    n_classes = int(len(uniq))
    model_cfg = {"name": args.model, "n_chans": n_chans, "n_times": n_times, "n_classes": n_classes}

    if args.cv_fold.lower() == "all":
        fold_ids = range(len(splits))
    else:
        fold_ids = [int(args.cv_fold)]

    outdir = Path(args.outdir)
    leaderboard = []
    missing = []
    
    # Global accumulators for calculating dataset-wide F1 (concatenated)
    all_y_true_epoch = []
    all_y_pred_epoch = []
    all_y_true_patient = []
    all_y_pred_patient = []

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
        
        # Pop arrays to avoid adding them to the CSV / leaderboard dict
        # and accumulate them for global F1 calculation
        if "y_true_epoch" in res:
            all_y_true_epoch.append(res.pop("y_true_epoch"))
            all_y_pred_epoch.append(res.pop("y_pred_epoch"))
            all_y_true_patient.append(res.pop("y_true_patient"))
            all_y_pred_patient.append(res.pop("y_pred_patient"))

        leaderboard.append({
            "fold": k,
            "run_dir": str(run_dir),
            "checkpoint": str(ckpt),
            **res
        })
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Fold {k} -> "
              f"epoch_acc={res['epoch_acc']:.4f}, epoch_f1={res['epoch_f1']:.4f}, "
              f"patient_acc={res['patient_acc']:.4f}, patient_f1={res['patient_f1']:.4f}")

    if leaderboard:
        import pandas as pd
        df = pd.DataFrame(leaderboard).sort_values("fold")
        lb_path = outdir / f"leaderboard_{args.model}_{args.cv_scheme}.csv"
        df.to_csv(lb_path, index=False)
        print(f"\n Leaderboard written to: {lb_path}")

        # Compute global F1 (concatenated)
        global_epoch_f1 = 0.0
        global_patient_f1 = 0.0
        if all_y_true_epoch:
            glob_y_true_e = np.concatenate(all_y_true_epoch)
            glob_y_pred_e = np.concatenate(all_y_pred_epoch)
            global_epoch_f1 = f1_score(glob_y_true_e, glob_y_pred_e, average='macro', zero_division=0)
            
            glob_y_true_p = np.concatenate(all_y_true_patient)
            glob_y_pred_p = np.concatenate(all_y_pred_patient)
            global_patient_f1 = f1_score(glob_y_true_p, glob_y_pred_p, average='macro', zero_division=0)

        summary = {
            "model": args.model,
            "cv_scheme": args.cv_scheme,
            "n_evaluated_folds": len(df),
            # Per-fold means (legacy/standard CV metric)
            "mean_epoch_acc": float(df["epoch_acc"].mean()),
            "std_epoch_acc": float(df["epoch_acc"].std(ddof=0)),
            "mean_epoch_f1": float(df["epoch_f1"].mean()),
            "std_epoch_f1": float(df["epoch_f1"].std(ddof=0)),
            "mean_patient_acc": float(df["patient_acc"].mean()),
            "std_patient_acc": float(df["patient_acc"].std(ddof=0)),
            "mean_patient_f1": float(df["patient_f1"].mean()),
            "std_patient_f1": float(df["patient_f1"].std(ddof=0)),
            # Global (concatenated) metrics - more representative for F1 in LOOCV
            "global_epoch_f1": float(global_epoch_f1),
            "global_patient_f1": float(global_patient_f1),
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
