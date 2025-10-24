# scripts/aggregate_results.py
import argparse
import json
from pathlib import Path
import pandas as pd

def main():
    ap = argparse.ArgumentParser(description="Aggregate eval summaries into one CSV.")
    ap.add_argument("--root", type=str, default="runs", help="Root runs/ directory")
    args = ap.parse_args()

    root = Path(args.root)
    rows = []

    # Walk all subdirs under runs/, collect summary_*.json
    for summ in root.rglob("summary_*.json"):
        try:
            data = json.loads(summ.read_text())
        except Exception:
            continue
        # Infer experiment tag from parent dir
        exp_dir = summ.parent
        tag = exp_dir.name  # e.g., m=shallow_lr=0.001_wd=0.0_bs=64_ep=30_loso
        # Try to parse hyperparams from tag (simple split on '_')
        meta = {kv.split('=')[0]: kv.split('=')[1] for kv in tag.split('_') if '=' in kv}
        rows.append({
            "exp_dir": str(exp_dir),
            "tag": tag,
            "model": data.get("model", meta.get("m", "")),
            "cv_scheme": data.get("cv_scheme", meta.get("loso", "loso")),
            "n_folds": data.get("n_evaluated_folds", None),
            "mean_epoch_acc": data.get("mean_epoch_acc", None),
            "std_epoch_acc": data.get("std_epoch_acc", None),
            "mean_patient_acc": data.get("mean_patient_acc", None),
            "std_patient_acc": data.get("std_patient_acc", None),
            "lr": meta.get("lr", None),
            "weight_decay": meta.get("wd", None),
            "batch_size": meta.get("bs", None),
            "epochs": meta.get("ep", None),
        })

    if not rows:
        print("No summary_*.json files found under", root)
        return 1

    df = pd.DataFrame(rows)
    # Order columns nicely
    cols = [
        "tag", "exp_dir", "model", "cv_scheme", "n_folds",
        "lr", "weight_decay", "batch_size", "epochs",
        "mean_patient_acc", "std_patient_acc",
        "mean_epoch_acc", "std_epoch_acc",
    ]
    df = df[[c for c in cols if c in df.columns]].copy()
    # Sort best → worst by patient-level performance
    df.sort_values(["mean_patient_acc", "mean_epoch_acc"], ascending=False, inplace=True)

    out_path = root / "master_summary.csv"
    df.to_csv(out_path, index=False)
    print(f" Wrote {out_path} with {len(df)} rows.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
