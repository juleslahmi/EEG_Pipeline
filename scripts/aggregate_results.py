# scripts/aggregate_results.py
import argparse
import json
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd

def parse_tag_meta(tag: str) -> Dict[str, str]:
    """
    Parse simple key=value segments separated by underscores from a folder tag.
    Example: m=shallow_lr=0.001_wd=0.0_bs=64_ep=30_loso
    """
    meta = {}
    for piece in tag.split("_"):
        if "=" in piece:
            k, v = piece.split("=", 1)
            meta[k] = v
    return meta

def safe_get(d: Dict[str, Any], key: str, default=None):
    val = d.get(key, default)
    return val if val is not None else default

def collect_summaries(root: Path, filename: str = "summary.json") -> pd.DataFrame:
    """
    Collect all summary.json files under root (any depth).
    Each row corresponds to one experiment directory that contains the file.
    """
    rows = []
    for summ in root.rglob(filename):
        try:
            data = json.loads(summ.read_text())
        except Exception:
            continue
        exp_dir = summ.parent
        tag = exp_dir.name
        meta = parse_tag_meta(tag)
        rows.append({
            "tag": tag,
            "exp_dir": str(exp_dir),
            "model": safe_get(data, "model", meta.get("m")),
            "cv_scheme": safe_get(data, "cv_scheme", "loso"),
            "n_folds": safe_get(data, "n_evaluated_folds", None),
            "mean_epoch_acc": safe_get(data, "mean_epoch_acc", None),
            "std_epoch_acc": safe_get(data, "std_epoch_acc", None),
            "mean_patient_acc": safe_get(data, "mean_patient_acc", None),
            "std_patient_acc": safe_get(data, "std_patient_acc", None),
            "lr": meta.get("lr", None),
            "weight_decay": meta.get("wd", None),
            "batch_size": meta.get("bs", None),
            "epochs": meta.get("ep", None),
        })
    if not rows:
        raise SystemExit(f"No {filename} files found under {root}")
    df = pd.DataFrame(rows)
    # Nice column order if present
    cols = [
        "tag", "exp_dir", "model", "cv_scheme", "n_folds",
        "lr", "weight_decay", "batch_size", "epochs",
        "mean_patient_acc", "std_patient_acc",
        "mean_epoch_acc", "std_epoch_acc",
    ]
    df = df[[c for c in cols if c in df.columns]].copy()
    return df

def parse_metrics(metrics_str: str) -> List[str]:
    return [m.strip() for m in metrics_str.split(",") if m.strip()]

def main():
    ap = argparse.ArgumentParser(description="Aggregate summaries and produce top-k leaderboards.")
    ap.add_argument("--root", type=str, default="runs", help="Root runs/ directory")
    ap.add_argument("--filename", type=str, default="summary.json", help="Summary filename to look for")
    ap.add_argument("--metrics", type=str,
                    default="mean_patient_acc,mean_epoch_acc",
                    help="Comma-separated metrics for overall ranking (in priority order).")
    ap.add_argument("--ascending", type=str, default="",
                    help="Comma-separated metrics that should be sorted ascending (e.g., loss). Default is descending.")
    ap.add_argument("--top-k", type=int, default=5, help="Top-k to output for each metric and overall.")
    ap.add_argument("--out-prefix", type=str, default="",
                    help="Optional prefix for output CSV files (defaults to none).")
    args = ap.parse_args()

    root = Path(args.root)
    out_prefix = (args.out_prefix + "_") if args.out_prefix else ""

    # 1) Collect all summaries
    df = collect_summaries(root, filename=args.filename)

    # Save the full master summary
    master_path = root / f"{out_prefix}master_summary.csv"
    df.to_csv(master_path, index=False)
    print(f"[INFO] Wrote {master_path} with {len(df)} rows.")

    # 2) Overall multi-metric ranking
    metrics = parse_metrics(args.metrics)
    if not metrics:
        raise SystemExit("No metrics provided for overall ranking (--metrics).")

    asc_set = set(parse_metrics(args.ascending))
    # Determine ascending flags aligned with metrics order
    ascending_flags = [m in asc_set for m in metrics if m in df.columns]

    # Only keep metrics that actually exist in df
    usable_metrics = [m for m in metrics if m in df.columns]
    if not usable_metrics:
        raise SystemExit(f"None of the requested metrics exist in data: {metrics}")

    overall_sorted = df.sort_values(usable_metrics, ascending=ascending_flags if ascending_flags else False)
    overall_topk = overall_sorted.head(args.top_k).copy()
    overall_path = root / f"{out_prefix}topk_overall.csv"
    overall_topk.to_csv(overall_path, index=False)
    print(f"[INFO] Wrote overall top-{args.top_k} → {overall_path} (by {', '.join(usable_metrics)})")

    # 3) Per-metric leaderboards
    for m in usable_metrics:
        asc = m in asc_set
        if m not in df.columns:
            continue
        topk_m = df.sort_values(m, ascending=asc).head(args.top_k).copy()
        out_m = root / f"{out_prefix}topk_by_{m}.csv"
        topk_m.to_csv(out_m, index=False)
        print(f"[INFO] Wrote top-{args.top_k} by {m} ({'asc' if asc else 'desc'}) → {out_m}")

    # 4) Optional: print small console summary
    print("\n=== Overall top-k ===")
    print(overall_topk[["tag", "exp_dir"] + usable_metrics].to_string(index=False))

    return 0

if __name__ == "__main__":
    main()
