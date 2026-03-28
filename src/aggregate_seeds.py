#!/usr/bin/env python3
"""
Aggregate epoch and patient accuracy across multiple seed runs.

Usage:
  python src/aggregate_seeds.py --runs-root runs --base-tag "m=biot_lr=..._ep=..._loso" --model biot --cv loso

This script looks for directories under `--runs-root` that start with `--base-tag` (they will
typically have `_seed=...` appended), reads each run's `leaderboard_{model}_{cv}.csv` file and
computes per-seed mean epoch/patient accuracy, then reports the across-seed mean and std.
"""
import argparse
from pathlib import Path
import pandas as pd
import json
import sys


def find_run_dirs(runs_root: Path, base_tag: str):
    return sorted([p for p in runs_root.iterdir() if p.is_dir() and p.name.startswith(base_tag)])


def read_leaderboard(run_dir: Path, model: str, cv: str):
    lb_path = run_dir / f"leaderboard_{model}_{cv}.csv"
    if lb_path.exists():
        return pd.read_csv(lb_path)
    metrics_files = list(run_dir.rglob('metrics.csv'))
    if metrics_files:
        dfs = [pd.read_csv(p) for p in metrics_files]
        return pd.concat(dfs, ignore_index=True)
    return None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs-root', type=str, default='runs')
    ap.add_argument('--base-tag', type=str, required=True,
                    help='Base tag prefix used for runs (without _seed=...)')
    ap.add_argument('--model', type=str, required=True)
    ap.add_argument('--cv', type=str, required=True)
    ap.add_argument('--out', type=str, default=None, help='Optional output JSON path')
    args = ap.parse_args(argv)

    runs_root = Path(args.runs_root)
    if not runs_root.exists():
        print(f"Runs root not found: {runs_root}")
        sys.exit(2)

    # Check for nested structure first: runs_root/base_tag/seed=X
    nested_dir = runs_root / args.base_tag
    if nested_dir.exists() and nested_dir.is_dir():
        run_dirs = sorted([p for p in nested_dir.iterdir() if p.is_dir()])
    else:
        # Fallback to flat structure: runs_root/base_tag_seed=X
        run_dirs = find_run_dirs(runs_root, args.base_tag)

    if not run_dirs:
        print(f"No run dirs found starting with: {args.base_tag} in {runs_root}")
        sys.exit(1)

    per_seed = []
    for d in run_dirs:
        df = read_leaderboard(d, args.model, args.cv)
        if df is None:
            print(f"Warning: no leaderboard or metrics found in {d}, skipping")
            continue
        
        # defaults
        epoch_mean = 0.0
        patient_mean = 0.0
        epoch_f1_mean = 0.0
        patient_f1_mean = 0.0

        if 'epoch_acc' in df.columns and 'patient_acc' in df.columns:
            epoch_mean = float(df['epoch_acc'].mean())
            patient_mean = float(df['patient_acc'].mean())
            epoch_f1_mean = float(df['epoch_f1'].mean()) if 'epoch_f1' in df.columns else 0.0
            patient_f1_mean = float(df['patient_f1'].mean()) if 'patient_f1' in df.columns else 0.0
        else:
            possible_epoch = [c for c in df.columns if ('epoch' in c.lower() or 'valid' in c.lower()) and 'acc' in c.lower()]
            possible_patient = [c for c in df.columns if 'patient' in c.lower() and 'acc' in c.lower()]
            if possible_epoch and possible_patient:
                epoch_mean = float(df[possible_epoch[0]].mean())
                patient_mean = float(df[possible_patient[0]].mean())
            else:
                print(f"Could not find epoch/patient acc columns in {d}, skipping")
                continue
        
        summ_path = d / "summary.json"
        if summ_path.exists():
            try:
                sdata = json.loads(summ_path.read_text())
                if 'global_epoch_f1' in sdata:
                    epoch_f1_mean = float(sdata['global_epoch_f1'])
                if 'global_patient_f1' in sdata:
                    patient_f1_mean = float(sdata['global_patient_f1'])
            except Exception:
                pass

        per_seed.append({
            'run_dir': str(d), 
            'epoch_mean': epoch_mean, 
            'patient_mean': patient_mean,
            'epoch_f1_mean': epoch_f1_mean,
            'patient_f1_mean': patient_f1_mean
        })

    if not per_seed:
        print('No valid seeds found to aggregate.')
        sys.exit(1)

    stats = pd.DataFrame(per_seed)
    out = {
        'n_seeds': len(stats),
        'epoch_mean_across_seeds': float(stats['epoch_mean'].mean()),
        'epoch_std_across_seeds': float(stats['epoch_mean'].std(ddof=0)),
        'epoch_f1_mean_across_seeds': float(stats['epoch_f1_mean'].mean()),
        'epoch_f1_std_across_seeds': float(stats['epoch_f1_mean'].std(ddof=0)),
        'patient_mean_across_seeds': float(stats['patient_mean'].mean()),
        'patient_std_across_seeds': float(stats['patient_mean'].std(ddof=0)),
        'patient_f1_mean_across_seeds': float(stats['patient_f1_mean'].mean()),
        'patient_f1_std_across_seeds': float(stats['patient_f1_mean'].std(ddof=0)),
        'per_seed': per_seed,
    }

    out_path = Path(args.out) if args.out else runs_root / f'summary_seeds_{args.base_tag.replace("/","_")}.json'
    out_path.write_text(json.dumps(out, indent=2))
    print(f'Wrote summary to: {out_path}')
    print('Summary:')
    print(json.dumps({
        'n_seeds': out['n_seeds'],
        'epoch_mean': out['epoch_mean_across_seeds'],
        'epoch_std': out['epoch_std_across_seeds'],
        'epoch_f1_mean': out['epoch_f1_mean_across_seeds'],
        'epoch_f1_std': out['epoch_f1_std_across_seeds'],
        'patient_mean': out['patient_mean_across_seeds'],
        'patient_std': out['patient_std_across_seeds'],
        'patient_f1_mean': out['patient_f1_mean_across_seeds'],
        'patient_f1_std': out['patient_f1_std_across_seeds'],
    }, indent=2))


if __name__ == '__main__':
    main()
