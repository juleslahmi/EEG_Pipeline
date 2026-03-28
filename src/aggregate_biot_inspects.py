#!/usr/bin/env python3

from pathlib import Path
import argparse
import numpy as np
import json
import sys
import matplotlib.pyplot as plt


def find_inspect_dirs(root: Path):
    return sorted([p for p in root.iterdir() if p.is_dir()])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--inspect-root', type=str, required=True)
    ap.add_argument('--out', type=str, default=None)
    args = ap.parse_args(argv)

    root = Path(args.inspect_root)
    if not root.exists():
        print('inspect_root not found:', root)
        sys.exit(2)

    dirs = find_inspect_dirs(root)
    sal_list = []
    ch_list = []
    time_list = []
    valid_dirs = []
    for d in dirs:
        sfile = d / 'saliency_mean.npy'
        chfile = d / 'channel_importance.npy'
        tfile = d / 'time_importance.npy'
        if sfile.exists() and chfile.exists() and tfile.exists():
            sal = np.load(sfile)
            ch = np.load(chfile)
            t = np.load(tfile)
            sal_list.append(sal)
            ch_list.append(ch)
            time_list.append(t)
            valid_dirs.append(str(d))
        else:
            print('Skipping', d, 'missing required files')

    if not sal_list:
        print('No valid inspection outputs found in', root)
        sys.exit(1)

    sal_stack = np.stack(sal_list, axis=0)  # (n_folds, n_chans, n_times)
    ch_stack = np.stack(ch_list, axis=0)   # (n_folds, n_chans)
    t_stack = np.stack(time_list, axis=0)  # (n_folds, n_times)

    sal_mean = sal_stack.mean(axis=0)
    sal_std = sal_stack.std(axis=0)

    ch_mean = ch_stack.mean(axis=0)
    ch_std = ch_stack.std(axis=0)

    t_mean = t_stack.mean(axis=0)
    t_std = t_stack.std(axis=0)

    # Save aggregates
    out_dir = root
    np.save(out_dir / 'aggregate_saliency_mean.npy', sal_mean)
    np.save(out_dir / 'aggregate_saliency_std.npy', sal_std)
    np.save(out_dir / 'aggregate_channel_mean.npy', ch_mean)
    np.save(out_dir / 'aggregate_channel_std.npy', ch_std)
    np.save(out_dir / 'aggregate_time_mean.npy', t_mean)
    np.save(out_dir / 'aggregate_time_std.npy', t_std)

    # Top channels across folds by mean
    top_idx = np.argsort(-ch_mean)[:10].tolist()
    top_vals = [float(ch_mean[i]) for i in top_idx]

    summary = {
        'n_folds': int(len(sal_list)),
        'inspect_dirs': valid_dirs,
        'top_channels_mean': top_idx,
        'top_channel_values_mean': top_vals,
        'channel_mean_shape': ch_mean.shape,
        'time_mean_shape': t_mean.shape,
    }

    out_path = Path(args.out) if args.out else out_dir / 'aggregate_summary.json'
    out_path.write_text(json.dumps(summary, indent=2))
    print('Wrote aggregate summary to', out_path)

    # Save heatmap of mean saliency
    plt.figure(figsize=(10, 6))
    plt.imshow(sal_mean, aspect='auto', origin='lower')
    plt.colorbar(label='mean |grad| (across folds and samples)')
    plt.xlabel('time')
    plt.ylabel('channel')
    plt.title('Aggregate BIOT saliency mean (channel x time)')
    plt.tight_layout()
    plt.savefig(out_dir / 'aggregate_saliency_mean.png', dpi=150)
    plt.close()

    print('Saved aggregate heatmap to', out_dir / 'aggregate_saliency_mean.png')


if __name__ == '__main__':
    main()
