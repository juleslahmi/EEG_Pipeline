# scripts/run_train.py  (template)
"""
CLI entrypoint to train one model with a chosen CV scheme.

Usage examples:
  python scripts/run_train.py --data-root /data --model shallow --cv-scheme loso --cv-fold all
  python scripts/run_train.py --data-root /data --model eegnet --cv-scheme groupkfold --cv-n-splits 10 --cv-fold 3
"""

def main():
    # 1) parse args
    #   --data-root
    #   --model  (shallow|deep4|eegnet)
    #   --lr, --weight-decay, --batch-size, --epochs
    #   --cv-scheme (loso|groupkfold)
    #   --cv-n-splits, --cv-fold (all|int), --cv-seed
    #   --outdir, --seed

    # 2) load dataset bundle
    #   bundle = data_load.load_dataset(args.data_root, ext=".set")
    #   ds = bundle["dataset"]; n_chans, n_times = bundle["n_chans"], bundle["n_times"]

    # 3) make splits
    #   splits = split.make_splits(ds, scheme=args.cv_scheme, n_splits=args.cv_n_splits, seed=args.cv_seed)

    # 4) prepare configs (plain dicts)
    #   model_cfg = {"name": args.model, "n_chans": n_chans, "n_times": n_times, "n_classes": 2}
    #   train_cfg = {"lr": args.lr, "weight_decay": args.weight_decay, "batch_size": args.batch_size,
    #                "max_epochs": args.epochs, "scheduler": {"type": "reduce_on_plateau", "patience": 3}}

    # 5) choose folds to run
    #   fold_ids = range(len(splits)) if args.cv_fold == "all" else [int(args.cv_fold)]

    # 6) loop folds → train_one_fold(...)
    #   for k in fold_ids:
    #       run_result = train.train_one_fold(
    #           dataset=ds,
    #           fold_spec=splits[k],
    #           model_cfg=model_cfg,
    #           train_cfg=train_cfg,
    #           outdir=args.outdir,
    #           seed=args.seed,
    #       )
    #       # collect per-fold summaries for printing

    # 7) print final summary (mean patient-acc across folds, etc.)
    return 0

if __name__ == "__main__":
    main()
