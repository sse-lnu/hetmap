"""predict_example.py — map files that have no ground truth (the realistic use case).

HetMap's benchmark runners (run_experiments.py, hgt_runner.py, mlp_runner.py) assume
every file already has a ground-truth Module, because that's what reproducing the
paper's numbers requires. In practice you often have the opposite: a small set of
already-mapped files (L) and a larger set of files with no known module yet — new
files, or files nobody has classified. This script shows that workflow directly:

  1. Load one project and treat only a small labeled subset as "known" (simulates a
     partially-mapped codebase — in a real project you'd load your own partial mapping
     instead of blanking out an existing one).
  2. Train HGT on that labeled subset (`file_train`), exactly as during a normal run.
  3. Call few_shot_learning(..., has_gt=False) to get predictions for the files with
     no ground truth — no metrics are computed (there's nothing to score against),
     just a {file_path, pred_module, confidence} row per file.

Run:
    python predict_example.py --dataset SH-3D
"""
from __future__ import annotations

import argparse

import numpy as np

from hetmap.build_graph.hetero_data import HeterogeneousData
from hetmap.process_data.registry import load_datasets
from hetmap.experiments.config import GraphConfig, W2V_TRAINING
from hetmap.training.hgt_trainer import few_shot_learning


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="SH-3D", help="HetMap dataset name (see hetmap/process_data/registry.py).")
    ap.add_argument("--unlabeled-frac", type=float, default=0.3,
                    help="Fraction of files to treat as having no ground truth (simulated).")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    datasets, dependencies = load_datasets([args.dataset])
    df      = datasets[args.dataset].copy()
    df_dep  = dependencies[args.dataset]

    # Simulate a partially-mapped project: blank out Module for a random subset of
    # files. In a real project, skip this — your own df would already have Module
    # set only for the files you've mapped so far, and NaN/missing for the rest.
    rng   = np.random.default_rng(args.seed)
    files = df["File"].dropna().unique()
    n_unlabeled = int(len(files) * args.unlabeled_frac)
    unlabeled_files = set(rng.choice(files, size=n_unlabeled, replace=False).tolist())
    df.loc[df["File"].isin(unlabeled_files), "Module"] = np.nan
    print(f"[{args.dataset}] {len(files)} files total, "
          f"{n_unlabeled} treated as unmapped (no ground truth).")

    data = HeterogeneousData(
        df, df_dep,
        GraphConfig(pooling="w2v", folder_edges=True, keep_member_nodes=True),
        dataset_name=args.dataset,
    )

    # file_train: labeled subset (L) — always required, exactly like a normal run.
    file_train, _ = data.generate_split(
        split_ratio=W2V_TRAINING.split_ratio,
        min_pt=W2V_TRAINING.min_train_points,
        max_pt=W2V_TRAINING.max_train_points,
    )
    # file_test: the genuinely unmapped files — no ground truth, only predictions.
    file_test = (~data.has_gt).nonzero(as_tuple=True)[0].to(file_train.device)

    print(f"train (labeled) = {file_train.numel()} files, "
          f"predict (unmapped) = {file_test.numel()} files")

    _, results, preds_rows = few_shot_learning(
        data, file_train, file_test, W2V_TRAINING, has_gt=False,
    )
    assert results == {}, "has_gt=False should skip metric computation entirely"

    print(f"\n{'file_path':50s} {'pred_module':15s} confidence")
    for row in preds_rows[:20]:
        print(f"{row['file_path']:50s} {row['pred_module']:15s} {row['confidence']:.3f}")
    if len(preds_rows) > 20:
        print(f"... ({len(preds_rows) - 20} more)")


if __name__ == "__main__":
    main()
