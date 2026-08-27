from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

from hetmap.process_data.registry import load_datasets
from hetmap.experiments.config import MLPExperimentConfig, GraphConfig
from hetmap.node_features.unixcoder.loader import load_embeddings
from hetmap.training.mlp_trainer import FileLevelDataset, few_shot_learning_mlp
from hetmap.training.hgt_trainer import resolve_device


def build_mlp_datasets(config: MLPExperimentConfig) -> Dict[str, FileLevelDataset]:
    raw_datasets, dependencies = load_datasets(
        config.include_datasets, config.exclude_datasets
    )
    result: Dict[str, FileLevelDataset] = {}
    for name, df in raw_datasets.items():
        try:
            if config.pooling == "n2v":
                from hetmap.node_features.n2v import load_or_train_n2v
                emb_matrix, dep_keys = load_or_train_n2v(
                    name, df, dependencies[name], GraphConfig()
                )
            elif config.pooling == "w2v":
                from hetmap.node_features.w2v import load_or_train_w2v_file_embs
                emb_matrix, dep_keys = load_or_train_w2v_file_embs(
                    name, df, dependencies[name], GraphConfig()
                )
            else:
                emb_matrix, dep_keys = load_embeddings(
                    name, pooling=config.pooling, emb_dir=config.emb_dir
                )
            print(f"  [{name}] embeddings: {emb_matrix.shape}  pooling={config.pooling}")
        except FileNotFoundError as exc:
            print(f"  [MLP] {exc} — skipping '{name}'.")
            continue
        result[name] = FileLevelDataset(df, emb_matrix, dep_keys)
    return result


def run_experiments(config: MLPExperimentConfig) -> pd.DataFrame:
    datasets = build_mlp_datasets(config)
    config.results_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device()

    all_results: List[dict] = []
    for dataset_name, dataset in datasets.items():
        print(f"\n===== MLP: {dataset_name} (pooling={config.pooling}) =====")
        csv_path   = config.results_dir / f"{dataset_name}.csv"
        preds_path = config.results_dir / f"{dataset_name}_predictions.csv"

        existing   = pd.read_csv(csv_path) if csv_path.exists() else pd.DataFrame()
        done_runs  = set(existing["run_id"].tolist()) if not existing.empty else set()
        rows: List[dict]      = existing.to_dict("records") if not existing.empty else []
        all_preds: List[dict] = (pd.read_csv(preds_path).to_dict("records") if preds_path.exists() else [])

        for run_id in range(config.mlp.num_runs):
            if run_id in done_runs:
                continue

            seed = config.mlp.random_seed
            if seed is not None:
                np.random.seed(seed + run_id)
                torch.manual_seed(seed + run_id)
            file_train, file_test = dataset.generate_split(
                split_ratio=config.mlp.split_ratio,
                min_pt=config.mlp.min_train_points,
                max_pt=config.mlp.max_train_points,
                device=device,
            )

            metrics, preds_rows = few_shot_learning_mlp(dataset, file_train, file_test, config.mlp)
            for r in preds_rows:
                r["run_id"] = run_id
            all_preds.extend(preds_rows)

            row = {
                "run_id":            run_id,
                "dataset":           dataset_name,
                "pooling":           config.pooling,
                "split_ratio":       config.mlp.split_ratio,
                "train_size":        int(file_train.numel()),
                "epochs":            config.mlp.epochs,
                "hidden_channels":   config.mlp.hidden_channels,
                "num_layers":        config.mlp.num_layers,
                "self_train_rounds": config.mlp.self_train_rounds,
                "threshold":         config.mlp.threshold,
                "emb_coverage":      round(dataset.emb_coverage * 100, 2),
                **{k: round(v * 100, 4) if isinstance(v, float) else v for k, v in metrics.items()},
            }
            rows.append(row)

            if (run_id + 1) % config.mlp.flush_every == 0 or run_id == config.mlp.num_runs - 1:
                pd.DataFrame(rows).to_csv(csv_path, index=False)
                pd.DataFrame(all_preds).to_csv(preds_path, index=False)
                print(
                    f"  run {run_id + 1}/{config.mlp.num_runs} "
                    f"| f1_micro={metrics['f1_micro']:.3f} f1_macro={metrics['f1_macro']:.3f}"
                )

        all_results.extend(rows)

    return pd.DataFrame(all_results)
