from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

from hetmap.build_graph.builder import build_datasets
from hetmap.experiments.config import (
    ExperimentConfig, N2V_TRAINING, W2V_TRAINING, UXC_TRAINING,
)
from hetmap.training.hgt_trainer import few_shot_learning

_METHOD_TRAINING = {
    "n2v":     N2V_TRAINING,
    "w2v":     W2V_TRAINING,
    "file_ft": UXC_TRAINING,
    "file_zs": UXC_TRAINING,
}


def infer_graph_level(config: ExperimentConfig) -> str:
    has_folders = bool(config.graph.folder_edges)
    has_members = bool(config.graph.keep_member_nodes)
    use_agg     = config.graph.pooling != "n2v"
    suffix      = "-Agg" if (has_members and use_agg) else ""
    if has_folders and has_members:
        return f"FFM{suffix}"
    if has_folders:
        return "FF"
    if has_members:
        return f"FM{suffix}"
    return "File_only"


def run_experiments(config: ExperimentConfig) -> pd.DataFrame:
    data_dict = build_datasets(config)
    config.results_dir.mkdir(parents=True, exist_ok=True)
    graph_level  = infer_graph_level(config)
    training_cfg = _METHOD_TRAINING.get(config.graph.pooling, config.training)

    all_results: List[dict] = []
    for dataset_name, data in data_dict.items():
        print(f"\n===== HGT: {dataset_name} (pooling={config.graph.pooling}) =====")
        csv_path   = config.results_dir / f"{dataset_name}.csv"
        preds_path = config.results_dir / f"{dataset_name}_predictions.csv"

        existing   = pd.read_csv(csv_path) if csv_path.exists() else pd.DataFrame()
        done_runs  = set(existing["run_id"].tolist()) if not existing.empty else set()
        rows: List[dict]      = existing.to_dict("records") if not existing.empty else []
        all_preds: List[dict] = (pd.read_csv(preds_path).to_dict("records") if preds_path.exists() else [])

        device = data["file"].x.device

        for run_id in range(training_cfg.num_runs):
            if run_id in done_runs:
                continue

            if config.graph.random_seed is not None:
                np.random.seed(config.graph.random_seed + run_id)
                torch.manual_seed(config.graph.random_seed + run_id)
            file_train, file_test = data.generate_split(
                split_ratio=training_cfg.split_ratio,
                min_pt=training_cfg.min_train_points,
                max_pt=training_cfg.max_train_points,
            )
            _, metrics, preds_rows = few_shot_learning(data, file_train, file_test, training_cfg)
            for r in preds_rows:
                r["run_id"] = run_id
            all_preds.extend(preds_rows)

            row = {
                "run_id":            run_id,
                "dataset":           dataset_name,
                "pooling":           config.graph.pooling,
                "graph_level":       graph_level,
                "split_ratio":       training_cfg.split_ratio,
                "train_size":        int(file_train.numel()),
                "epochs":            training_cfg.epochs,
                "heads":             training_cfg.heads,
                "hidden_channels":   training_cfg.hidden_channels,
                "self_train_rounds": training_cfg.self_train_rounds,
                "threshold":         training_cfg.threshold,
                **{k: round(v * 100, 4) if isinstance(v, float) else v for k, v in metrics.items()},
            }
            rows.append(row)

            if (run_id + 1) % training_cfg.flush_every == 0 or run_id == training_cfg.num_runs - 1:
                pd.DataFrame(rows).to_csv(csv_path, index=False)
                pd.DataFrame(all_preds).to_csv(preds_path, index=False)
                print(
                    f"  run {run_id + 1}/{training_cfg.num_runs} "
                    f"| f1_micro={metrics['f1_micro']:.3f} f1_macro={metrics['f1_macro']:.3f}"
                )

        all_results.extend(rows)

    return pd.DataFrame(all_results)
