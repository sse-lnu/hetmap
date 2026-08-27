from __future__ import annotations

import gc
from typing import Dict

import numpy as np

from hetmap.process_data.registry import load_datasets
from hetmap.experiments.config import ExperimentConfig
from hetmap.node_features.unixcoder.loader import load_embeddings
from .hetero_data import UniXcoderHeterogeneousData


def build_datasets(config: ExperimentConfig) -> Dict[str, UniXcoderHeterogeneousData]:
    datasets, dependencies = load_datasets(
        config.include_datasets, config.exclude_datasets
    )

    result: Dict[str, UniXcoderHeterogeneousData] = {}
    for hetmap_name, df in datasets.items():
        if config.graph.pooling == "w2v":
            from hetmap.build_graph.hetero_data import HeterogeneousData
            print(f"  [{hetmap_name}] W2V features")
            result[hetmap_name] = HeterogeneousData(
                df, dependencies[hetmap_name], config.graph, dataset_name=hetmap_name
            )
            gc.collect()
            continue

        if config.graph.pooling == "n2v":
            from hetmap.node_features.n2v import load_or_train_n2v
            emb_matrix, dep_keys = load_or_train_n2v(
                hetmap_name, df, dependencies[hetmap_name], config.graph
            )
            mem_emb_matrix = emb_matrix if config.graph.keep_member_nodes else None
            mem_dep_keys   = dep_keys   if config.graph.keep_member_nodes else None
            result[hetmap_name] = UniXcoderHeterogeneousData(
                df, dependencies[hetmap_name], config.graph,
                emb_matrix, dep_keys, mem_emb_matrix, mem_dep_keys,
            )
            gc.collect()
            continue

        if config.graph.pooling == "constant":
            unique_files = df["File"].dropna().drop_duplicates().tolist()
            emb_matrix   = np.zeros((len(unique_files), 256), dtype=np.float32)
            dep_keys     = unique_files
            print(f"  [{hetmap_name}] structure-only (zero-init 256-d, {len(unique_files)} files)")
        else:
            try:
                emb_matrix, dep_keys = load_embeddings(
                    hetmap_name, pooling=config.graph.pooling, emb_dir=config.graph.emb_dir
                )
                print(f"  [{hetmap_name}] file emb: {emb_matrix.shape}  pooling={config.graph.pooling}")
            except FileNotFoundError as exc:
                print(f"  [UniXcoder] {exc} — skipping '{hetmap_name}'.")
                continue

        mem_emb_matrix, mem_dep_keys = None, None
        if config.graph.keep_member_nodes and config.graph.pooling != "constant":
            # frozen file pooling ("file", "file_pkg") → frozen member ("member")
            # fine-tuned file pooling ("file_ft", "file_pkg_ft") → fine-tuned member ("member_ft")
            mem_pooling = "member_ft" if config.graph.pooling.endswith("_ft") else "member"
            try:
                mem_emb_matrix, mem_dep_keys = load_embeddings(
                    hetmap_name, pooling=mem_pooling, emb_dir=config.graph.emb_dir
                )
                print(f"  [{hetmap_name}] member emb: {mem_emb_matrix.shape}")
            except FileNotFoundError:
                print(f"  [{hetmap_name}] no {mem_pooling} embeddings — members will copy file emb.")

        result[hetmap_name] = UniXcoderHeterogeneousData(
            df, dependencies[hetmap_name], config.graph,
            emb_matrix, dep_keys, mem_emb_matrix, mem_dep_keys,
        )
        del emb_matrix, mem_emb_matrix
        gc.collect()

    return result
