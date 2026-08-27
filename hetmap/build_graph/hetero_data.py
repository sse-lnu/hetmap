from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from torch_geometric.data import HeteroData

from hetmap.process_data.preprocessing import Preprocessor
from hetmap.experiments.config import GraphConfig
from hetmap.node_features.w2v import build_w2v_features

_W2V_CACHE_DIR = Path("data/w2v_emb")


class HeterogeneousData(HeteroData):
    def __init__(self, df: pd.DataFrame, df_dep: pd.DataFrame, config: GraphConfig,
                 dataset_name: str = ""):
        super().__init__()
        self.config       = config
        self.dataset_name = dataset_name

        prep = Preprocessor(df.copy(), df_dep.copy(), config)
        prep.preprocess()

        self.df            = prep.df_files.copy()
        self.df_dep_raw    = prep.df_dep.copy()
        self.df_dep_file   = prep.df_dep_file.copy()
        self.df_members    = prep.df_members.copy()

        self.file_tokens                 = prep.file_tokens
        self.folder_hierarchy            = prep.folder_hierarchy
        self.folder_tokens               = prep.folder_tokens
        self.file_folder_tokens          = prep.file_folder_tokens
        self.member_tokens               = prep.member_tokens
        self.member_file_map             = prep.member_file_map
        self.file_to_aggregate_member_id = prep.file_to_aggregate_member_id

        self.file_vocab   = prep.file_dict   or []
        self.folder_vocab = prep.folder_dict or []
        self.member_vocab = prep.member_dict or []

        self.file_id:      Dict[str, int]          = {}
        self.folder_id:    Dict[str, int]          = {}
        self.member_id:    Dict[int, int]          = {}
        self.label_encoder: Optional[LabelEncoder] = None
        self.num_classes:   Optional[int]          = None

        self.use_member_graph = bool(config.keep_member_nodes and len(self.member_tokens) > 0)

        self._build_ids()
        self._build_edges()
        self._create_node_features()
        self._add_file_labels()

    @staticmethod
    def _as_bool(value) -> bool:
        if isinstance(value, bool):
            return value
        if pd.isna(value):
            return False
        return str(value).strip().lower() in {"true", "1", "yes", "y", "t"}

    def _build_ids(self) -> None:
        files = self.df["File"].dropna().astype(str).drop_duplicates().tolist()
        self.file_id = {f: i for i, f in enumerate(files)}

        folders = sorted({node for levels in self.folder_hierarchy.values() for node in levels})
        self.folder_id = {node: i for i, node in enumerate(folders)}

        if self.use_member_graph:
            self.member_id = {mid: i for i, mid in enumerate(sorted(self.member_tokens.keys()))}
        else:
            self.member_id = {}

    def _vectorize_binary(self, texts: List[str], vocabulary: List[str]) -> np.ndarray:
        if not vocabulary:
            return np.zeros((len(texts), 0), dtype=np.float32)
        vec = CountVectorizer(binary=True, vocabulary=vocabulary)
        return vec.transform(texts).toarray().astype(np.float32)

    def _empty_or_constant(self, num_nodes: int) -> np.ndarray:
        if self.config.add_constant_feature_if_empty:
            return np.ones((num_nodes, 1), dtype=np.float32)
        return np.zeros((num_nodes, 0), dtype=np.float32)

    def _stack_features(self, parts: List[np.ndarray], num_nodes: int) -> np.ndarray:
        valid = [p for p in parts if p.size > 0 and p.shape[1] > 0]
        if not valid:
            return self._empty_or_constant(num_nodes)
        return np.hstack(valid).astype(np.float32)

    @staticmethod
    def _l2_normalize(x: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        return x / np.where(norms > 0, norms, 1.0)

    def _create_node_features(self) -> None:
        folder_x: Optional[np.ndarray] = None
        if self.config.folder_edges or self.config.loc_features:
            folder_nodes = list(self.folder_id.keys())
            folder_texts = [" ".join(self.folder_tokens.get(n, [])) for n in folder_nodes]
            folder_x = self._vectorize_binary(folder_texts, self.folder_vocab)
            if folder_x.shape[1] == 0:
                folder_x = self._empty_or_constant(len(folder_nodes))
            if self.config.folder_edges:
                self["folder"].x = torch.tensor(folder_x, dtype=torch.float)

        file_nodes  = list(self.file_id.keys())
        file_parts: List[np.ndarray] = []

        if self.config.use_w2v_features:
            file_texts = [" ".join(self.file_tokens.get(f, [])) for f in file_nodes]
            name = self.dataset_name
            if name:
                _fe = _W2V_CACHE_DIR / f"{name}_w2v_file_emb.npy"
                if _fe.exists():
                    _cached = np.load(_fe)
                    if _cached.shape[0] == len(file_nodes):
                        print(f"  [{name}] W2V file embs cached — loading")
                        file_w2v = _cached
                    else:
                        print(f"  [{name}] W2V file cache shape mismatch ({_cached.shape[0]} vs {len(file_nodes)}) — retraining")
                        file_w2v = None
                else:
                    file_w2v = None
                if file_w2v is None:
                    file_w2v = build_w2v_features(
                        file_texts, self.file_vocab,
                        vector_size=self.config.w2v_vector_size, window=self.config.w2v_window,
                        min_count=self.config.w2v_min_count, sg=self.config.w2v_sg,
                        workers=self.config.w2v_workers, epochs=self.config.w2v_epochs,
                        seed=self.config.random_seed,
                    )
                    _W2V_CACHE_DIR.mkdir(parents=True, exist_ok=True)
                    np.save(_fe, file_w2v)
                    (_W2V_CACHE_DIR / f"{name}_w2v_file_keys.txt").write_text(
                        "\n".join(file_nodes), encoding="utf-8"
                    )
            else:
                file_w2v = build_w2v_features(
                    file_texts, self.file_vocab,
                    vector_size=self.config.w2v_vector_size, window=self.config.w2v_window,
                    min_count=self.config.w2v_min_count, sg=self.config.w2v_sg,
                    workers=self.config.w2v_workers, epochs=self.config.w2v_epochs,
                    seed=self.config.random_seed,
                )
            file_parts.append(file_w2v)
        if self.config.loc_features and folder_x is not None and folder_x.shape[1] > 0:
            file_folder_agg = np.zeros((len(file_nodes), folder_x.shape[1]), dtype=np.float32)
            for i, f in enumerate(file_nodes):
                for fn in self.folder_hierarchy.get(f, []):
                    if fn in self.folder_id:
                        file_folder_agg[i] += folder_x[self.folder_id[fn]]
            file_parts.append(file_folder_agg)

        self["file"].x = torch.tensor(self._stack_features(file_parts, len(file_nodes)), dtype=torch.float)

        if self.use_member_graph:
            member_nodes  = list(self.member_id.keys())
            member_parts: List[np.ndarray] = []
            if self.config.use_w2v_features:
                member_texts = [" ".join(self.member_tokens.get(mid, [])) for mid in member_nodes]
                name = self.dataset_name
                if name:
                    _me = _W2V_CACHE_DIR / f"{name}_w2v_member_emb.npy"
                    if _me.exists():
                        _cached_m = np.load(_me)
                        if _cached_m.shape[0] == len(member_nodes):
                            print(f"  [{name}] W2V member embs cached — loading")
                            member_w2v = _cached_m
                        else:
                            print(f"  [{name}] W2V member cache shape mismatch ({_cached_m.shape[0]} vs {len(member_nodes)}) — retraining")
                            member_w2v = None
                    else:
                        member_w2v = None
                    if member_w2v is None:
                        member_w2v = build_w2v_features(
                            member_texts, self.member_vocab,
                            vector_size=self.config.w2v_vector_size, window=self.config.w2v_window,
                            min_count=self.config.w2v_min_count, sg=self.config.w2v_sg,
                            workers=self.config.w2v_workers, epochs=self.config.w2v_epochs,
                            seed=self.config.random_seed,
                        )
                        _W2V_CACHE_DIR.mkdir(parents=True, exist_ok=True)
                        np.save(_me, member_w2v)
                        (_W2V_CACHE_DIR / f"{name}_w2v_member_keys.txt").write_text(
                            "\n".join(str(m) for m in member_nodes), encoding="utf-8"
                        )
                else:
                    member_w2v = build_w2v_features(
                        member_texts, self.member_vocab,
                        vector_size=self.config.w2v_vector_size, window=self.config.w2v_window,
                        min_count=self.config.w2v_min_count, sg=self.config.w2v_sg,
                        workers=self.config.w2v_workers, epochs=self.config.w2v_epochs,
                        seed=self.config.random_seed,
                    )
                member_parts.append(member_w2v)
            self["member"].x = torch.tensor(self._stack_features(member_parts, len(member_nodes)), dtype=torch.float)

    @staticmethod
    def _is_real_label(v) -> bool:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return False
        s = str(v).strip()
        return bool(s) and s.lower() not in {"nan", "none", "unmapped", "unknown", ""}

    def _add_file_labels(self) -> None:
        # A file has real ground truth only if its Module survived preprocessing —
        # files without GT (has_gt=False) get label -1 rather than a fake class, so
        # they can still be predicted on without contaminating num_classes/training.
        real_mask = self.df["Module"].map(self._is_real_label)
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.df.loc[real_mask, "Module"].astype(str) if real_mask.any() else ["__none__"])

        self.df["Label"] = -1
        if real_mask.any():
            self.df.loc[real_mask, "Label"] = self.label_encoder.transform(
                self.df.loc[real_mask, "Module"].astype(str)
            )
        file_label_map  = self.df.set_index("File")["Label"].to_dict()
        file_gt_map     = self.df.set_index("File")["Module"].map(self._is_real_label).to_dict()
        file_nodes      = list(self.file_id.keys())
        file_labels     = np.array([int(file_label_map.get(f, -1)) for f in file_nodes], dtype=np.int64)
        self["file"].y  = torch.tensor(file_labels, dtype=torch.long)
        self.has_gt     = torch.tensor(
            [bool(file_gt_map.get(f, False)) for f in file_nodes], dtype=torch.bool
        )
        self.num_classes = int(len(self.label_encoder.classes_))

        if self.use_member_graph:
            member_nodes = list(self.member_id.keys())
            member_labels = np.array(
                [int(file_label_map.get(self.member_file_map.get(mid), -1)) for mid in member_nodes],
                dtype=np.int64,
            )
            self["member"].y = torch.tensor(member_labels, dtype=torch.long)

    def _build_edges(self) -> None:
        self._build_folder_edges()
        self._build_member_containment_edges()
        self._build_dependency_edges()

    def _build_folder_edges(self) -> None:
        if not self.config.folder_edges:
            return
        parent_child: set = set()
        for levels in self.folder_hierarchy.values():
            for i in range(1, len(levels)):
                p, c = levels[i - 1], levels[i]
                if p in self.folder_id and c in self.folder_id:
                    parent_child.add((self.folder_id[p], self.folder_id[c]))
        if parent_child:
            src, tgt = zip(*sorted(parent_child))
            self["folder", "parent_of", "folder"].edge_index = torch.tensor([src, tgt], dtype=torch.long)
        else:
            self["folder", "parent_of", "folder"].edge_index = torch.empty((2, 0), dtype=torch.long)

        contains: set = set()
        for fpath, levels in self.folder_hierarchy.items():
            if levels:
                leaf = levels[-1]
                if leaf in self.folder_id and fpath in self.file_id:
                    contains.add((self.folder_id[leaf], self.file_id[fpath]))
        if contains:
            src, tgt = zip(*sorted(contains))
            self["folder", "contains", "file"].edge_index = torch.tensor([src, tgt], dtype=torch.long)
        else:
            self["folder", "contains", "file"].edge_index = torch.empty((2, 0), dtype=torch.long)

    def _build_member_containment_edges(self) -> None:
        if not self.use_member_graph:
            return
        contains: set = set()
        for mid, mid_idx in self.member_id.items():
            f = self.member_file_map.get(mid)
            if f in self.file_id:
                contains.add((mid_idx, self.file_id[f]))
        if contains:
            src, tgt = zip(*sorted(contains))
            self["member", "belongs_to", "file"].edge_index = torch.tensor([src, tgt], dtype=torch.long)
        else:
            self["member", "belongs_to", "file"].edge_index = torch.empty((2, 0), dtype=torch.long)

    def _is_member_endpoint(self, r, side: str) -> bool:
        if not self.use_member_graph:
            return False
        member_id   = getattr(r, f"{side}_Member_ID", np.nan)
        member_type = str(getattr(r, f"{side}_Member_Type", "file"))
        return pd.notna(member_id) and member_type != "file"

    def _endpoint(self, r, side: str) -> Tuple[Optional[str], Optional[int]]:
        file_name = str(getattr(r, f"{side}_File"))
        if self._is_member_endpoint(r, side):
            member_key = self.file_to_aggregate_member_id.get(file_name)
            member_idx = self.member_id.get(member_key)
            if member_idx is not None:
                return "member", member_idx
        file_idx = self.file_id.get(file_name)
        if file_idx is not None:
            return "file", file_idx
        return None, None

    def _build_dependency_edges(self) -> None:
        dep_edges = defaultdict(lambda: [[], []])
        if not self.use_member_graph:
            for r in self.df_dep_file.itertuples(index=False):
                src_idx = self.file_id.get(str(r.Source_File))
                tgt_idx = self.file_id.get(str(r.Target_File))
                if src_idx is None or tgt_idx is None:
                    continue
                rel = ("file", str(r.Dependency_Type), "file")
                dep_edges[rel][0].append(src_idx)
                dep_edges[rel][1].append(tgt_idx)
        else:
            for r in self.df_dep_raw.itertuples(index=False):
                src_type, src_idx = self._endpoint(r, "Source")
                tgt_type, tgt_idx = self._endpoint(r, "Target")
                if src_type is None or tgt_type is None:
                    continue
                rel = (src_type, str(r.Dependency_Type), tgt_type)
                dep_edges[rel][0].append(src_idx)
                dep_edges[rel][1].append(tgt_idx)
        for rel, (srcs, tgts) in dep_edges.items():
            self[rel].edge_index = torch.tensor([srcs, tgts], dtype=torch.long)

    def generate_split(self, split_ratio: float = 0.05, min_pt: int = 30, max_pt: Optional[int] = None):
        # Only files with real ground truth are eligible for the labeled/test split —
        # files without GT (has_gt=False) are never auto-sampled into either set.
        device      = self["file"].x.device
        all_labels  = self["file"].y.cpu().numpy()
        has_gt      = self.has_gt.cpu().numpy() if hasattr(self, "has_gt") else np.ones_like(all_labels, dtype=bool)
        labeled_idx = np.where(has_gt)[0]
        labels      = all_labels[labeled_idx]
        n           = len(labeled_idx)
        shuffled    = labeled_idx.copy()
        np.random.shuffle(shuffled)
        target    = min(max(int(np.ceil(split_ratio * n)), min_pt), max_pt if max_pt is not None else int(np.ceil(split_ratio * n)))
        train_set = set(shuffled[:target].tolist())
        for label in np.unique(labels):
            idxs = labeled_idx[labels == label]
            if not any(i in train_set for i in idxs):
                train_set.add(int(np.random.choice(idxs, size=1)))
        train_idx = np.array(sorted(train_set), dtype=np.int64)
        test_idx  = np.setdiff1d(labeled_idx, train_idx)
        return (
            torch.tensor(train_idx, dtype=torch.long, device=device),
            torch.tensor(test_idx,  dtype=torch.long, device=device),
        )


# ── UniXcoder subclass ────────────────────────────────────────────────────────

class UniXcoderHeterogeneousData(HeterogeneousData):
    """HeterogeneousData with file/member features from pre-computed UniXcoder embeddings."""

    def _aligned_embedding_tensor(self, node_keys: List[str]) -> Tuple[torch.Tensor, int]:
        emb_dim = int(self._emb_matrix.shape[1])
        aligned = torch.zeros((len(node_keys), emb_dim), dtype=torch.float32)
        node_pos, emb_pos = [], []
        for np_i, key in enumerate(node_keys):
            idx = self._key_to_idx.get(key)
            if idx is None and "/" in key:
                idx = self._key_to_idx.get(key.split("/", 1)[1])
            if idx is not None:
                node_pos.append(np_i)
                emb_pos.append(idx)
        if emb_pos:
            aligned[np.asarray(node_pos, dtype=np.int64)] = torch.as_tensor(
                np.asarray(self._emb_matrix[np.asarray(emb_pos, dtype=np.int64)], dtype=np.float32)
            )
        return aligned, len(emb_pos)

    def __init__(
        self,
        df: pd.DataFrame,
        df_dep: pd.DataFrame,
        config: GraphConfig,
        emb_matrix: np.ndarray,
        dep_keys: List[str],
        mem_emb_matrix: Optional[np.ndarray] = None,
        mem_dep_keys: Optional[List[str]] = None,
    ):
        object.__setattr__(self, "_emb_matrix",     emb_matrix)
        object.__setattr__(self, "_key_to_idx",     {k: i for i, k in enumerate(dep_keys)})
        object.__setattr__(self, "_mem_emb_matrix", mem_emb_matrix)
        object.__setattr__(self, "_mem_key_to_idx",
                           {k: i for i, k in enumerate(mem_dep_keys)} if mem_dep_keys else None)
        super().__init__(df, df_dep, config)

    def _create_node_features(self) -> None:
        folder_x: Optional[np.ndarray] = None
        if self.config.folder_edges or self.config.loc_features:
            folder_nodes = list(self.folder_id.keys())
            folder_texts = [" ".join(self.folder_tokens.get(n, [])) for n in folder_nodes]
            folder_x = self._vectorize_binary(folder_texts, self.folder_vocab)
            if folder_x.shape[1] == 0:
                folder_x = self._empty_or_constant(len(folder_nodes))
            if self.config.folder_edges:
                self["folder"].x = torch.tensor(folder_x, dtype=torch.float)

        file_nodes = list(self.file_id.keys())
        file_x, matched = self._aligned_embedding_tensor(file_nodes)
        self["file"].x         = file_x
        self["file"].num_nodes = len(file_nodes)
        if matched / max(len(file_nodes), 1) < 0.5:
            print(f"  [UniXcoder] Warning: only {matched}/{len(file_nodes)} files matched.")

        if self.config.loc_features and folder_x is not None and folder_x.shape[1] > 0:
            file_folder_agg = np.zeros((len(file_nodes), folder_x.shape[1]), dtype=np.float32)
            for i, f in enumerate(file_nodes):
                for fn in self.folder_hierarchy.get(f, []):
                    if fn in self.folder_id:
                        file_folder_agg[i] += folder_x[self.folder_id[fn]]
            self["file"].x = torch.cat(
                [self["file"].x, torch.tensor(file_folder_agg, dtype=torch.float)], dim=1
            )

        if self.use_member_graph:
            member_nodes  = list(self.member_id.keys())
            member_files  = [self.member_file_map.get(mid, "") for mid in member_nodes]

            if self._mem_emb_matrix is not None:
                emb_dim  = int(self._mem_emb_matrix.shape[1])
                member_x = torch.zeros((len(member_files), emb_dim), dtype=torch.float32)
                pos, eidx = [], []
                for i, f in enumerate(member_files):
                    j = self._mem_key_to_idx.get(f)
                    if j is None and "/" in f:
                        j = self._mem_key_to_idx.get(f.split("/", 1)[1])
                    if j is not None:
                        pos.append(i); eidx.append(j)
                if eidx:
                    member_x[np.asarray(pos, dtype=np.int64)] = torch.as_tensor(
                        np.asarray(self._mem_emb_matrix[np.asarray(eidx, dtype=np.int64)], dtype=np.float32)
                    )
                matched_m = len(eidx)
            else:
                member_x, matched_m = self._aligned_embedding_tensor(member_files)

            self["member"].x         = member_x
            self["member"].num_nodes = len(member_nodes)
            if matched_m / max(len(member_nodes), 1) < 0.5:
                print(f"  [UniXcoder] Warning: only {matched_m}/{len(member_nodes)} members matched.")

        object.__delattr__(self, "_emb_matrix")
        object.__delattr__(self, "_key_to_idx")
        object.__delattr__(self, "_mem_emb_matrix")
        object.__delattr__(self, "_mem_key_to_idx")
