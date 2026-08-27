from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder

from hetmap.evaluation.metrics import eval_metrics
from hetmap.experiments.config import MLPConfig
from hetmap.models.mlp import UniXcoderMLP
from hetmap.training.hgt_trainer import resolve_device


class FileLevelDataset:
    """Per-file embeddings, labels, and label encoder — no graph."""

    def __init__(self, df: pd.DataFrame, emb_matrix: np.ndarray, dep_keys: List[str]):
        key_to_idx = {k: i for i, k in enumerate(dep_keys)}
        files      = df["File"].dropna().astype(str).drop_duplicates().tolist()
        emb_dim    = emb_matrix.shape[1]

        file_emb = np.zeros((len(files), emb_dim), dtype=np.float32)
        matched  = 0
        for j, fkey in enumerate(files):
            idx = key_to_idx.get(fkey)
            if idx is None and "/" in fkey:
                idx = key_to_idx.get(fkey.split("/", 1)[1])
            if idx is not None:
                file_emb[j] = emb_matrix[idx]
                matched += 1

        coverage = matched / len(files) if files else 0.0
        if coverage < 0.5:
            print(f"  [MLP] Warning: only {matched}/{len(files)} ({coverage:.0%}) files matched.")

        # A file has real ground truth only if its Module is present and not one of the
        # "no label" sentinels — files without GT (e.g. newly added, unmapped files) are
        # tracked separately (`has_gt`, y = -1) rather than folded into a fake class.
        file_label_map = df.drop_duplicates(subset=["File"]).set_index("File")["Module"].to_dict()

        def _is_real_label(v) -> bool:
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return False
            s = str(v).strip()
            return bool(s) and s.lower() not in {"nan", "none", "unmapped", "unknown", ""}

        raw_labels  = [file_label_map.get(f) for f in files]
        has_gt_mask = np.array([_is_real_label(v) for v in raw_labels], dtype=bool)
        labeled_vals = [str(v) for v, ok in zip(raw_labels, has_gt_mask) if ok]

        le = LabelEncoder()
        le.fit(labeled_vals if labeled_vals else ["__none__"])
        encoded = np.full(len(files), -1, dtype=np.int64)
        if labeled_vals:
            encoded[has_gt_mask] = le.transform(labeled_vals)

        self.x             = torch.tensor(file_emb, dtype=torch.float)
        self.y             = torch.tensor(encoded,  dtype=torch.long)
        self.has_gt         = has_gt_mask
        self.num_files     = len(files)
        self.num_classes   = int(len(le.classes_))
        self.label_encoder = le
        self.files         = files
        self.emb_coverage  = coverage

    def generate_split(self, split_ratio: float, min_pt: int, max_pt: int,
                       device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        # Only files with real ground truth are eligible for the labeled/test split —
        # files without GT (has_gt=False) are never auto-sampled into either set.
        all_y       = self.y.cpu().numpy()
        labeled_idx = np.where(self.has_gt)[0]
        labels      = all_y[labeled_idx]
        n           = len(labeled_idx)
        shuffled    = labeled_idx.copy()
        np.random.shuffle(shuffled)
        target    = min(max(int(np.ceil(split_ratio * n)), min_pt), max_pt)
        train_set = set(shuffled[:target].tolist())
        for label in np.unique(labels):
            idxs = labeled_idx[labels == label]
            if not any(i in train_set for i in idxs):
                train_set.add(int(np.random.choice(idxs)))
        train_idx = np.array(sorted(train_set), dtype=np.int64)
        test_idx  = np.setdiff1d(labeled_idx, train_idx)
        return (
            torch.tensor(train_idx, dtype=torch.long, device=device),
            torch.tensor(test_idx,  dtype=torch.long, device=device),
        )


def few_shot_learning_mlp(
    dataset: FileLevelDataset,
    file_train: torch.Tensor,
    file_test: torch.Tensor,
    cfg: MLPConfig,
    has_gt: bool = True,
) -> Tuple[dict, List[dict]]:
    """Train the MLP on `file_train` (the initial labeled set L) and evaluate/predict on `file_test`.

    `has_gt=True` (default, used by the paper's benchmark runner) evaluates against
    `dataset.y` and returns F1/precision/recall metrics. `has_gt=False` is the
    real-world mapping use case: `file_test` contains files with no ground truth, so
    only predictions are returned — no metrics. `file_train` must always carry real labels.
    """
    device = resolve_device()
    x = dataset.x.to(device)
    y = dataset.y.to(device)

    model = UniXcoderMLP(
        in_dim=x.size(1), hidden=cfg.hidden_channels,
        out_dim=dataset.num_classes, num_layers=cfg.num_layers, dropout=cfg.dropout,
    ).to(device)

    optimizer    = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    criterion    = torch.nn.CrossEntropyLoss()
    true_labels  = y.clone()
    n            = dataset.num_files
    original_mask = torch.zeros(n, dtype=torch.bool, device=device)
    original_mask[file_train] = True
    pseudo_mask   = torch.zeros(n, dtype=torch.bool, device=device)
    train_labels  = true_labels.clone()

    warmup_idx = torch.where(original_mask)[0]
    for _ in range(cfg.epochs):
        model.train(); optimizer.zero_grad()
        criterion(model(x)[warmup_idx], train_labels[warmup_idx]).backward()
        optimizer.step()

    prev_added = None
    for _ in range(cfg.self_train_rounds):
        if prev_added is not None and prev_added < 5:
            break
        model.eval()
        with torch.no_grad():
            probs = model(x).softmax(dim=-1)
            conf, pred = probs.max(dim=-1)

        existing = torch.where(pseudo_mask)[0]
        if existing.numel() > 0:
            hi = existing[conf[existing] > cfg.threshold]
            if hi.numel() > 0:
                changed = hi[pred[hi] != train_labels[hi]]
                if changed.numel() > 0:
                    train_labels[changed] = pred[changed]

        candidates = torch.where(~(original_mask | pseudo_mask))[0]
        if candidates.numel() == 0:
            break
        new_idx = candidates[conf[candidates] > cfg.threshold]
        if new_idx.numel() == 0:
            break
        prev_added = int(new_idx.numel())
        pseudo_mask[new_idx]  = True
        train_labels[new_idx] = pred[new_idx]

        train_idx = torch.where(original_mask | pseudo_mask)[0]
        for _ in range(cfg.self_train_epochs):
            model.train(); optimizer.zero_grad()
            criterion(model(x)[train_idx], train_labels[train_idx]).backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        out      = model(x)
        probs    = out.softmax(dim=-1)
        conf_all = probs.max(dim=-1).values.cpu().numpy()
        all_pred = out.argmax(dim=-1).cpu().numpy()

    test_np   = file_test.cpu().numpy()
    test_pred = all_pred[test_np]
    classes   = dataset.label_encoder.classes_

    if not has_gt:
        # No ground truth for these files — return predictions only, skip metrics.
        preds_rows = [
            {
                "file_path":  dataset.files[int(idx)],
                "pred_module": classes[test_pred[i]],
                "confidence":  round(float(conf_all[idx]), 4),
            }
            for i, idx in enumerate(test_np)
        ]
        return {}, preds_rows

    true_np   = true_labels.cpu().numpy()
    test_true = true_np[test_np]

    metrics = eval_metrics(test_pred, test_true)
    mapped_mask            = pseudo_mask[file_test].cpu().numpy()
    mapped_count           = int(mapped_mask.sum())
    metrics["mapped_count"]   = mapped_count
    metrics["unmapped_count"] = len(test_np) - mapped_count
    metrics["coverage"]       = mapped_count / len(test_np) if len(test_np) > 0 else 0.0

    preds_rows = [
        {
            "file_path":   dataset.files[int(idx)],
            "true_module": classes[test_true[i]],
            "pred_module": classes[test_pred[i]],
            "confidence":  round(float(conf_all[idx]), 4),
            "correct":     bool(test_pred[i] == test_true[i]),
        }
        for i, idx in enumerate(test_np)
    ]
    return metrics, preds_rows
