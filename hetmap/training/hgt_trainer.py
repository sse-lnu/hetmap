from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score

from hetmap.experiments.config import TrainingConfig
from hetmap.models.hgt import HGTNet


def resolve_device(preferred: Optional[str] = None) -> torch.device:
    if preferred is not None:
        norm = str(preferred).strip().lower()
        if norm == "cpu":
            return torch.device("cpu")
        if norm == "cuda":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        raise ValueError(f"Unsupported device preference: {preferred}")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _test(model: HGTNet, data, file_test: torch.Tensor,
          true_file_labels: Optional[torch.Tensor] = None,
          mapped_test_mask: Optional[np.ndarray] = None,
          has_gt: bool = True) -> Tuple[Dict, List[dict]]:
    device = next(model.parameters()).device
    data   = data.to(device)
    model.eval()

    with torch.no_grad():
        logits, _ = model(data)

    probs     = logits.softmax(dim=-1)
    conf_all  = probs.max(dim=-1).values.cpu().numpy()
    all_pred  = logits.argmax(dim=-1).cpu().numpy()
    test_np   = file_test.cpu().numpy()
    test_pred = all_pred[test_np]

    classes       = data.label_encoder.classes_
    inv_file_id   = {v: k for k, v in data.file_id.items()}

    if not has_gt:
        # No ground truth for these files (real-world mapping use case) — return
        # predictions only, skip every metric that requires a true label.
        preds_rows = [
            {
                "file_path":  inv_file_id.get(int(idx), str(idx)),
                "pred_module": classes[test_pred[i]],
                "confidence":  round(float(conf_all[idx]), 4),
            }
            for i, idx in enumerate(test_np)
        ]
        return {}, preds_rows

    if true_file_labels is None:
        true_file_labels = data["file"].y.clone()
    true_np   = true_file_labels.cpu().numpy()
    test_true = true_np[test_np]
    present   = np.unique(test_true)

    preds_rows = [
        {
            "file_path":   inv_file_id.get(int(idx), str(idx)),
            "true_module": classes[test_true[i]],
            "pred_module": classes[test_pred[i]],
            "confidence":  round(float(conf_all[idx]), 4),
            "correct":     bool(test_pred[i] == test_true[i]),
        }
        for i, idx in enumerate(test_np)
    ]

    results = {
        "f1_micro":        f1_score(test_true, test_pred, average="micro"),
        "f1_macro":        f1_score(test_true, test_pred, average="macro", labels=present, zero_division=0),
        "precision_micro": precision_score(test_true, test_pred, average="micro", zero_division=0),
        "precision_macro": precision_score(test_true, test_pred, average="macro", labels=present, zero_division=0),
        "recall_micro":    recall_score(test_true, test_pred, average="micro", zero_division=0),
        "recall_macro":    recall_score(test_true, test_pred, average="macro", labels=present, zero_division=0),
    }

    if mapped_test_mask is None:
        mapped_test_mask = np.zeros(len(test_true), dtype=bool)
    else:
        mapped_test_mask = np.asarray(mapped_test_mask, dtype=bool)

    mapped_count   = int(mapped_test_mask.sum())
    unmapped_count = int(len(test_true) - mapped_count)
    coverage       = float(mapped_count / len(test_true)) if len(test_true) > 0 else 0.0
    results.update({"mapped_count": mapped_count, "unmapped_count": unmapped_count, "coverage": coverage})

    if mapped_count > 0:
        mt, mp = test_true[mapped_test_mask], test_pred[mapped_test_mask]
        ml     = np.unique(mt)
        results.update({
            "mapped_f1_micro":        f1_score(mt, mp, average="micro"),
            "mapped_f1_macro":        f1_score(mt, mp, average="macro", labels=ml, zero_division=0),
            "mapped_precision_micro": precision_score(mt, mp, average="micro", zero_division=0),
            "mapped_precision_macro": precision_score(mt, mp, average="macro", labels=ml, zero_division=0),
            "mapped_recall_micro":    recall_score(mt, mp, average="micro", zero_division=0),
            "mapped_recall_macro":    recall_score(mt, mp, average="macro", labels=ml, zero_division=0),
        })
    else:
        for k in ["mapped_f1_micro", "mapped_f1_macro", "mapped_precision_micro",
                  "mapped_precision_macro", "mapped_recall_micro", "mapped_recall_macro"]:
            results[k] = float("nan")

    return results, preds_rows


def few_shot_learning(data, file_train: torch.Tensor, file_test: torch.Tensor,
                      training: TrainingConfig, has_gt: bool = True):
    """Train HGT on `file_train` (the initial labeled set L) and evaluate/predict on `file_test`.

    `has_gt=True` (default, used by the paper's benchmark runner) evaluates against
    `data["file"].y` and returns F1/precision/recall metrics. `has_gt=False` is the
    real-world mapping use case: `file_test` contains files with no ground truth
    (e.g. newly added, unmapped files), so only predictions are returned — no metrics.
    The initial labeled set `file_train` is unaffected either way; it must always carry
    real labels.
    """
    device = resolve_device(training.device)
    data   = data.to(device)

    model = HGTNet(
        data=data, hidden_channels=training.hidden_channels,
        out_channels=data.num_classes, heads=training.heads,
        num_layers=training.num_layers, dropout=training.dropout,
    ).to(device)
    model.apply(lambda m: m.reset_parameters() if hasattr(m, "reset_parameters") else None)

    optimizer        = torch.optim.Adam(model.parameters(), lr=training.lr)
    criterion        = nn.CrossEntropyLoss()
    true_file_labels = data["file"].y.clone()
    num_files        = data["file"].x.size(0)
    original_mask    = torch.zeros(num_files, dtype=torch.bool, device=device)
    original_mask[file_train] = True
    pseudo_mask      = torch.zeros(num_files, dtype=torch.bool, device=device)
    train_labels     = true_file_labels.clone()

    warmup_idx = torch.where(original_mask)[0]
    for _ in range(training.epochs):
        model.train(); optimizer.zero_grad()
        logits, _ = model(data)
        criterion(logits[warmup_idx], train_labels[warmup_idx]).backward()
        optimizer.step()

    prev_added = None
    for _ in range(training.self_train_rounds):
        if prev_added is not None and prev_added < 5:
            break
        model.eval()
        with torch.no_grad():
            logits, _ = model(data)
            probs = logits.softmax(dim=-1)
            conf, pred = probs.max(dim=-1)

        existing = torch.where(pseudo_mask)[0]
        if existing.numel() > 0:
            hi = existing[conf[existing] > training.threshold]
            if hi.numel() > 0:
                changed = hi[pred[hi] != train_labels[hi]]
                if changed.numel() > 0:
                    train_labels[changed] = pred[changed]

        candidates = torch.where(~(original_mask | pseudo_mask))[0]
        if candidates.numel() == 0:
            break
        new_idx = candidates[conf[candidates] > training.threshold]
        if new_idx.numel() == 0:
            break
        prev_added = int(new_idx.numel())
        pseudo_mask[new_idx]  = True
        train_labels[new_idx] = pred[new_idx]

        train_idx = torch.where(original_mask | pseudo_mask)[0]
        for _ in range(training.self_train_epochs):
            model.train(); optimizer.zero_grad()
            logits, _ = model(data)
            criterion(logits[train_idx], train_labels[train_idx]).backward()
            optimizer.step()

    results, preds_rows = _test(
        model, data, file_test=file_test, true_file_labels=true_file_labels,
        mapped_test_mask=pseudo_mask[file_test].detach().cpu().numpy(),
        has_gt=has_gt,
    )
    return model, results, preds_rows
