from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score


def eval_metrics(pred: np.ndarray, true: np.ndarray) -> dict:
    present = np.unique(true)
    return {
        "f1_micro":        f1_score(true, pred, average="micro"),
        "f1_macro":        f1_score(true, pred, average="macro", labels=present, zero_division=0),
        "precision_micro": precision_score(true, pred, average="micro", zero_division=0),
        "precision_macro": precision_score(true, pred, average="macro", labels=present, zero_division=0),
        "recall_micro":    recall_score(true, pred, average="micro", zero_division=0),
        "recall_macro":    recall_score(true, pred, average="macro", labels=present, zero_division=0),
    }
