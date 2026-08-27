from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np

from hetmap.node_features.node2vec_model import N2vData, Node2VecModel

_N2V_EMB_DIR = Path("data/n2v_emb")


def load_or_train_n2v(
    name: str,
    df,
    df_dep,
    graph_cfg,
) -> Tuple[np.ndarray, List[str]]:
    emb_path  = _N2V_EMB_DIR / f"{name}_n2v_emb.npy"
    keys_path = _N2V_EMB_DIR / f"{name}_n2v_keys.txt"

    if emb_path.exists() and keys_path.exists():
        print(f"  [{name}] N2V cached — loading")
        return np.load(emb_path), keys_path.read_text(encoding="utf-8").splitlines()

    print(f"  [{name}] N2V training...")
    data = N2vData(df, df_dep)
    dim  = getattr(graph_cfg, "n2v_dim", 128)
    emb  = Node2VecModel(dimensions=dim).fit_transform(data)

    _N2V_EMB_DIR.mkdir(parents=True, exist_ok=True)
    np.save(emb_path, emb)
    keys_path.write_text("\n".join(data.node_list), encoding="utf-8")
    print(f"  [{name}] saved {emb_path.name}  shape={emb.shape}")
    return emb, data.node_list
