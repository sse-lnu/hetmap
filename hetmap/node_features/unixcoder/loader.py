from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np

_DEFAULT_EMB_DIR = Path(__file__).parent.parent.parent / "data" / "unixcoder_emb"


def load_embeddings(
    hetmap_name: str,
    pooling: str = "file_ft",
    emb_dir: Path = _DEFAULT_EMB_DIR,
) -> Tuple[np.ndarray, List[str]]:
    """Load pre-computed UniXcoder embeddings for one project.

    Returns
    -------
    emb_matrix : float32 array (N, 768), L2-normalised
    dep_keys   : list of N dep_key strings (join key for HetMap File column)
    """
    emb_dir   = Path(emb_dir)
    emb_path  = emb_dir / f"{hetmap_name}_{pooling}_emb.npy"
    keys_path = emb_dir / f"{hetmap_name}_{pooling}_keys.txt"

    if not emb_path.exists():
        raise FileNotFoundError(
            f"No embedding found for '{hetmap_name}' (pooling={pooling}) at {emb_path}"
        )

    emb = np.load(str(emb_path), mmap_mode="r")
    if emb.dtype != np.float32:
        emb = emb.astype(np.float32)
    dep_keys = keys_path.read_text(encoding="utf-8").splitlines()

    assert len(dep_keys) == emb.shape[0], (
        f"dep_keys ({len(dep_keys)}) != emb rows ({emb.shape[0]}) for '{hetmap_name}'"
    )
    return emb, dep_keys
