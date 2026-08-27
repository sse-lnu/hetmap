from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

_W2V_CACHE_DIR = Path("data/w2v_emb")


def load_or_train_w2v_file_embs(
    name: str,
    df,
    df_dep,
    graph_cfg,
) -> Tuple[np.ndarray, List[str]]:
    """Return (emb_matrix, file_keys) for file-level W2V with disk caching."""
    emb_path  = _W2V_CACHE_DIR / f"{name}_w2v_file_emb.npy"
    keys_path = _W2V_CACHE_DIR / f"{name}_w2v_file_keys.txt"
    if emb_path.exists() and keys_path.exists():
        print(f"  [{name}] W2V file embs cached — loading")
        return np.load(emb_path), keys_path.read_text(encoding="utf-8").splitlines()
    from hetmap.process_data.preprocessing import Preprocessor
    prep = Preprocessor(df.copy(), df_dep.copy(), graph_cfg)
    prep.preprocess()
    file_nodes = prep.df_files["File"].dropna().astype(str).drop_duplicates().tolist()
    file_texts = [" ".join(prep.file_tokens.get(f, [])) for f in file_nodes]
    emb = build_w2v_features(
        file_texts, prep.file_dict or [],
        vector_size=graph_cfg.w2v_vector_size, window=graph_cfg.w2v_window,
        min_count=graph_cfg.w2v_min_count, sg=graph_cfg.w2v_sg,
        workers=graph_cfg.w2v_workers, epochs=graph_cfg.w2v_epochs,
        seed=graph_cfg.random_seed,
    )
    _W2V_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(emb_path, emb)
    keys_path.write_text("\n".join(file_nodes), encoding="utf-8")
    print(f"  [{name}] W2V file embs saved  shape={emb.shape}")
    return emb, file_nodes


def build_w2v_features(
    texts: List[str],
    vocabulary: List[str],
    vector_size: int = 128,
    window: int = 5,
    min_count: int = 1,
    sg: int = 1,
    workers: int = 4,
    epochs: int = 20,
    seed: Optional[int] = 42,
) -> np.ndarray:
    """Train Word2Vec on *texts* and return mean-pooled embeddings (N, vector_size)."""
    from gensim.models import Word2Vec

    if not texts:
        return np.zeros((0, vector_size), dtype=np.float32)

    vocab_set = set(vocabulary) if vocabulary else None
    tokenized: List[List[str]] = []
    for text in texts:
        toks = [t for t in str(text).split() if t]
        if vocab_set is not None:
            toks = [t for t in toks if t in vocab_set]
        tokenized.append(toks)

    train_corpus = [toks for toks in tokenized if toks]
    if not train_corpus:
        return np.zeros((len(texts), vector_size), dtype=np.float32)

    kwargs = dict(sentences=train_corpus, vector_size=vector_size, window=window,
                  min_count=min_count, sg=sg, workers=workers, epochs=epochs)
    if seed is not None:
        kwargs["seed"] = seed

    w2v = Word2Vec(**kwargs)
    x = np.zeros((len(texts), w2v.vector_size), dtype=np.float32)
    for i, toks in enumerate(tokenized):
        vecs = [w2v.wv[t] for t in toks if t in w2v.wv]
        if vecs:
            x[i] = np.mean(vecs, axis=0).astype(np.float32)
    return x
