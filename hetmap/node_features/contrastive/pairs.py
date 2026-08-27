"""Positive-pair construction from dependency graphs."""

import random
from typing import Dict, List, Tuple

from torch.utils.data import Dataset


def build_pairs(
    paths: List[str],
    deps_data: Dict,
    pairs_per_file: int = 2,
    hop2_prob: float = 0.5,
    topk_n2: int = 8,
    seed: int = 0,
    directed: bool = False,
) -> Tuple[List[Tuple[int, int]], Dict]:
    """Sample positive pairs from the 1- and 2-hop dependency graph.

    For each file, samples `pairs_per_file` positive partners from its 1-hop
    dependency neighbours (files it calls/extends/imports or is called by).
    With probability `hop2_prob`, a 2-hop neighbour (reachable via two edges)
    is chosen instead, preferring those with the highest overlap count.

    Returns:
        pairs: list of (idx_a, idx_b) index tuples into `paths`
        stats: dict with pair counts and coverage info
    """
    rng = random.Random(seed)
    path2idx = {p: i for i, p in enumerate(paths)}

    hop1: Dict[str, List[str]] = {p: [] for p in path2idx}
    for p in path2idx:
        rec = deps_data.get(p, {"fwd": [], "rev": []})
        seen = set()
        for entry in rec.get("fwd", []):
            t = entry[0]
            if t in path2idx and t != p and t not in seen:
                hop1[p].append(t)
                seen.add(t)
        if not directed:
            for entry in rec.get("rev", []):
                s = entry[0]
                if s in path2idx and s != p and s not in seen:
                    hop1[p].append(s)
                    seen.add(s)

    pairs: List[Tuple[int, int]] = []
    n_no1 = n_no2 = 0

    for p, i in path2idx.items():
        N1 = hop1[p]
        if not N1:
            n_no1 += 1
            continue

        N1_set = set(N1)
        overlap: Dict[str, int] = {}
        for q in N1:
            for r in hop1[q]:
                if r in N1_set or r == p:
                    continue
                overlap[r] = overlap.get(r, 0) + 1

        if not overlap:
            n_no2 += 1
            top_N2: List[str] = []
        else:
            top_N2 = sorted(overlap, key=lambda x: -overlap[x])[:topk_n2]

        for _ in range(pairs_per_file):
            if top_N2 and rng.random() < hop2_prob:
                j = path2idx[rng.choice(top_N2)]
            else:
                j = path2idx[rng.choice(N1)]
            pairs.append((i, j))

    rng.shuffle(pairs)
    return pairs, {"pairs": len(pairs), "no_1hop": n_no1, "no_2hop": n_no2}


def build_pos_set(
    paths: List[str],
    deps_data: Dict,
    directed: bool = False,
) -> frozenset:
    """Return the complete 1-hop edge set as a frozenset of (min_idx, max_idx) tuples.

    Used to build the per-batch connected mask in masked NT-Xent training.
    Runs once before the training loop — O(E) where E is the number of dep edges.
    """
    path2idx = {p: i for i, p in enumerate(paths)}
    pos_set: set = set()
    for p, i in path2idx.items():
        rec = deps_data.get(p, {"fwd": [], "rev": []})
        for entry in rec.get("fwd", []):
            j = path2idx.get(entry[0])
            if j is not None and j != i:
                pos_set.add((min(i, j), max(i, j)))
        if not directed:
            for entry in rec.get("rev", []):
                j = path2idx.get(entry[0])
                if j is not None and j != i:
                    pos_set.add((min(i, j), max(i, j)))
    return frozenset(pos_set)


class PairDatasetWithIdx(Dataset):
    """Yields (text_a, text_b, idx_a, idx_b).

    The indices are needed by the training loop to build the per-batch
    connected mask: knowing which file each row corresponds to lets us
    look up whether any two batch members share a dependency edge.
    """

    def __init__(self, pairs: List[Tuple[int, int]], texts: List[str]):
        self.pairs = pairs
        self.texts = texts

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx):
        i, j = self.pairs[idx]
        return self.texts[i], self.texts[j], i, j
