"""run_experiments.py — reproduce HetMap's RQ1 / RQ2 / RQ3 experiments.

Mirrors the paper's three research questions directly (see HetMap.tex):

  RQ1  How does HGT perform relative to existing GNN-based approaches?
       -> one HGT run: FFM graph, W2V file/folder features with folder-path
          tokens concatenated onto the file feature (loc_features=True).
       -> results/RQ1/

  RQ2  How much architectural information do structural (N2V), lexical (W2V),
       and contextual (UniXcoder zero-shot / fine-tuned) representations carry
       on their own, with no graph structure (feature-only MLP)?
       -> results/RQ2/{N2V,W2V,UXC_ZS,UXC_FT}/

  RQ3  Does that ranking persist once the same features initialize HGT, across
       F / FM / FF / FFM graph conditions?
       -> results/RQ3/{W2V,N2V,UXC_FT}/{F,FM,FF,FFM}/

Usage:
    python run_experiments.py --rq 1
    python run_experiments.py --rq 2 --datasets SH-3D A.UML
    python run_experiments.py --rq 3 --num-runs 5
    python run_experiments.py --rq all

Results land under results/RQ{1,2,3}/ as one CSV per dataset (plus a
"*_predictions.csv" with per-file predictions), same schema across all cells.
"""
from __future__ import annotations

import argparse
import dataclasses
import shutil
from pathlib import Path
from typing import List, Optional

from hetmap.experiments import hgt_runner
from hetmap.experiments.config import (
    ExperimentConfig, GraphConfig, MLPExperimentConfig, MLPConfig,
    N2V_TRAINING, W2V_TRAINING, UXC_TRAINING,
)
from hetmap.experiments.hgt_runner import run_experiments
from hetmap.experiments.mlp_runner import run_experiments as run_mlp

DATA_DIR = Path("data")
EMB_DIR  = Path("data/unixcoder_emb")
RESULTS  = Path("results")

# The paper's 10 benchmark systems (see Table "Benchmark Systems" in HetMap.tex).
# HetMap registry names, not the paper's display names (Jabref == JabRef, TeamMates == T.Mates).
ALL_DATASETS = [
    "A.UML", "C.Img", "Jabref", "Lucene", "SH-3D",
    "TeamMates", "Bash", "HDF", "Chrome", "HDC",
]


def _bootstrap_embeddings() -> None:
    """First-run convenience: the repo ships curated embeddings for the 4 example
    datasets under embeddings/{n2v_emb,w2v_emb,unixcoder_emb}/, but the loaders
    (hetmap.node_features.{n2v,w2v}, hetmap.node_features.unixcoder.loader) read
    from data/{n2v_emb,w2v_emb,unixcoder_emb}/ by default. Populate data/ from the
    bundled embeddings/ the first time this script runs, so a fresh clone works
    out of the box without manually copying files.
    """
    emb_root = Path("embeddings")
    if not emb_root.exists():
        return
    for sub in ("n2v_emb", "w2v_emb", "unixcoder_emb"):
        src = emb_root / sub
        dst = DATA_DIR / sub
        if not src.exists():
            continue
        dst.mkdir(parents=True, exist_ok=True)
        for f in src.iterdir():
            target = dst / f.name
            if not target.exists():
                shutil.copy2(f, target)


def _override_num_runs(num_runs: Optional[int]) -> None:
    """hgt_runner resolves N2V/W2V/UXC training config from a module-level dict keyed
    by pooling, ignoring ExperimentConfig.training — so overriding num_runs for a quick
    smoke test means patching that dict directly rather than passing it through config.
    """
    if num_runs is None:
        return
    hgt_runner._METHOD_TRAINING = {
        pooling: dataclasses.replace(cfg, num_runs=num_runs)
        for pooling, cfg in hgt_runner._METHOD_TRAINING.items()
    }


def run_rq1(datasets: List[str], num_runs: Optional[int] = None) -> None:
    """RQ1: HGT on the FFM graph, W2V file/folder features, folder-path tokens
    concatenated onto the file feature (loc_features=True) — matches the paper's
    HGT row in Table "rq1_results", compared against the MLP/GAT baselines (not
    reproduced here — those live in a separate repo, see paper Section 4.2)."""
    _override_num_runs(num_runs)
    run_experiments(ExperimentConfig(
        data_dir=DATA_DIR, results_dir=RESULTS / "RQ1",
        include_datasets=datasets,
        graph=GraphConfig(pooling="w2v", folder_edges=True, keep_member_nodes=True,
                          loc_features=True),
    ))


def run_rq2(datasets: List[str], num_runs: Optional[int] = None) -> None:
    """RQ2: feature-only MLP (no graph) for each of the four representations."""
    mlp_cfg = MLPConfig(num_runs=num_runs) if num_runs is not None else MLPConfig()

    run_mlp(MLPExperimentConfig(
        data_dir=DATA_DIR, results_dir=RESULTS / "RQ2" / "N2V",
        include_datasets=datasets, pooling="n2v", mlp=mlp_cfg,
    ))
    run_mlp(MLPExperimentConfig(
        data_dir=DATA_DIR, results_dir=RESULTS / "RQ2" / "W2V",
        include_datasets=datasets, pooling="w2v", mlp=mlp_cfg,
    ))
    run_mlp(MLPExperimentConfig(
        data_dir=DATA_DIR, results_dir=RESULTS / "RQ2" / "UXC_ZS",
        include_datasets=datasets, pooling="file", emb_dir=EMB_DIR, mlp=mlp_cfg,
    ))
    run_mlp(MLPExperimentConfig(
        data_dir=DATA_DIR, results_dir=RESULTS / "RQ2" / "UXC_FT",
        include_datasets=datasets, pooling="file_ft", emb_dir=EMB_DIR, mlp=mlp_cfg,
    ))


def run_rq3(datasets: List[str], num_runs: Optional[int] = None) -> None:
    """RQ3: HGT initialized with W2V / N2V / UXC-FT across F, FM, FF, FFM.
    (MLP baseline for each feature is RQ2's — not rerun here, see paper Section 5.3.)"""
    _override_num_runs(num_runs)

    variants = {
        "F":   dict(folder_edges=False, keep_member_nodes=False),
        "FM":  dict(folder_edges=False, keep_member_nodes=True),
        "FF":  dict(folder_edges=True,  keep_member_nodes=False),
        "FFM": dict(folder_edges=True,  keep_member_nodes=True),
    }

    for name, kw in variants.items():
        run_experiments(ExperimentConfig(
            data_dir=DATA_DIR, results_dir=RESULTS / "RQ3" / "W2V" / name,
            include_datasets=datasets,
            graph=GraphConfig(pooling="w2v", **kw),
        ))

    for name, kw in variants.items():
        run_experiments(ExperimentConfig(
            data_dir=DATA_DIR, results_dir=RESULTS / "RQ3" / "N2V" / name,
            include_datasets=datasets,
            graph=GraphConfig(pooling="n2v", **kw),
        ))

    for name, kw in variants.items():
        run_experiments(ExperimentConfig(
            data_dir=DATA_DIR, results_dir=RESULTS / "RQ3" / "UXC_FT" / name,
            include_datasets=datasets,
            graph=GraphConfig(pooling="file_ft", use_w2v_features=False, emb_dir=EMB_DIR, **kw),
        ))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rq", choices=["1", "2", "3", "all"], default="all",
                    help="Which research question's experiments to run.")
    ap.add_argument("--datasets", nargs="+", default=None,
                    help=f"Subset of datasets to run (default: all {len(ALL_DATASETS)} paper systems).")
    ap.add_argument("--num-runs", type=int, default=None,
                    help="Override the number of self-training runs per dataset "
                         "(paper uses 100; use a small number like 2-5 for a smoke test).")
    args = ap.parse_args()

    datasets = args.datasets or ALL_DATASETS
    _bootstrap_embeddings()

    if args.rq in ("1", "all"):
        print("\n===== RQ1 =====")
        run_rq1(datasets, args.num_runs)
    if args.rq in ("2", "all"):
        print("\n===== RQ2 =====")
        run_rq2(datasets, args.num_runs)
    if args.rq in ("3", "all"):
        print("\n===== RQ3 =====")
        run_rq3(datasets, args.num_runs)


if __name__ == "__main__":
    main()
