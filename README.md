# HetMap: Code-to-Architecture Mapping with a Heterogeneous Graph Transformer

## Project Overview

A semi-supervised pipeline for mapping source files to architectural modules from a heterogeneous software graph. A **Heterogeneous Graph Transformer (HGT)** is trained with an iterative self-training loop over a graph of **folder**, **file**, and **member** nodes, alongside a **feature-only MLP** baseline (no graph structure) used to isolate what each node representation contributes on its own.

### Motivation

As software evolves, implementation can drift from its intended architecture, while manually maintaining file-to-module mappings does not scale. HetMap treats this as semi-supervised code-to-architecture mapping: given a small set of mapped files, infer the modules of the remaining files. It examines how node initialization and heterogeneous graph structure, including folder organization, member-level information, and typed dependencies, affect mapping performance.

### Approach

Each system is modeled as a heterogeneous graph:
- **Nodes** are folders, files, and members (one summarized member node per file, aggregating its methods/fields/constructors).
- **Edges** are typed: `parent_of` (folder hierarchy), `contains` (folder → file), `belongs_to` (file → member), and typed static dependencies (`Call`, `Import`, `Extend`, `Use`, ...).
- **Node features** come from one of three representations, compared independently:
  - **N2V** — Node2Vec structural embeddings trained on the file dependency graph alone.
  - **W2V** — Word2Vec lexical embeddings over identifier tokens (file/member names, camelCase/snake_case-split).
  - **UXC** — UniXcoder contextual embeddings, in a zero-shot (frozen) and a dependency-guided fine-tuned variant.

HGT conditions attention and message passing on source type, edge type, and target type, then runs an iterative self-training loop: warm up on a labeled seed set, then repeatedly promote high-confidence predictions to pseudo-labels and retrain. The feature-only MLP uses the same node features with no graph structure, isolating what each representation contributes before message passing is introduced.

### Repository Contents

```
HetMap/
├── hetmap/
│   ├── dataset/            # registry.py (dataset → CSV mapping), preprocessing.py (tokenization)
│   ├── datasets/           # raw per-project CSVs (10 systems)
│   ├── build_graph/        # heterogeneous graph construction (HeteroData)
│   ├── node_features/      # W2V / N2V / UniXcoder feature extraction + fine-tuning
│   ├── models/              # HGTNet, feature-only MLP
│   ├── training/            # iterative self-training loop (HGT + MLP)
│   ├── evaluation/          # F1 / precision / recall helpers
│   └── experiments/         # config.py, hgt_runner.py, mlp_runner.py
├── embeddings/              # pre-computed N2V/W2V/UniXcoder embeddings (4 example datasets)
├── run_experiments.py       # reproduce RQ1 / RQ2 / RQ3
├── predict_example.py       # map files with no ground truth
├── results_analysis.ipynb   # figures/tables from results/ (needs matplotlib/seaborn/scipy)
└── requirements.txt
```

---

## Research Questions

- **RQ1 — Effectiveness**: How does HGT perform on code-to-architecture mapping relative to existing GNN-based approaches?
- **RQ2 — Signal strength**: How much architectural information do structural, lexical, and contextual representations carry on their own, with no graph structure?
- **RQ3 — Structure vs. features**: Does that ranking persist once the same features initialize HGT, and how much does graph structure (F / FM / FF / FFM) add beyond each feature type alone?

### RQ1 — Effectiveness

```bash
python run_experiments.py --rq 1
```

One HGT run on the full folder–file–member (FFM) graph, W2V file/folder features with folder-path tokens concatenated onto the file feature. Results land in `results/RQ1/`. The MLP/GAT baselines this is compared against in the paper are **not** reproduced here — they come from an earlier, separate C2A study; this repo only reproduces HetMap's own HGT row.

### RQ2 — Signal strength

```bash
python run_experiments.py --rq 2
```

Feature-only MLP (no graph, no message passing) for each of the four representations — N2V, W2V, UniXcoder zero-shot, UniXcoder fine-tuned. Results land in `results/RQ2/{N2V,W2V,UXC_ZS,UXC_FT}/`.

### RQ3 — Structure vs. features

```bash
python run_experiments.py --rq 3
```

HGT initialized with W2V, N2V, and UniXcoder fine-tuned, across four graph conditions (F, FM, FF, FFM). Results land in `results/RQ3/{W2V,N2V,UXC_FT}/{F,FM,FF,FFM}/`. The graph-free baseline for each feature is RQ2's MLP result — it isn't rerun here.

Run everything with `python run_experiments.py --rq all` (the default). Add `--datasets SH-3D A.UML` to restrict to specific systems, and `--num-runs N` to override the number of self-training runs per dataset (the paper uses 100; use something small like 2–5 for a smoke test).

### Mapping without evaluation

```bash
python predict_example.py --dataset SH-3D
```

Every RQ above assumes full ground truth, since that's what reproducing the paper's numbers requires. In practice you'll often have the opposite: a small set of already-mapped files and a larger set with no known module yet. `predict_example.py` shows that workflow directly — it trains HGT on a labeled subset (`file_train`) exactly as during a normal run, then calls the same training function with `has_gt=False` to get predictions for files with no ground truth: no metrics are computed (there's nothing to score against), just a `{file_path, pred_module, confidence}` row per file, printed to stdout. `--unlabeled-frac` controls how many files are (simulated as) unmapped; in your own project you'd load a genuinely partial mapping instead.

---

## Setup

```bash
pip install -r requirements.txt
```

Tested with `torch>=2.0` / `torch-geometric>=2.4`; both support CUDA or CPU.

The UniXcoder base checkpoint is not bundled (963MB, gitignored) — fetch it once before running anything that uses `pooling="file_ft"`/`"file"` (RQ2's UXC cells, RQ3's UXC_FT cells, or `hetmap/node_features/unixcoder_emb.py`):

```bash
huggingface-cli download microsoft/unixcoder-base --local-dir hetmap/node_features/unixcoder/unixcoder-base
```

N2V and W2V embeddings don't need this — they train directly from the bundled CSVs on first use (and are cached to `data/n2v_emb/` / `data/w2v_emb/` afterward).

---

## Quick Start

```bash
python run_experiments.py --rq 2 --datasets SH-3D --num-runs 2
```

Runs the feature-only MLP for all four representations on SH-3D, 2 runs each — a few seconds on CPU. `results/RQ2/{N2V,W2V,UXC_ZS,UXC_FT}/SH-3D.csv` will contain 2 rows each.

---

## Running Experiments

`run_experiments.py`:
- `--rq {1,2,3,all}` — which research question to run (default: `all`).
- `--datasets <name> [<name> ...]` — subset of the 10 systems (default: all).
- `--num-runs <n>` — override the number of self-training runs per dataset.

`predict_example.py`:
- `--dataset <name>` — HetMap dataset name (default: `SH-3D`).
- `--unlabeled-frac <f>` — fraction of files simulated as unmapped (default: `0.3`).
- `--seed <n>` — random seed for the simulated split (default: `42`).

Default hyperparameters (`hetmap/experiments/config.py` — `N2V_TRAINING`/`W2V_TRAINING`/`UXC_TRAINING`, all identical): 4 attention heads, hidden size 256, 1 HGT message-passing layer, learning rate 0.001, 100 supervised warm-up epochs, self-training threshold 0.95, 3 self-training rounds of 30 epochs each (the feature-only MLP uses 4 rounds), 100 runs per dataset, 5% initial labeled set (min 30, max 200 files).

---

## Input and Output

Each dataset consists of two CSVs under `hetmap/datasets/`:

```
hetmap/datasets/
├── {stem}.csv         # columns: File, ID, Member_Name, Member_Type, Entity, Member_ID, Module
└── {stem}_deps.csv    # columns: Source_File, Target_File, Dependency_Type, Source_Member, Target_Member,
                        #          Source_Member_Type, Target_Member_Type, Dependency_Count, Is_Member_Level, ...
```

`Module` may be missing for some files — those become unlabeled graph nodes rather than being dropped, so they can be predicted on via `has_gt=False` (see `predict_example.py`) instead of requiring full ground truth for evaluation.

Output files, all under `results/RQ{1,2,3}/...`:
- `{dataset}.csv` — metrics per run (`run_id`, `dataset`, `pooling`, `graph_level`, `train_size`, `heads`, `threshold`, `f1_micro`, `f1_macro`, `precision_micro/macro`, `recall_micro/macro`, plus `mapped_`-prefixed variants for the confidently pseudo-labeled subset).
- `{dataset}_predictions.csv` — per-file, per-run `file_path`, `true_module`, `pred_module`, `confidence`, `correct`.

---

## Data

Raw CSVs for 9 of the 10 paper systems (A.UML, C.Img, JabRef, Lucene, SH-3D, TeamMates, Bash, HDF, HDC) are bundled in `hetmap/datasets/` — these are small dependency/metadata tables, not embeddings or source code, so every one of these 9 systems can be used out of the box for N2V and W2V (which train directly from these CSVs). **Chrome's CSVs are not bundled** — at 11M LoC / 284k dependencies they exceed GitHub's file size limits (72MB/140MB); fetch `chromium.csv`/`chromium_deps.csv` from GAER (below) and place them in `hetmap/datasets/` to use Chrome.

`embeddings/{n2v_emb,w2v_emb,unixcoder_emb}/` ships **pre-computed** embeddings for 4 example systems only — **SH-3D, A.UML, Lucene, C.Img** — copied into `data/` automatically the first time `run_experiments.py` runs, so those 4 need no training/encoding step at all. For the other 6 systems, N2V/W2V embeddings are trained on first use from the bundled CSVs (cached afterward); UniXcoder embeddings require the source repos themselves (not bundled) and running `hetmap/node_features/unixcoder_emb.py` — the full set of source repos for all 10 systems is available via the [GAER repository](https://github.com/sse-lnu/GAER).
