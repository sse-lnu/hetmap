from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence


@dataclass
class GraphConfig:
    keep_member_nodes:             bool          = True
    folder_edges:                  bool          = True
    loc_features:                  bool          = False
    use_w2v_features:              bool          = True
    add_members_to_file_features:  bool          = False
    add_constant_feature_if_empty: bool          = True
    exclude_dep_types:             Sequence[str] = field(default_factory=list)
    w2v_vector_size:               int           = 128
    w2v_window:                    int           = 5
    w2v_min_count:                 int           = 1
    w2v_sg:                        int           = 1
    w2v_epochs:                    int           = 20
    w2v_workers:                   int           = 4
    random_seed:                   Optional[int] = 42
    # UniXcoder-specific
    emb_dir:  Path = field(default_factory=lambda: Path("data") / "unixcoder_emb")
    pooling:  str  = "file_ft"
    # Node2Vec-specific
    n2v_dim:  int  = 128


@dataclass
class TrainingConfig:
    num_runs:          int           = 100
    split_ratio:       float         = 0.05
    epochs:            int           = 100
    hidden_channels:   int           = 256
    heads:             int           = 4
    num_layers:        int           = 1
    dropout:           float         = 0.0
    lr:                float         = 1e-3
    self_train_rounds: int           = 3
    self_train_epochs: int           = 30
    threshold:         float         = 0.95
    flush_every:       int           = 2
    min_train_points:  int           = 30
    max_train_points:  int           = 200
    verbose:           bool          = False
    device:            Optional[str] = None


# ── Per-method HGT training defaults ─────────────────────────────────────────
# Edit here to change N2V / W2V / UXC settings across all experiments.
# N2V and W2V: 128-d features → 8 heads, 4 self-training rounds, 90% threshold
N2V_TRAINING = TrainingConfig(heads=4, self_train_rounds=3, threshold=0.95)
W2V_TRAINING = TrainingConfig(heads=4, self_train_rounds=3, threshold=0.95)
UXC_TRAINING = TrainingConfig(heads=4, self_train_rounds=3, threshold=0.95)


@dataclass
class ExperimentConfig:
    data_dir:         Path                    = field(default_factory=lambda: Path("data"))
    results_dir:      Path                    = field(default_factory=lambda: Path("results"))
    include_datasets: Optional[Sequence[str]] = None
    exclude_datasets: Sequence[str]           = field(default_factory=tuple)
    graph:            GraphConfig             = field(default_factory=GraphConfig)
    training:         TrainingConfig          = field(default_factory=TrainingConfig)


@dataclass
class MLPConfig:
    hidden_channels:   int           = 256
    num_layers:        int           = 2
    dropout:           float         = 0.0
    lr:                float         = 1e-3
    epochs:            int           = 100
    self_train_rounds: int           = 4
    self_train_epochs: int           = 30
    threshold:         float         = 0.95
    num_runs:          int           = 100
    split_ratio:       float         = 0.05
    min_train_points:  int           = 30
    max_train_points:  int           = 200
    flush_every:       int           = 2
    random_seed:       Optional[int] = 42


@dataclass
class MLPExperimentConfig:
    data_dir:         Path                    = field(default_factory=lambda: Path("data"))
    results_dir:      Path                    = field(default_factory=lambda: Path("results"))
    include_datasets: Optional[Sequence[str]] = None
    exclude_datasets: Sequence[str]           = field(default_factory=tuple)
    mlp:              MLPConfig               = field(default_factory=MLPConfig)
    pooling:          str                     = "file_ft"
    emb_dir:          Path                    = field(default_factory=lambda: Path("data"))
