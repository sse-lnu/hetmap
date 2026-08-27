from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import pandas as pd

# CSVs live inside the hetmap package — single source of truth
_DATASETS_DIR = Path(__file__).resolve().parent.parent / "datasets"

DATASET_REGISTRY: Dict[str, Tuple[str, str]] = {
    "A.UML":     ("argouml.csv",   "argouml_deps.csv"),
    "C.Img":     ("commons.csv",   "commons_deps.csv"),
    "Jabref":    ("jabref.csv",    "jabref_deps.csv"),
    "Lucene":    ("lucene.csv",    "lucene_deps.csv"),
    "SH-3D":     ("sweetHome.csv", "sweetHome_deps.csv"),
    "TeamMates": ("teammates.csv", "teammates_deps.csv"),
    "Bash":      ("bash.csv",      "bash_deps.csv"),
    "HDF":       ("hdf.csv",       "hdf_deps.csv"),
    "HDC":       ("hdc.csv",       "hdc_deps.csv"),
    "Chrome":    ("chromium.csv",  "chromium_deps.csv"),
}


def load_datasets(
    include: Optional[Sequence[str]] = None,
    exclude: Optional[Sequence[str]] = None,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """Load project CSVs from the bundled hetmap/datasets/ directory.

    Returns (datasets, dependencies) where keys are HetMap project names.
    """
    include_set = set(include) if include else None
    exclude_set = set(exclude or [])
    datasets: Dict[str, pd.DataFrame] = {}
    dependencies: Dict[str, pd.DataFrame] = {}
    for name, (df_file, dep_file) in DATASET_REGISTRY.items():
        if include_set is not None and name not in include_set:
            continue
        if name in exclude_set:
            continue
        df_path, dep_path = _DATASETS_DIR / df_file, _DATASETS_DIR / dep_file
        if not df_path.exists() or not dep_path.exists():
            # Not every registered system's CSVs are bundled (e.g. Chrome's exceed
            # GitHub's file size limits) — skip rather than crash; fetch from GAER.
            print(f"  [registry] '{name}' CSVs not found under {_DATASETS_DIR} — skipping "
                  f"(see README's Data section).")
            continue
        datasets[name]     = pd.read_csv(df_path)
        dependencies[name] = pd.read_csv(dep_path)
    return datasets, dependencies
