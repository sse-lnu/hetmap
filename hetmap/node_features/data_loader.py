"""Load project file list + dependency graph from bundled hetmap/datasets/ CSVs.

Used by unixcoder_emb.py to get dep_keys and deps_data without depending on arc_recovery.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

# CSVs are bundled inside hetmap/datasets/ — same source used by registry.py
_DATASETS_DIR = Path(__file__).resolve().parent.parent / "datasets"

# Maps arc_key → (csv_file, deps_csv_file, repo_subfolder_within_project_dir)
# repo_subfolder=None means the project_dir IS the repo root (e.g. chromium)
_PROJECT_REGISTRY: Dict[str, Tuple[str, str, str]] = {
    "argouml":   ("argouml.csv",   "argouml_deps.csv",   "argouml"),
    "commons":   ("commons.csv",   "commons_deps.csv",
                  "commons/verify-commons-imaging-1.0-alpha2/"
                  "apache-src/commons-imaging-1.0-alpha2-src/src"),
    "jabref":    ("jabref.csv",    "jabref_deps.csv",    "jabref/src/main/java"),
    "lucene":    ("lucene.csv",    "lucene_deps.csv",    "lucene"),
    "sweetHome": ("sweetHome.csv", "sweetHome_deps.csv", "SweetHome3D"),
    "teammates": ("teammates.csv", "teammates_deps.csv", "teammates"),
    "chromium":  ("chromium.csv",  "chromium_deps.csv",  None),  # absolute; pass project_dir directly
    "bash":      ("bash.csv",      "bash_deps.csv",      "bash"),
    "hdf":       ("hdf.csv",       "hdf_deps.csv",       "hdf"),
    "hdc":       ("hdc.csv",       "hdc_deps.csv",       "hdc"),
}


def load_project(arc_key: str, project_dir: str) -> Tuple[pd.DataFrame, Dict]:
    """Return (df, deps_data) for a project.

    df has a `dep_key` column (relative file path) and is filtered to rows
    whose source file exists under project_dir.

    deps_data is {dep_key: {"fwd": [(target, short_name), ...], "rev": [...]}}.
    """
    if arc_key not in _PROJECT_REGISTRY:
        raise ValueError(f"Unknown project '{arc_key}'. Available: {list(_PROJECT_REGISTRY)}")

    csv_file, deps_file, repo_sub = _PROJECT_REGISTRY[arc_key]
    csv_path  = _DATASETS_DIR / csv_file
    deps_path = _DATASETS_DIR / deps_file

    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    path_col = next((c for c in ["path", "Path", "file", "File"] if c in df.columns), None)
    if path_col is None:
        raise ValueError(f"No path column found in {csv_path}")
    df = df.rename(columns={path_col: "path"}).drop_duplicates(subset=["path"]).reset_index(drop=True)
    df["dep_key"] = df["path"].astype(str)

    # resolve repo root
    if repo_sub is None:
        repo_root = Path(project_dir)
    else:
        repo_root = Path(project_dir) / repo_sub

    # filter to files that actually exist on disk
    df = df[df["dep_key"].map(lambda k: (repo_root / k.lstrip("/\\")).exists())].reset_index(drop=True)

    deps_data = _load_deps(deps_path)
    return df, deps_data


def _load_deps(deps_path: Path) -> Dict:
    if not deps_path.exists():
        return {}
    df_dep = pd.read_csv(deps_path)
    src_col = next((c for c in ["Source_File", "Source", "source", "File", "From"] if c in df_dep.columns), None)
    tgt_col = next((c for c in ["Target_File", "Target", "target", "Dep", "To"]   if c in df_dep.columns), None)
    if not src_col or not tgt_col:
        return {}

    deps: Dict = defaultdict(lambda: {"fwd": [], "rev": []})
    seen: set = set()
    for row in df_dep[[src_col, tgt_col]].itertuples(index=False):
        s = str(getattr(row, src_col))
        t = str(getattr(row, tgt_col))
        key = (s, t)
        if key in seen:
            continue
        seen.add(key)
        short_s = s.split("/")[-1].rsplit(".", 1)[0]
        short_t = t.split("/")[-1].rsplit(".", 1)[0]
        deps[s]["fwd"].append((t, short_t))
        deps[t]["rev"].append((s, short_s))

    return dict(deps)
