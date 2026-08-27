"""
collect_results.py
==================
Produces one collective all_results.csv per results root:

  results/HGT_UniXcoder/all_results.csv
  results/HGT_W2V/all_results.csv      (includes GCN + GAT flat files)
  results/HGT_Node2Vec/all_results.csv

Each row = one run. Key columns: method, dataset, f1_micro, f1_macro,
precision_micro, precision_macro, recall_micro, recall_macro.

Run:
    python results/collect_results.py
"""

from pathlib import Path
import pandas as pd

RESULTS_BASE = Path(__file__).parent

SCORE_COLS = [
    "f1_micro", "f1_macro",
    "precision_micro", "precision_macro",
    "recall_micro", "recall_macro",
]

# GCN/GAT CSVs use different dataset name spellings — map to HetMap canonical names
DATASET_NAME_MAP = {
    "C_Img":     "C.Img",
    "Jabref":     "JabRef",
    "ArgoUML":   "A.UML",
    "SweetHome": "SH-3D",
    "SH3D":      "SH-3D",
    "TeamMates": "T.Mates",
    "argouml":   "A.UML",
    "commons":   "C.Img",
    "sweethome3D": "SH-3D",
}


def normalise_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Convert 0-100 scores to 0-1 if needed."""
    for col in SCORE_COLS:
        if col in df.columns and df[col].median() > 1:
            df[col] = df[col] / 100.0
    return df


def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to match the shared schema."""
    renames = {}
    # capital-D Dataset → dataset
    if "Dataset" in df.columns and "dataset" not in df.columns:
        renames["Dataset"] = "dataset"
    # GCN/GAT have precision_macro / recall_macro but not micro versions
    # copy macro → micro when micro is missing so downstream analysis can use either
    df = df.rename(columns=renames)
    if "precision_micro" not in df.columns and "precision_macro" in df.columns:
        df["precision_micro"] = df["precision_macro"]
    if "recall_micro" not in df.columns and "recall_macro" in df.columns:
        df["recall_micro"] = df["recall_macro"]
    return df


def normalise_dataset_names(df: pd.DataFrame) -> pd.DataFrame:
    if "dataset" in df.columns:
        df["dataset"] = df["dataset"].map(lambda x: DATASET_NAME_MAP.get(x, x))
    return df


def load_flat_csv(path: Path, method_name: str) -> pd.DataFrame:
    """Load a single flat CSV (all datasets in one file, e.g. gcn_results.csv)."""
    df = pd.read_csv(path)
    df = normalise_columns(df)
    df = normalise_dataset_names(df)
    df = normalise_scores(df)
    df.insert(0, "method", method_name)
    return df


def collect_root(name: str, root: Path, skip: set, flat_files: dict, output_name: str) -> None:
    """
    flat_files: {filename_stem: method_display_name} for flat CSVs sitting directly
                in the root (not inside a subfolder), e.g. {"gcn_results": "GCN"}.
    """
    if not root.exists():
        print(f"  [SKIP] {root} does not exist")
        return

    all_frames = []

    # ── subfolder-based methods ───────────────────────────────────────────────
    for method_dir in sorted(root.iterdir()):
        if not method_dir.is_dir():
            continue
        if method_dir.name in skip:
            continue
        if method_dir.name.startswith("_"):
            continue

        csvs = [
            p for p in sorted(method_dir.glob("*.csv"))
            if "_predictions" not in p.name
        ]
        if not csvs:
            continue

        frames = []
        for csv in csvs:
            try:
                if csv == method_dir / "Ant.csv":
                    continue 
                df = pd.read_csv(csv)
                df = normalise_columns(df)
                df = normalise_dataset_names(df)
                df = normalise_scores(df)
                frames.append(df)
            except Exception as e:
                print(f"    WARN: could not read {csv.name}: {e}")

        if not frames:
            continue

        combined = pd.concat(frames, ignore_index=True)

        # ensure dataset column (infer from filename if missing)
        if "dataset" not in combined.columns:
            parts = []
            for csv, df in zip(csvs, frames):
                df = df.copy()
                if "dataset" not in df.columns:
                    df["dataset"] = DATASET_NAME_MAP.get(csv.stem, csv.stem)

                parts.append(df)
            combined = pd.concat(parts, ignore_index=True)

        combined.insert(0, "method", method_dir.name)
        all_frames.append(combined)
        n_ds = combined["dataset"].nunique() if "dataset" in combined.columns else "?"
        print(f"  [{method_dir.name}]  {len(combined)} runs  {n_ds} datasets")

    # ── flat CSV files sitting directly in root ───────────────────────────────
    for stem, method_name in flat_files.items():
        csv_path = root / f"{stem}.csv"
        if not csv_path.exists():
            print(f"  [SKIP flat] {csv_path.name} not found")
            continue
        try:
            df = load_flat_csv(csv_path, method_name)
            all_frames.append(df)
            n_ds = df["dataset"].nunique() if "dataset" in df.columns else "?"
            print(f"  [{method_name}] (flat)  {len(df)} runs  {n_ds} datasets")
        except Exception as e:
            print(f"  WARN: could not load {csv_path.name}: {e}")

    if not all_frames:
        print(f"  No data found in {root}")
        return

    out = pd.concat(all_frames, ignore_index=True)

    # put key columns first
    key_cols = ["method", "dataset"] + SCORE_COLS
    extra_cols = [c for c in out.columns if c not in key_cols]
    out = out[key_cols + extra_cols]

    # skip rows that are themselves an all_results.csv (avoid double-counting on re-run)
    out_path = root / output_name
    out.to_csv(out_path, index=False)
    print(f"\n  => Saved {out_path}  ({len(out)} rows, {out['method'].nunique()} methods)\n")


ROOTS = {
    "HGT_UXC": {
        "path":       RESULTS_BASE / "HGT_UXC",
        "skip":       {"Whole_Embedding_Results"},
        "flat_files": {},
        "output":     "all_results.csv",
    },
    "HGT_W2V": {
        "path":       RESULTS_BASE / "HGT_W2V",
        "skip":       {"others"},
        "flat_files": {
            "gcn_results":          "GCN",
            "gat_results":          "GAT",
            "mlp_results":          "MLP_FT",

        },
        "output":     "all_results.csv",
    },
    "HGT_N2V": {
        "path":       RESULTS_BASE / "HGT_N2V",
        "skip":       set(),
        "flat_files": {},
        "output":     "all_results.csv",
    },
}


def main() -> None:
    for name, cfg in ROOTS.items():
        print(f"\n{'='*55}")
        print(f"  {name}")
        print(f"{'='*55}")
        collect_root(
            name,
            cfg["path"],
            cfg["skip"],
            cfg.get("flat_files", {}),
            cfg["output"],
        )
    print("Done.")


if __name__ == "__main__":
    main()
