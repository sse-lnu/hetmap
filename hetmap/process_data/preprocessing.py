"""Preprocessor: converts raw df + df_dep DataFrames into graph-ready tokens and structures.

Extracted from the variant scripts. Accepts GraphConfig (from hetmap.experiments.config).
Always uses file_aggregate mode for member nodes (one synthetic node per file).
"""
from __future__ import annotations

import os
import re
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from hetmap.experiments.config import GraphConfig


class Preprocessor:
    def __init__(self, df: pd.DataFrame, df_dep: pd.DataFrame, config: GraphConfig):
        self.df_raw     = df.copy()
        self.df_dep_raw = df_dep.copy()
        self.config     = config

        self.df_files:   pd.DataFrame = pd.DataFrame()
        self.df_members: pd.DataFrame = pd.DataFrame()
        self.df_dep:     pd.DataFrame = pd.DataFrame()
        self.df_dep_file: pd.DataFrame = pd.DataFrame()

        self.file_folder_tokens:          Dict[str, List[str]] = {}
        self.folder_hierarchy:            Dict[str, List[str]] = {}
        self.folder_tokens:               Dict[str, List[str]] = {}
        self.file_tokens:                 Dict[str, List[str]] = {}
        self.member_tokens:               Dict[int, List[str]] = {}
        self.member_file_map:             Dict[int, str]       = {}
        self.file_to_aggregate_member_id: Dict[str, int]       = {}

        self.file_dict:   List[str] = []
        self.folder_dict: List[str] = []
        self.member_dict: List[str] = []

        self.stop_sw = set(ENGLISH_STOP_WORDS) | {
            "get", "set", "add", "remove", "put", "new", "value", "class", "void", "boolean",
            "object", "default", "true", "false", "manager", "controller", "update",
            "factory", "impl", "result", "for", "if", "init", "logger", "log", "added",
            "sh", "c", "h", "java", "jabref", "jab", "ref", "comment", "delete",
            "create", "save", "change", "copy",
        }
        self._token_cache: Dict[Tuple[str, str], List[str]] = {}

    def preprocess(self) -> None:
        self._clean_dataframes()
        self._extract_folders()
        self._extract_file_tokens()
        self._extract_member_tokens()
        self._build_vocabularies()

    @staticmethod
    def _as_bool(value) -> bool:
        if isinstance(value, bool):
            return value
        if pd.isna(value):
            return False
        return str(value).strip().lower() in {"true", "1", "yes", "y", "t"}

    def _clean_dataframes(self) -> None:
        df = self.df_raw.copy()
        df = df.dropna(subset=["File"]).reset_index(drop=True)
        df["File"] = df["File"].astype(str)
        if "Member_ID" in df.columns:
            df["Member_ID"] = pd.to_numeric(df["Member_ID"], errors="coerce")
        if "Member_Type" in df.columns:
            df["Member_Type"] = df["Member_Type"].replace({"type": "type_member"})

        # Only "File" is required — a file with no "Module" (no ground truth, e.g. a
        # newly added/unmapped file) still becomes a graph node so it can be predicted
        # on; its Module stays NaN rather than being dropped or coerced to "nan".
        self.df_files = (
            df[["File", "Module"]].dropna(subset=["File"])
            .drop_duplicates(subset=["File"]).reset_index(drop=True)
        )
        self.df_files["File"] = self.df_files["File"].astype(str)

        member_cols = [c for c in ["File", "Member_ID", "Member_Name", "Member_Type"] if c in df.columns]
        if "Member_ID" in member_cols:
            self.df_members = (
                df[member_cols].dropna(subset=["File", "Member_ID"])
                .drop_duplicates(subset=["File", "Member_ID"]).reset_index(drop=True)
            )
            self.df_members["File"]      = self.df_members["File"].astype(str)
            self.df_members["Member_ID"] = self.df_members["Member_ID"].astype("Int64")
            if "Member_Name" not in self.df_members.columns:
                self.df_members["Member_Name"] = ""
            if "Member_Type" not in self.df_members.columns:
                self.df_members["Member_Type"] = "member"
            self.df_members["Member_Type"] = self.df_members["Member_Type"].replace({"type": "type_member"})
        else:
            self.df_members = pd.DataFrame(columns=["File", "Member_ID", "Member_Name", "Member_Type"])

        valid_files = set(self.df_files["File"].astype(str))
        dep = self.df_dep_raw.copy()
        dep["Source_File"] = dep["Source_File"].astype(str)
        dep["Target_File"] = dep["Target_File"].astype(str)
        dep = dep[
            dep["Source_File"].isin(valid_files) & dep["Target_File"].isin(valid_files)
        ].reset_index(drop=True)

        if self.config.exclude_dep_types:
            dep = dep[~dep["Dependency_Type"].isin(self.config.exclude_dep_types)].reset_index(drop=True)

        for col in ["Source_Member_ID", "Target_Member_ID"]:
            dep[col] = (
                pd.to_numeric(dep.get(col, pd.Series(dtype=float)), errors="coerce")
                if col in dep.columns else np.nan
            )
        for col in ["Source_Member_Type", "Target_Member_Type"]:
            if col in dep.columns:
                dep[col] = dep[col].fillna("file").replace({"type": "type_member"})
            else:
                dep[col] = "file"
        for col in ["Source_Member", "Target_Member"]:
            if col not in dep.columns:
                dep[col] = ""
        if "Is_Member_Level" not in dep.columns:
            dep["Is_Member_Level"] = False
        dep["Is_Member_Level"] = dep["Is_Member_Level"].map(self._as_bool)
        if "Dependency_Count" in dep.columns:
            dep["Dependency_Count"] = pd.to_numeric(dep["Dependency_Count"], errors="coerce").fillna(1).astype(int)
        else:
            dep["Dependency_Count"] = 1

        self.df_dep      = dep
        self.df_dep_file = self._collapse_dependencies_to_file_level(dep)

    @staticmethod
    def _collapse_dependencies_to_file_level(dep: pd.DataFrame) -> pd.DataFrame:
        if dep.empty:
            return pd.DataFrame(columns=["Source_File", "Target_File", "Dependency_Type", "Dependency_Count"])
        return (
            dep.groupby(["Source_File", "Target_File", "Dependency_Type"], as_index=False, dropna=False)
            ["Dependency_Count"].sum().reset_index(drop=True)
        )

    def _extract_folders(self) -> None:
        files = self.df_files["File"].astype(str).tolist()
        folder_segs = {
            f: [p.lower() for p in f.replace("\\", "/").split("/")[:-1] if p]
            for f in files
        }
        token_counts = Counter(tok for segs in folder_segs.values() for tok in segs)
        common   = {tok for tok, cnt in token_counts.items() if cnt == len(files)} if files else set()
        to_remove = common | {"src", "main", "java"}
        cleaned: Dict[str, List[str]] = {}
        for f, segs in folder_segs.items():
            dedup: List[str] = []
            for seg in [s for s in segs if s not in to_remove]:
                if not dedup or dedup[-1] != seg:
                    dedup.append(seg)
            cleaned[f] = dedup
        self._build_folder_hierarchy(cleaned)

    def _build_folder_hierarchy(self, cleaned: Dict[str, List[str]]) -> None:
        folder_hierarchy:   Dict[str, List[str]] = {}
        folder_tokens:      Dict[str, List[str]] = {}
        file_folder_tokens: Dict[str, List[str]] = {}
        for f, toks in cleaned.items():
            if not toks:
                folder_hierarchy[f]   = ["__root__"]
                folder_tokens["__root__"] = ["__root__"]
                file_folder_tokens[f] = []
                continue
            levels = []
            for i in range(1, len(toks) + 1):
                node = "/".join(toks[:i])
                levels.append(node)
                folder_tokens[node] = toks[:i]
            folder_hierarchy[f]   = levels
            file_folder_tokens[f] = toks
        self.folder_hierarchy   = folder_hierarchy
        self.folder_tokens      = folder_tokens
        self.file_folder_tokens = file_folder_tokens

    def _extract_file_tokens(self) -> None:
        if not self.df_members.empty and "Member_Name" in self.df_members.columns:
            members_per_file = (
                self.df_members.dropna(subset=["Member_Name"]).groupby("File")["Member_Name"]
                .apply(lambda v: sorted({str(x) for x in v if str(x).strip()})).to_dict()
            )
        else:
            members_per_file = {}
        file_tokens: Dict[str, List[str]] = {}
        for f in self.df_files["File"].astype(str):
            fname = os.path.splitext(os.path.basename(f))[0].lower()
            toks  = list(self._tokenize_code(fname, f) or [fname])
            if self.config.add_members_to_file_features:
                for mn in members_per_file.get(f, []):
                    toks.extend(self._tokenize_code(str(mn), f))
            file_tokens[f] = sorted({t for t in toks if not str(t).isdigit()})
        self.file_tokens = file_tokens

    def _extract_member_tokens(self) -> None:
        # Always use file_aggregate: one synthetic member node per file,
        # token-bag from all member names in that file.
        self._extract_aggregate_member_tokens()

    def _extract_aggregate_member_tokens(self) -> None:
        if not self.df_members.empty and "Member_Name" in self.df_members.columns:
            members_per_file = (
                self.df_members.dropna(subset=["Member_Name"]).groupby("File")["Member_Name"]
                .apply(lambda v: sorted({str(x) for x in v if str(x).strip()})).to_dict()
            )
        else:
            members_per_file = {}
        member_tokens:  Dict[int, List[str]] = {}
        member_file_map: Dict[int, str]      = {}
        file_to_agg:    Dict[str, int]       = {}
        for agg_id, f in enumerate(self.df_files["File"].dropna().astype(str).drop_duplicates().tolist()):
            toks: List[str] = []
            for mn in members_per_file.get(f, []):
                toks.extend(self._tokenize_code(str(mn), f))
            if not toks:
                fname = os.path.splitext(os.path.basename(f))[0]
                toks  = self._tokenize_code(fname, f) or [fname.lower()]
            member_tokens[int(agg_id)]  = sorted({t for t in toks if not str(t).isdigit()})
            member_file_map[int(agg_id)] = f
            file_to_agg[f]               = int(agg_id)
        self.member_tokens               = member_tokens
        self.member_file_map             = member_file_map
        self.file_to_aggregate_member_id = file_to_agg

    def _build_vocabularies(self) -> None:
        self.file_dict   = sorted({tok for toks in self.file_tokens.values()   for tok in toks})
        self.folder_dict = sorted({tok for toks in self.folder_tokens.values() for tok in toks})
        self.member_dict = sorted({tok for toks in self.member_tokens.values() for tok in toks})

    def _tokenize_code(self, text: str, file_path: str = "") -> List[str]:
        key = (text, file_path)
        if key in self._token_cache:
            return self._token_cache[key]
        if not isinstance(text, str) or not text.strip():
            return []
        folder_tok = set(self.file_folder_tokens.get(file_path, []))
        s = re.sub(r"[^A-Za-z0-9_]", " ", text)
        s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", s)
        s = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", s)
        s = s.replace("_", " ")
        tokens = [t.lower() for t in s.split() if len(t) > 1]
        out = [t for t in tokens if t in folder_tok or t not in self.stop_sw]
        self._token_cache[key] = out
        return out
