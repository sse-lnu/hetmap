from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Optional

import networkx as nx
import numpy as np
import pandas as pd
from node2vec import Node2Vec
from sklearn.preprocessing import LabelEncoder


class N2vData:
    def __init__(self, df: pd.DataFrame, df_dep: pd.DataFrame, use_majority_vote: bool = True):
        self.df = df.copy()
        self.df_dep = df_dep.copy()
        self.use_majority_vote = bool(use_majority_vote)

        self.file_to_label: pd.Series = pd.Series(dtype=str)
        self.G: nx.Graph = nx.Graph()
        self.node_list: list[str] = []
        self.nodes: pd.Series = pd.Series(dtype=str)
        self.y_true: Optional[np.ndarray] = None
        self.label_encoder = None
        self.num_classes = 0

        self._build_data()

    def _parse_module_list(self, x):
        if isinstance(x, list):
            vals = x
        elif pd.isna(x):
            vals = []
        else:
            s = str(x).strip()
            if not s:
                vals = []
            else:
                try:
                    vals = ast.literal_eval(s)
                    if not isinstance(vals, (list, tuple, set)):
                        vals = [vals]
                except Exception:
                    vals = [s]

        out = []
        for v in vals:
            v = str(v).strip().lower()
            if v and v not in {"nan", "none", "", "unmapped", "__none__"}:
                out.append(v)
        return sorted(set(out))

    def _build_data(self) -> None:
        if "File" not in self.df.columns:
            raise ValueError("df must contain a 'File' column.")

        d = self.df.copy()
        d["File"] = d["File"].astype(str).str.strip()

        file_order = pd.Index(pd.unique(d["File"]), name="File")

        if "Module" in d.columns:
            d["Module"] = (
                d["Module"]
                .astype(str)
                .str.strip()
                .str.lower()
                .replace({"nan": None, "none": None, "": None, "unmapped": None, "__none__": None})
            )
        else:
            d["Module"] = None

        if "Module_List" in d.columns:
            d["Module_List"] = d["Module_List"].apply(self._parse_module_list)
        else:
            d["Module_List"] = [[] for _ in range(len(d))]

        d["Module_List"] = [
            sorted(set(lst + ([mod] if mod is not None else [])))
            for lst, mod in zip(d["Module_List"], d["Module"])
        ]

        vote_df = d.loc[d["Module"].notna(), ["File", "Module"]].copy()

        if vote_df.empty:
            exploded = d[["File", "Module_List"]].explode("Module_List").rename(
                columns={"Module_List": "Module"}
            )
            exploded["Module"] = exploded["Module"].replace({"__none__": None})
            vote_df = exploded.loc[exploded["Module"].notna(), ["File", "Module"]].copy()

        if self.use_majority_vote:
            c = vote_df.groupby(["File", "Module"], sort=False).size().reset_index(name="cnt")
            c = c.sort_values(["File", "cnt", "Module"], ascending=[True, False, True])
            primary_df = c.drop_duplicates("File", keep="first")[["File", "Module"]]
        else:
            primary_df = vote_df.drop_duplicates("File", keep="first")[["File", "Module"]]

        primary_map = dict(zip(primary_df["File"], primary_df["Module"]))

        nodes_df = (
            d.groupby("File", sort=False, as_index=False)
            .agg(
                Module_List=("Module_List", lambda s: sorted(set(
                    m for lst in s for m in lst if m is not None
                )))
            )
        )

        nodes_df = (
            pd.DataFrame({"File": file_order})
            .merge(nodes_df, on="File", how="left")
        )

        empty = nodes_df["Module_List"].isna() | (nodes_df["Module_List"].apply(len) == 0)
        if empty.any():
            nodes_df.loc[empty, "Module_List"] = [["__none__"]] * int(empty.sum())

        nodes_df["Primary_Module"] = nodes_df["File"].map(primary_map)
        nodes_df["Primary_Module"] = nodes_df.apply(
            lambda r: r["Primary_Module"] if pd.notna(r["Primary_Module"]) else r["Module_List"][0],
            axis=1,
        )
        nodes_df["Module"] = nodes_df["Primary_Module"]
        nodes_df["Duplicated"] = nodes_df["Module_List"].apply(lambda x: len(x) > 1)

        self.df = nodes_df.reset_index(drop=True)

        self.label_encoder = LabelEncoder()
        self.df["Label"] = self.label_encoder.fit_transform(self.df["Primary_Module"].astype(str))
        self.num_classes = len(self.label_encoder.classes_)

        self.file_to_label = self.df.set_index("File")["Primary_Module"]

        keep_list = self.df["File"].astype(str).tolist()
        keep_set = set(keep_list)

        dep = self.df_dep.copy()
        if dep is None or dep.empty:
            dep = pd.DataFrame(columns=["Source_File", "Target_File"])

        if "Source_File" not in dep.columns or "Target_File" not in dep.columns:
            raise ValueError("df_dep must contain 'Source_File' and 'Target_File' columns.")

        dep["Source_File"] = dep["Source_File"].astype(str).str.strip()
        dep["Target_File"] = dep["Target_File"].astype(str).str.strip()
        dep = dep[dep["Source_File"] != dep["Target_File"]]
        dep = dep.drop_duplicates(subset=["Source_File", "Target_File"]).reset_index(drop=True)
        dep = dep[dep["Source_File"].isin(keep_set) & dep["Target_File"].isin(keep_set)].reset_index(drop=True)

        self.df_dep = dep

        G = nx.Graph()
        G.add_nodes_from(keep_list)
        if not dep.empty:
            G.add_edges_from(
                dep[["Source_File", "Target_File"]].itertuples(index=False, name=None)
            )

        self.G = G
        self.node_list = keep_list
        self.nodes = pd.Series(self.node_list, name="File")
        self.y_true = self.df["Label"].to_numpy(dtype=int)


@dataclass
class Node2VecModel:
    dimensions: int = 128
    walk_length: int = 30
    num_walks: int = 200
    window: int = 15
    epochs: int = 5
    p: float = 1.0
    q: float = 1.0
    workers: int = 4
    negative: int = 5

    def fit_transform(self, data: N2vData) -> np.ndarray:
        n2v = Node2Vec(
            data.G,
            dimensions=int(self.dimensions),
            walk_length=int(self.walk_length),
            num_walks=int(self.num_walks),
            p=float(self.p),
            q=float(self.q),
            workers=int(self.workers),
        )

        model = n2v.fit(
            window=int(self.window),
            min_count=1,
            sg=1,
            epochs=int(self.epochs),
            negative=int(self.negative),
            hs=0,
            workers=int(self.workers),
        )

        return np.vstack([model.wv[n] for n in data.node_list]).astype(np.float32)
