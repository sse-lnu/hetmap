from __future__ import annotations

import torch
import torch.nn as nn


class HGTNet(nn.Module):
    def __init__(
        self,
        data,
        hidden_channels: int,
        out_channels: int,
        heads: int = 8,
        dropout: float = 0.0,
        num_layers: int = 1,
        target_node_type: str = "file",
    ):
        super().__init__()
        from torch_geometric.nn import HGTConv

        self.node_types, self.edge_types = data.metadata()
        self.target_node_type = target_node_type
        self.dropout = nn.Dropout(dropout)

        self.lin_dict = nn.ModuleDict()
        for ntype in self.node_types:
            in_dim = data[ntype].x.size(-1)
            if in_dim <= 0:
                raise ValueError(
                    f"Node type '{ntype}' has zero feature dimension. "
                    "Enable add_constant_feature_if_empty=True or add a feature source."
                )
            self.lin_dict[ntype] = nn.Linear(in_dim, hidden_channels)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                HGTConv(in_channels=hidden_channels, out_channels=hidden_channels,
                        metadata=(self.node_types, self.edge_types), heads=heads)
            )
            self.norms.append(nn.LayerNorm(hidden_channels))

        self.classifier = nn.Linear(hidden_channels, out_channels)

    def forward(self, data, return_embeddings_only: bool = False):
        x_dict = {ntype: self.lin_dict[ntype](data[ntype].x).relu_()
                  for ntype in self.node_types}

        for conv, norm in zip(self.convs, self.norms):
            prev     = x_dict
            out_dict = conv(x_dict, data.edge_index_dict)
            x_dict   = {}
            for ntype in self.node_types:
                h = out_dict.get(ntype, prev[ntype]) + prev[ntype]
                x_dict[ntype] = self.dropout(norm(h))

        if return_embeddings_only:
            return x_dict

        logits = self.classifier(x_dict[self.target_node_type])
        return logits, x_dict
