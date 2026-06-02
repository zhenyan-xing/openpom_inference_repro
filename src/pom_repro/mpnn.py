"""OpenPOM-compatible MPNN message passing module."""

from __future__ import annotations

import torch.nn as nn

try:
    from dgl.nn.pytorch import NNConv
    from dgllife.model.gnn import MPNNGNN
except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover
    raise ImportError("pom_repro.mpnn requires DGL and DGL-LifeSci.") from exc


class CustomMPNNGNN(MPNNGNN):
    """OpenPOM's MPNNGNN variant with configurable NNConv aggregation."""

    def __init__(
        self,
        node_in_feats: int = 134,
        edge_in_feats: int = 6,
        node_out_feats: int = 100,
        edge_hidden_feats: int = 75,
        num_step_message_passing: int = 5,
        residual: bool = True,
        message_aggregator_type: str = "sum",
    ) -> None:
        super().__init__(
            node_in_feats=node_in_feats,
            edge_in_feats=edge_in_feats,
            node_out_feats=node_out_feats,
            edge_hidden_feats=edge_hidden_feats,
            num_step_message_passing=num_step_message_passing,
        )

        edge_network = nn.Sequential(
            nn.Linear(edge_in_feats, edge_hidden_feats),
            nn.ReLU(),
            nn.Linear(edge_hidden_feats, node_out_feats * node_out_feats),
        )
        self.gnn_layer = NNConv(
            in_feats=node_out_feats,
            out_feats=node_out_feats,
            edge_func=edge_network,
            aggregator_type=message_aggregator_type,
            residual=residual,
        )
