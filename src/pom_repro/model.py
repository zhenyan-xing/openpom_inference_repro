"""OpenPOM-compatible MPNNPOM model skeleton."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from dgl.nn.pytorch import Set2Set
except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover
    raise ImportError("pom_repro.model requires DGL.") from exc

from pom_repro.ffn import CustomPositionwiseFeedForward
from pom_repro.mpnn import CustomMPNNGNN
from pom_repro.readout import atom_bond_readout


class MPNNPOM(nn.Module):
    """Checkpoint-shaped reproduction of OpenPOM's MPNNPOM core."""

    def __init__(
        self,
        n_tasks: int = 138,
        node_out_feats: int = 100,
        edge_hidden_feats: int = 75,
        edge_out_feats: int = 100,
        num_step_message_passing: int = 5,
        mpnn_residual: bool = True,
        message_aggregator_type: str = "sum",
        mode: str = "classification",
        number_atom_features: int = 134,
        number_bond_features: int = 6,
        n_classes: int = 1,
        nfeat_name: str = "x",
        efeat_name: str = "edge_attr",
        readout_type: str = "set2set",
        num_step_set2set: int = 3,
        num_layer_set2set: int = 2,
        ffn_hidden_list: Sequence[int] = (392, 392),
        ffn_embeddings: int | None = 256,
        ffn_activation: str = "relu",
        ffn_dropout_p: float = 0.12,
        ffn_dropout_at_input_no_act: bool = False,
    ) -> None:
        if mode not in ["classification", "regression"]:
            raise ValueError("mode must be either 'classification' or 'regression'.")

        super().__init__()

        self.n_tasks = n_tasks
        self.mode = mode
        self.n_classes = n_classes
        self.nfeat_name = nfeat_name
        self.efeat_name = efeat_name
        self.readout_type = readout_type
        self.ffn_embeddings = ffn_embeddings
        self.ffn_activation = ffn_activation
        self.ffn_dropout_p = ffn_dropout_p

        if mode == "classification":
            self.ffn_output = n_tasks * n_classes
        else:
            self.ffn_output = n_tasks

        self.mpnn = CustomMPNNGNN(
            node_in_feats=number_atom_features,
            node_out_feats=node_out_feats,
            edge_in_feats=number_bond_features,
            edge_hidden_feats=edge_hidden_feats,
            num_step_message_passing=num_step_message_passing,
            residual=mpnn_residual,
            message_aggregator_type=message_aggregator_type,
        )

        self.project_edge_feats = nn.Sequential(
            nn.Linear(number_bond_features, edge_out_feats),
            nn.ReLU(),
        )

        if readout_type == "set2set":
            self.readout_set2set = Set2Set(
                input_dim=node_out_feats + edge_out_feats,
                n_iters=num_step_set2set,
                n_layers=num_layer_set2set,
            )
            ffn_input = 2 * (node_out_feats + edge_out_feats)
        elif readout_type == "global_sum_pooling":
            ffn_input = node_out_feats + edge_out_feats
        else:
            raise ValueError("readout_type must be 'set2set' or 'global_sum_pooling'.")

        d_hidden_list = list(ffn_hidden_list)
        if ffn_embeddings is not None:
            d_hidden_list = d_hidden_list + [ffn_embeddings]

        self.ffn = CustomPositionwiseFeedForward(
            d_input=ffn_input,
            d_hidden_list=d_hidden_list,
            d_output=self.ffn_output,
            activation=ffn_activation,
            dropout_p=ffn_dropout_p,
            dropout_at_input_no_act=ffn_dropout_at_input_no_act,
        )

    def _readout(
        self,
        g: Any,
        node_encodings: torch.Tensor,
        edge_feats: torch.Tensor,
    ) -> torch.Tensor:
        return atom_bond_readout(
            g=g,
            node_encodings=node_encodings,
            edge_feats=edge_feats,
            project_edge_feats=self.project_edge_feats,
            readout_set2set=getattr(self, "readout_set2set", None),
            readout_type=self.readout_type,
        )

    def forward(
        self,
        g: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None] | torch.Tensor:
        node_feats = g.ndata[self.nfeat_name]
        edge_feats = g.edata[self.efeat_name]

        node_encodings = self.mpnn(g, node_feats, edge_feats)
        molecular_encodings = self._readout(g, node_encodings, edge_feats)

        if self.readout_type == "global_sum_pooling":
            molecular_encodings = F.softmax(molecular_encodings, dim=1)

        embeddings, out = self.ffn(molecular_encodings)

        if self.mode == "classification":
            if self.n_tasks == 1:
                logits = out.view(-1, self.n_classes)
            else:
                logits = out.view(-1, self.n_tasks, self.n_classes)
            proba = torch.sigmoid(logits)
            if self.n_classes == 1:
                proba = proba.squeeze(-1)
            return proba, logits, embeddings

        return out
