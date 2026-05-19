"""OpenPOM-compatible feed-forward prediction head."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import torch
import torch.nn as nn


class CustomPositionwiseFeedForward(nn.Module):
    """Feed-forward network returning the POM embedding and final output."""

    def __init__(
        self,
        d_input: int = 400,
        d_hidden_list: Sequence[int] = (392, 392, 256),
        d_output: int = 138,
        activation: str = "relu",
        dropout_p: float = 0.12,
        dropout_at_input_no_act: bool = False,
        batch_norm: bool = True,
    ) -> None:
        super().__init__()

        self.dropout_at_input_no_act = dropout_at_input_no_act
        self.batch_norm = batch_norm

        self.activation: Callable[[Any], Any]
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "leakyrelu":
            self.activation = nn.LeakyReLU(0.1)
        elif activation == "prelu":
            self.activation = nn.PReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "selu":
            self.activation = nn.SELU()
        elif activation == "elu":
            self.activation = nn.ELU()
        elif activation == "linear":
            self.activation = lambda x: x
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        hidden_dims = list(d_hidden_list)
        d_output = d_output if d_output != 0 else d_input
        self.n_layers = len(hidden_dims) + 1

        if self.n_layers == 1:
            linears = [nn.Linear(d_input, d_output)]
        else:
            linears = [nn.Linear(d_input, hidden_dims[0])]
            for idx in range(1, len(hidden_dims)):
                linears.append(nn.Linear(hidden_dims[idx - 1], hidden_dims[idx]))
            linears.append(nn.Linear(hidden_dims[-1], d_output))

        self.linears = nn.ModuleList(linears)
        dropout_layer = nn.Dropout(dropout_p)
        self.dropout_p = nn.ModuleList(
            [dropout_layer for _ in range(self.n_layers)]
        )

        if batch_norm:
            self.batchnorms = nn.ModuleList(
                [nn.BatchNorm1d(hidden_dim) for hidden_dim in hidden_dims]
            )

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        if self.n_layers == 1:
            if self.dropout_at_input_no_act:
                return None, self.linears[0](self.dropout_p[0](x))
            return None, self.dropout_p[0](self.activation(self.linears[0](x)))

        if self.dropout_at_input_no_act:
            x = self.dropout_p[-1](x)

        if self.batch_norm:
            for i in range(self.n_layers - 2):
                x = self.dropout_p[i](
                    self.activation(self.batchnorms[i](self.linears[i](x)))
                )

            embeddings = self.linears[self.n_layers - 2](x)
            x = self.dropout_p[self.n_layers - 2](
                self.activation(self.batchnorms[self.n_layers - 2](embeddings))
            )
        else:
            for i in range(self.n_layers - 2):
                x = self.dropout_p[i](self.activation(self.linears[i](x)))

            embeddings = self.linears[self.n_layers - 2](x)
            x = self.dropout_p[self.n_layers - 2](self.activation(embeddings))

        output = self.linears[-1](x)
        return embeddings, output
