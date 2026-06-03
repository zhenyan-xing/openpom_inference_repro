"""MLP heads for frozen molecular language model embeddings."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn


class FrozenLMHead(nn.Module):
    """Small multilabel MLP head on top of frozen molecular LM embeddings."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 512,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")
        if output_dim <= 0:
            raise ValueError("output_dim must be positive.")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")
        if dropout < 0.0 or dropout >= 1.0:
            raise ValueError("dropout must be in [0, 1).")

        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)

        self.net = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.output_dim),
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.net(embeddings)


def top_k_predictions(
    probs: Sequence[Sequence[float]],
    labels: Sequence[str],
    top_k: int,
) -> list[list[dict[str, float | int | str]]]:
    """Return sorted top-k label predictions for each probability row."""

    if top_k <= 0:
        raise ValueError("top_k must be positive.")
    if top_k > len(labels):
        raise ValueError(f"top_k={top_k} exceeds number of labels={len(labels)}.")

    import numpy as np

    probs_array = np.asarray(probs, dtype=np.float64)
    if probs_array.ndim != 2:
        raise ValueError(f"Expected 2-D probs, got shape {probs_array.shape}.")
    if probs_array.shape[1] != len(labels):
        raise ValueError(
            "Probability width does not match labels: "
            f"{probs_array.shape[1]} vs {len(labels)}."
        )

    top_indices = np.argsort(-probs_array, axis=1, kind="stable")[:, :top_k]
    rows: list[list[dict[str, float | int | str]]] = []
    for row_probs, row_indices in zip(probs_array, top_indices, strict=True):
        rows.append(
            [
                {
                    "index": int(index),
                    "label": str(labels[int(index)]),
                    "prob": float(row_probs[int(index)]),
                }
                for index in row_indices
            ]
        )
    return rows
