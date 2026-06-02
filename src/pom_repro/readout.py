"""OpenPOM atom+bond graph readout helpers."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


def atom_bond_readout(
    g: Any,
    node_encodings: torch.Tensor,
    edge_feats: torch.Tensor,
    project_edge_feats: nn.Module,
    readout_set2set: nn.Module | None = None,
    readout_type: str = "set2set",
) -> torch.Tensor:
    """Fold atom and bond embeddings together, then pool graph features."""

    with g.local_scope():
        g.ndata["node_emb"] = node_encodings
        edge_emb = project_edge_feats(edge_feats)
        g.edata["edge_emb"] = edge_emb

        combined_dim = node_encodings.shape[-1] + edge_emb.shape[-1]
        if g.num_edges() == 0:
            g.ndata["src_msg_sum"] = node_encodings.new_zeros(
                (g.num_nodes(), combined_dim)
            )
        else:

            def message_func(edges: Any) -> dict[str, torch.Tensor]:
                src_msg = torch.cat(
                    (edges.src["node_emb"], edges.data["edge_emb"]),
                    dim=1,
                )
                return {"src_msg": src_msg}

            def reduce_func(nodes: Any) -> dict[str, torch.Tensor]:
                src_msg_sum = torch.sum(nodes.mailbox["src_msg"], dim=1)
                return {"src_msg_sum": src_msg_sum}

            g.send_and_recv(
                g.edges(),
                message_func=message_func,
                reduce_func=reduce_func,
            )

        if readout_type == "set2set":
            if readout_set2set is None:
                raise ValueError("readout_set2set is required for set2set readout.")
            return readout_set2set(g, g.ndata["src_msg_sum"])

        if readout_type == "global_sum_pooling":
            import dgl

            return dgl.sum_nodes(g, "src_msg_sum")

        raise ValueError(f"Unsupported readout_type: {readout_type}")
