"""DGL graph conversion and batching helpers."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from pom_repro.featurizer import GraphData


def to_dgl_graph(graph: GraphData, self_loop: bool = False):
    """Convert a local :class:`GraphData` object to a DGL graph."""

    return graph.to_dgl_graph(self_loop=self_loop)


def batch_graphs(
    graphs: Iterable[GraphData],
    self_loop: bool = False,
    device: str | Any | None = None,
):
    """Convert and batch local graphs with DGL."""

    try:
        import dgl
    except ModuleNotFoundError as exc:
        raise ImportError("This function requires DGL.") from exc

    dgl_graphs = [to_dgl_graph(graph, self_loop=self_loop) for graph in graphs]
    batched_graph = dgl.batch(dgl_graphs)
    if device is not None:
        batched_graph = batched_graph.to(device)
    return batched_graph
