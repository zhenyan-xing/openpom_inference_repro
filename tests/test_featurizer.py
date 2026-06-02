from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from pom_repro.featurizer import GraphConvConstants, GraphFeaturizer
from pom_repro.graph_batch import batch_graphs


REPO_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_JSON = REPO_ROOT / "reference_outputs/openpom_reference.json"
DEFAULT_SMILES = [
    "C",
    "CCO",
    "CC(=O)O",
    "c1ccccc1",
    "CC(C)O",
    "C1=CC=CC=C1O",
]


def sha256_array(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.shape).encode("utf-8"))
    digest.update(str(contiguous.dtype).encode("utf-8"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def max_abs_diff(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        return float("inf")
    if left.size == 0 and right.size == 0:
        return 0.0
    return float(np.max(np.abs(left - right)))


def test_basic_graph_shapes() -> None:
    graphs = GraphFeaturizer().featurize(["C", "CCO"])
    methane = graphs[0]
    ethanol = graphs[1]

    assert methane.node_features.shape == (1, GraphConvConstants.ATOM_FDIM)
    assert methane.edge_features.shape == (0, GraphConvConstants.BOND_FDIM)
    assert methane.edge_index.shape == (2, 0)

    assert ethanol.node_features.shape == (3, GraphConvConstants.ATOM_FDIM)
    assert ethanol.edge_features.shape == (4, GraphConvConstants.BOND_FDIM)
    assert ethanol.edge_index.shape == (2, 4)


def test_cco_edge_order_and_single_bond_features() -> None:
    graph = GraphFeaturizer().featurize(["CCO"])[0]

    np.testing.assert_array_equal(
        graph.edge_index,
        np.asarray([[0, 2, 2, 1], [2, 0, 1, 2]], dtype=int),
    )
    np.testing.assert_array_equal(
        graph.edge_features,
        np.asarray(
            [
                [0, 1, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
            ],
            dtype=float,
        ),
    )


def test_benzene_aromatic_ring_bond_features() -> None:
    graph = GraphFeaturizer().featurize(["c1ccccc1"])[0]
    expected_row = np.asarray([0, 0, 0, 0, 1, 1], dtype=float)

    assert graph.edge_features.shape == (12, GraphConvConstants.BOND_FDIM)
    np.testing.assert_array_equal(
        graph.edge_features,
        np.tile(expected_row, (12, 1)),
    )


def test_reference_output_hashes_match_phase1_export() -> None:
    payload = json.loads(REFERENCE_JSON.read_text(encoding="utf-8"))
    smiles = [sample["smiles"] for sample in payload["samples"]]
    graphs = GraphFeaturizer().featurize(smiles)

    assert smiles == DEFAULT_SMILES
    for graph, sample in zip(graphs, payload["samples"], strict=True):
        summary = sample["graph_summary"]
        assert list(graph.node_features.shape) == summary["node_feature_shape"]
        assert list(graph.edge_features.shape) == summary["edge_feature_shape"]
        assert list(graph.edge_index.shape) == summary["edge_index_shape"]
        assert sha256_array(graph.node_features) == summary["node_feature_sha256"]
        assert sha256_array(graph.edge_features) == summary["edge_feature_sha256"]
        assert sha256_array(graph.edge_index) == summary["edge_index_sha256"]


def import_openpom_featurizer() -> type[Any]:
    openpom_root = Path("/home/xing/openpom")
    if not openpom_root.exists():
        pytest.skip(f"OpenPOM root is unavailable: {openpom_root}")
    sys.path.insert(0, str(openpom_root))
    try:
        from openpom.feat.graph_featurizer import (
            GraphFeaturizer as OpenPOMGraphFeaturizer,
        )
    except Exception as exc:  # pragma: no cover - only used when deps are absent
        pytest.skip(f"OpenPOM featurizer is unavailable: {exc}")
    return OpenPOMGraphFeaturizer


def test_live_openpom_feature_parity() -> None:
    openpom_featurizer_cls = import_openpom_featurizer()
    local_graphs = GraphFeaturizer().featurize(DEFAULT_SMILES)
    openpom_graphs = openpom_featurizer_cls().featurize(DEFAULT_SMILES)

    for local_graph, openpom_graph in zip(local_graphs, openpom_graphs, strict=True):
        local_nodes = np.asarray(local_graph.node_features)
        local_edges = np.asarray(local_graph.edge_features)
        local_edge_index = np.asarray(local_graph.edge_index)
        ref_nodes = np.asarray(openpom_graph.node_features)
        ref_edges = np.asarray(openpom_graph.edge_features)
        ref_edge_index = np.asarray(openpom_graph.edge_index)

        assert max_abs_diff(local_nodes, ref_nodes) == 0
        assert max_abs_diff(local_edges, ref_edges) == 0
        np.testing.assert_array_equal(local_edge_index, ref_edge_index)


def test_dgl_batching_fields_and_shapes() -> None:
    pytest.importorskip("dgl")
    pytest.importorskip("torch")

    graphs = GraphFeaturizer().featurize(["C", "CCO"])
    batched_graph = batch_graphs(graphs)

    assert "x" in batched_graph.ndata
    assert "edge_attr" in batched_graph.edata
    assert tuple(batched_graph.ndata["x"].shape) == (
        4,
        GraphConvConstants.ATOM_FDIM,
    )
    assert tuple(batched_graph.edata["edge_attr"].shape) == (
        4,
        GraphConvConstants.BOND_FDIM,
    )
