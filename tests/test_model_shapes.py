from __future__ import annotations

import pytest

from pom_repro.featurizer import GraphFeaturizer
from pom_repro.graph_batch import batch_graphs


def _require_model_deps():
    torch = pytest.importorskip("torch")
    pytest.importorskip("dgl")
    pytest.importorskip("dgllife")
    return torch


def test_model_forward_shapes_and_finite_outputs() -> None:
    torch = _require_model_deps()
    from pom_repro.model import MPNNPOM

    graphs = GraphFeaturizer().featurize(["C", "CCO"])
    batched_graph = batch_graphs(graphs)

    model = MPNNPOM()
    model.eval()

    with torch.no_grad():
        proba, logits, embeddings = model(batched_graph)

    assert tuple(proba.shape) == (2, 138)
    assert tuple(logits.shape) == (2, 138, 1)
    assert tuple(embeddings.shape) == (2, 256)

    assert torch.isfinite(proba).all()
    assert torch.isfinite(logits).all()
    assert torch.isfinite(embeddings).all()


def test_model_state_dict_checkpoint_shape_contract() -> None:
    _require_model_deps()
    from pom_repro.model import MPNNPOM

    state_dict = MPNNPOM().state_dict()
    expected_shapes = {
        "mpnn.project_node_feats.0.weight": (100, 134),
        "mpnn.project_node_feats.0.bias": (100,),
        "mpnn.gnn_layer.bias": (100,),
        "mpnn.gnn_layer.edge_func.0.weight": (75, 6),
        "mpnn.gnn_layer.edge_func.0.bias": (75,),
        "mpnn.gnn_layer.edge_func.2.weight": (10000, 75),
        "mpnn.gnn_layer.edge_func.2.bias": (10000,),
        "mpnn.gru.weight_ih_l0": (300, 100),
        "mpnn.gru.weight_hh_l0": (300, 100),
        "mpnn.gru.bias_ih_l0": (300,),
        "mpnn.gru.bias_hh_l0": (300,),
        "project_edge_feats.0.weight": (100, 6),
        "project_edge_feats.0.bias": (100,),
        "readout_set2set.lstm.weight_ih_l0": (800, 400),
        "readout_set2set.lstm.weight_hh_l0": (800, 200),
        "readout_set2set.lstm.weight_ih_l1": (800, 200),
        "readout_set2set.lstm.weight_hh_l1": (800, 200),
        "ffn.linears.0.weight": (392, 400),
        "ffn.linears.0.bias": (392,),
        "ffn.linears.1.weight": (392, 392),
        "ffn.linears.1.bias": (392,),
        "ffn.linears.2.weight": (256, 392),
        "ffn.linears.2.bias": (256,),
        "ffn.linears.3.weight": (138, 256),
        "ffn.linears.3.bias": (138,),
    }

    assert len(state_dict) == 44
    for key, shape in expected_shapes.items():
        assert key in state_dict
        assert tuple(state_dict[key].shape) == shape
