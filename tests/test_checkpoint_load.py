from __future__ import annotations

from pathlib import Path

import pytest


CHECKPOINT_PATH = Path("checkpoints/openpom_experiments_1_checkpoint2.pt")


def _require_torch():
    return pytest.importorskip("torch")


def _require_model_deps():
    torch = _require_torch()
    pytest.importorskip("dgl")
    pytest.importorskip("dgllife")
    return torch


def test_extract_state_dict_from_wrapped_checkpoint() -> None:
    torch = _require_torch()
    from pom_repro.checkpoint import extract_state_dict

    state_dict = {"layer.weight": torch.zeros(1)}
    checkpoint = {
        "model_state_dict": state_dict,
        "optimizer_state_dict": {},
        "global_step": 1984,
    }

    extracted, source = extract_state_dict(checkpoint)

    assert extracted is state_dict
    assert source == "wrapped:model_state_dict"


def test_inspect_state_dict_compatibility_reports_key_and_shape_issues() -> None:
    torch = _require_torch()
    from torch import nn

    from pom_repro.checkpoint import inspect_state_dict_compatibility

    model = nn.Sequential(nn.Linear(2, 3))
    checkpoint_state = {
        "0.weight": torch.zeros(2, 2),
        "unexpected.weight": torch.zeros(1),
    }

    report = inspect_state_dict_compatibility(
        model=model,
        state_dict=checkpoint_state,
        checkpoint_path="synthetic.pt",
        state_dict_source="direct_state_dict",
    )

    assert report.checkpoint_path == "synthetic.pt"
    assert report.model_key_count == 2
    assert report.checkpoint_key_count == 2
    assert report.missing_keys == ["0.bias"]
    assert report.unexpected_keys == ["unexpected.weight"]
    assert len(report.shape_mismatches) == 1
    assert report.shape_mismatches[0].key == "0.weight"
    assert report.shape_mismatches[0].model_shape == (3, 2)
    assert report.shape_mismatches[0].checkpoint_shape == (2, 2)


def test_key_remapper_is_not_used_when_direct_strict_load_succeeds(tmp_path) -> None:
    torch = _require_torch()
    from torch import nn

    from pom_repro.checkpoint import load_checkpoint_strict

    checkpoint_path = tmp_path / "checkpoint.pt"
    source_model = nn.Linear(2, 3)
    target_model = nn.Linear(2, 3)
    torch.save({"model_state_dict": source_model.state_dict()}, checkpoint_path)

    remapper_called = False

    def key_remapper(state_dict):
        nonlocal remapper_called
        remapper_called = True
        return state_dict

    report = load_checkpoint_strict(
        target_model,
        checkpoint_path,
        key_remapper=key_remapper,
    )

    assert report.direct_strict_success is True
    assert report.strict_load_success is True
    assert report.remap_used is False
    assert remapper_called is False


def test_real_openpom_checkpoint_loads_strictly_without_remapping() -> None:
    _require_model_deps()
    if not CHECKPOINT_PATH.exists():
        pytest.skip(f"checkpoint not found: {CHECKPOINT_PATH}")

    from pom_repro.checkpoint import load_checkpoint_strict
    from pom_repro.model import MPNNPOM

    model = MPNNPOM()
    report = load_checkpoint_strict(model, CHECKPOINT_PATH)

    assert report.strict_load_success is True
    assert report.direct_strict_success is True
    assert report.remap_used is False
    assert report.model_key_count == 44
    assert report.checkpoint_key_count == 44
    assert report.missing_keys == []
    assert report.unexpected_keys == []
    assert report.shape_mismatches == []


def test_forward_pass_is_finite_after_strict_checkpoint_load() -> None:
    torch = _require_model_deps()
    if not CHECKPOINT_PATH.exists():
        pytest.skip(f"checkpoint not found: {CHECKPOINT_PATH}")

    from pom_repro.checkpoint import load_checkpoint_strict
    from pom_repro.featurizer import GraphFeaturizer
    from pom_repro.graph_batch import batch_graphs
    from pom_repro.model import MPNNPOM

    graphs = GraphFeaturizer().featurize(["C", "CCO"])
    batched_graph = batch_graphs(graphs)

    model = MPNNPOM()
    load_checkpoint_strict(model, CHECKPOINT_PATH)
    model.eval()

    with torch.no_grad():
        proba, logits, embeddings = model(batched_graph)

    assert tuple(proba.shape) == (2, 138)
    assert tuple(logits.shape) == (2, 138, 1)
    assert tuple(embeddings.shape) == (2, 256)
    assert torch.isfinite(proba).all()
    assert torch.isfinite(logits).all()
    assert torch.isfinite(embeddings).all()
