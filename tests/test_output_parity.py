from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts/compare_with_openpom.py"
CHECKPOINT_PATH = REPO_ROOT / "checkpoints/openpom_experiments_1_checkpoint2.pt"
REFERENCE_NPZ = REPO_ROOT / "reference_outputs/openpom_reference.npz"


def load_compare_module():
    spec = importlib.util.spec_from_file_location(
        "compare_with_openpom_for_tests",
        SCRIPT_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_metric_helpers_on_synthetic_arrays() -> None:
    compare = load_compare_module()
    reference = {
        "probs": np.asarray([[0.9, 0.8, 0.1, 0.0], [0.4, 0.3, 0.2, 0.1]]),
        "logits": np.asarray([[[9.0], [8.0], [1.0], [0.0]], [[4.0], [3.0], [2.0], [1.0]]]),
        "embeddings": np.asarray([[1.0, 0.0], [1.0, 1.0]]),
    }
    local = {
        "probs": np.asarray([[0.82, 0.85, 0.1, 0.0], [0.4, 0.29, 0.2, 0.1]]),
        "logits": np.asarray([[[8.5], [8.2], [1.0], [0.0]], [[4.0], [2.9], [2.0], [1.0]]]),
        "embeddings": np.asarray([[1.0, 0.0], [2.0, 2.0]]),
    }
    labels = ["a", "b", "c", "d"]

    metrics, comparisons = compare.compute_metrics(
        local=local,
        reference=reference,
        labels=labels,
        top_k=2,
    )

    assert metrics["probs_max_abs_diff"] == pytest.approx(0.08)
    assert metrics["probs_mean_abs_diff"] == pytest.approx(0.0175)
    assert metrics["logits_max_abs_diff"] == pytest.approx(0.5)
    assert metrics["logits_mean_abs_diff"] == pytest.approx(0.1)
    assert metrics["embedding_cosine_min"] == pytest.approx(1.0)
    assert metrics["embedding_cosine_mean"] == pytest.approx(1.0)
    assert metrics["top10_min_overlap"] == 2
    assert metrics["top10_all_order_match"] is False
    assert comparisons[0]["reference_labels"] == ["a", "b"]
    assert comparisons[0]["local_labels"] == ["b", "a"]
    assert comparisons[0]["overlap_fraction"] == 1.0
    assert comparisons[0]["order_match"] is False


def test_status_thresholds() -> None:
    compare = load_compare_module()

    ideal_metrics = {
        "probs_max_abs_diff": 1e-5,
        "top10_all_order_match": True,
        "embedding_cosine_min": 0.9999995,
        "top10_min_overlap": 10,
    }
    acceptable_metrics = {
        "probs_max_abs_diff": 5e-4,
        "top10_all_order_match": False,
        "embedding_cosine_min": 0.9995,
        "top10_min_overlap": 8,
    }
    fail_metrics = {
        "probs_max_abs_diff": 0.1,
        "top10_all_order_match": False,
        "embedding_cosine_min": 0.5,
        "top10_min_overlap": 2,
    }

    assert compare.classify_status(ideal_metrics, top_k=10) == "ideal"
    assert compare.classify_status(acceptable_metrics, top_k=10) == "acceptable"
    assert compare.classify_status(fail_metrics, top_k=10) == "fail"


def test_real_checkpoint_reference_parity_is_ideal() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("dgl")
    pytest.importorskip("dgllife")
    if not CHECKPOINT_PATH.exists():
        pytest.skip(f"checkpoint not found: {CHECKPOINT_PATH}")
    if not REFERENCE_NPZ.exists():
        pytest.skip(f"reference NPZ not found: {REFERENCE_NPZ}")

    compare = load_compare_module()
    reference = compare.load_reference_npz(REFERENCE_NPZ)
    smiles = [str(item) for item in reference["smiles"].tolist()]
    labels = [str(item) for item in reference["task_labels"].tolist()]

    torch.set_grad_enabled(False)
    local_outputs, checkpoint_report = compare.run_local_model(
        smiles=smiles,
        checkpoint_path=CHECKPOINT_PATH,
        device="cpu",
    )
    metrics, _ = compare.compute_metrics(
        local=local_outputs,
        reference=reference,
        labels=labels,
        top_k=10,
    )

    assert checkpoint_report["strict_load_success"] is True
    assert checkpoint_report["direct_strict_success"] is True
    assert compare.classify_status(metrics, top_k=10) == "ideal"
