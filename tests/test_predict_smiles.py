from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from pom_repro.predict import (
    ODOR_LABELS,
    predict_smiles,
    resolve_checkpoint_path,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_PATH = REPO_ROOT / "checkpoints/openpom_experiments_1_checkpoint2.pt"
OPENPOM_POINTER_PATH = Path(
    "/home/xing/openpom/models/ensemble_models/experiments_1/checkpoint2.pt"
)
SCRIPT_PATH = REPO_ROOT / "scripts/predict_smiles.py"
ENV_CHECKPOINTS = "POM_REPRO_ENSEMBLE_CHECKPOINTS"


def _require_torch():
    return pytest.importorskip("torch")


def _require_model_deps():
    torch = _require_torch()
    pytest.importorskip("dgl")
    pytest.importorskip("dgllife")
    return torch


def _require_checkpoint() -> None:
    if not CHECKPOINT_PATH.exists():
        pytest.skip(f"checkpoint not found: {CHECKPOINT_PATH}")


def _assert_prediction_contract(
    result: dict[str, Any],
    num_smiles: int,
    top_k: int,
    expect_embedding: bool,
) -> None:
    assert set(result) == {
        "smiles",
        "labels",
        "probs",
        "top_k",
        "checkpoint_paths",
        "embedding",
    }
    assert result["labels"] == list(ODOR_LABELS)
    assert len(result["smiles"]) == num_smiles
    assert len(result["top_k"]) == num_smiles

    probs = np.asarray(result["probs"])
    assert probs.shape == (num_smiles, len(ODOR_LABELS))
    assert np.isfinite(probs).all()
    assert np.all(probs >= 0.0)
    assert np.all(probs <= 1.0)

    for row_index, entries in enumerate(result["top_k"]):
        assert len(entries) == top_k
        entry_probs = [entry["prob"] for entry in entries]
        assert entry_probs == sorted(entry_probs, reverse=True)
        for entry in entries:
            index = entry["index"]
            assert 0 <= index < len(ODOR_LABELS)
            assert entry["label"] == ODOR_LABELS[index]
            assert entry["prob"] == pytest.approx(float(probs[row_index, index]))

    if expect_embedding:
        embedding = np.asarray(result["embedding"])
        assert embedding.shape == (num_smiles, 256)
        assert np.isfinite(embedding).all()
    else:
        assert result["embedding"] is None


def test_single_checkpoint_prediction_contract() -> None:
    _require_model_deps()
    _require_checkpoint()

    result = predict_smiles(
        smiles=["C", "CCO"],
        checkpoint_paths=[str(CHECKPOINT_PATH)],
        top_k=5,
        device="cpu",
        return_embedding=True,
    )

    _assert_prediction_contract(
        result=result,
        num_smiles=2,
        top_k=5,
        expect_embedding=True,
    )
    assert result["checkpoint_paths"] == [str(CHECKPOINT_PATH.resolve())]


def test_cli_smoke_outputs_json_without_embedding() -> None:
    _require_model_deps()
    _require_checkpoint()

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--smiles",
            "C",
            "--smiles",
            "CCO",
            "--checkpoint",
            str(CHECKPOINT_PATH),
            "--top-k",
            "3",
            "--no-embedding",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    result = json.loads(completed.stdout)
    _assert_prediction_contract(
        result=result,
        num_smiles=2,
        top_k=3,
        expect_embedding=False,
    )


def test_predict_smiles_rejects_empty_smiles() -> None:
    with pytest.raises(ValueError, match="smiles must contain"):
        predict_smiles(
            smiles=[],
            checkpoint_paths=[str(CHECKPOINT_PATH)],
        )


def test_predict_smiles_rejects_empty_checkpoint_list() -> None:
    with pytest.raises(ValueError, match="checkpoint_paths must contain"):
        predict_smiles(
            smiles=["C"],
            checkpoint_paths=[],
        )


def test_predict_smiles_rejects_invalid_top_k() -> None:
    with pytest.raises(ValueError, match="top_k must be positive"):
        predict_smiles(
            smiles=["C"],
            checkpoint_paths=[str(CHECKPOINT_PATH)],
            top_k=0,
        )

    with pytest.raises(ValueError, match="exceeds number of odor labels"):
        predict_smiles(
            smiles=["C"],
            checkpoint_paths=[str(CHECKPOINT_PATH)],
            top_k=len(ODOR_LABELS) + 1,
        )


def test_predict_smiles_rejects_invalid_smiles() -> None:
    _require_torch()
    _require_checkpoint()

    with pytest.raises(ValueError, match="Failed to featurize SMILES"):
        predict_smiles(
            smiles=["not-a-smiles"],
            checkpoint_paths=[str(CHECKPOINT_PATH)],
        )


def test_lfs_pointer_without_matching_real_weights_is_rejected(tmp_path) -> None:
    pointer = tmp_path / "checkpoint2.pt"
    pointer.write_text(
        "\n".join(
            [
                "version https://git-lfs.github.com/spec/v1",
                "oid sha256:" + ("0" * 64),
                "size 25106835",
                "",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Git LFS pointer"):
        resolve_checkpoint_path(pointer)


def test_openpom_lfs_pointer_resolves_to_local_real_checkpoint() -> None:
    if not OPENPOM_POINTER_PATH.exists():
        pytest.skip(f"OpenPOM pointer not found: {OPENPOM_POINTER_PATH}")
    _require_checkpoint()

    assert resolve_checkpoint_path(OPENPOM_POINTER_PATH) == CHECKPOINT_PATH.resolve()


def test_true_10_model_ensemble_from_environment() -> None:
    raw_checkpoints = os.environ.get(ENV_CHECKPOINTS)
    if not raw_checkpoints:
        pytest.skip(f"{ENV_CHECKPOINTS} is not set.")

    checkpoint_paths = [item for item in raw_checkpoints.split(os.pathsep) if item]
    assert len(checkpoint_paths) == 10

    resolved_paths = [resolve_checkpoint_path(path) for path in checkpoint_paths]
    assert len({str(path) for path in resolved_paths}) == 10

    _require_model_deps()
    result = predict_smiles(
        smiles=["C", "CCO"],
        checkpoint_paths=checkpoint_paths,
        top_k=10,
        device="cpu",
        return_embedding=True,
    )

    _assert_prediction_contract(
        result=result,
        num_smiles=2,
        top_k=10,
        expect_embedding=True,
    )
    assert result["checkpoint_paths"] == [str(path) for path in resolved_paths]
