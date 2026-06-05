from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts/train_frozen_lm_head.py"
PREDICT_SCRIPT_PATH = REPO_ROOT / "scripts/predict_lm_smiles.py"


def load_train_module():
    spec = importlib.util.spec_from_file_location(
        "train_frozen_lm_head_for_tests",
        SCRIPT_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_predict_module():
    spec = importlib.util.spec_from_file_location(
        "predict_lm_smiles_for_tests",
        PREDICT_SCRIPT_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_embedding_npz(path: Path) -> None:
    embeddings = np.asarray(
        [
            [0.0, 0.1, 0.2, 0.3],
            [1.0, 1.1, 1.2, 1.3],
            [2.0, 2.1, 2.2, 2.3],
            [3.0, 3.1, 3.2, 3.3],
            [4.0, 4.1, 4.2, 4.3],
            [5.0, 5.1, 5.2, 5.3],
        ],
        dtype=np.float32,
    )
    labels = np.asarray(
        [
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 1, 1],
            [0, 0, 1],
            [1, 0, 1],
        ],
        dtype=np.float32,
    )
    np.savez(
        path,
        smiles=np.asarray(["C", "CC", "CCC", "N", "NN", "O"], dtype=str),
        embeddings=embeddings,
        labels=labels,
        label_names=np.asarray(["floral", "woody", "green"], dtype=str),
        model_id=np.asarray("example/molformer"),
        model_path=np.asarray("/models/example"),
    )


def test_train_parser_accepts_server_command_metadata() -> None:
    module = load_train_module()

    args = module.build_parser().parse_args(
        [
            "--data",
            "data/openpom/curated_GS_LF_merged_4983.csv",
            "--model-id",
            "ibm-research/MoLFormer-XL-both-10pct",
            "--output-dir",
            "outputs/molformer_head",
            "--final-epochs",
            "2",
        ]
    )

    assert args.data == Path("data/openpom/curated_GS_LF_merged_4983.csv")
    assert args.model_id == "ibm-research/MoLFormer-XL-both-10pct"
    assert args.output_dir == Path("outputs/molformer_head")
    assert args.final_epochs == 2


def test_predict_parser_accepts_explicit_config_contract() -> None:
    module = load_predict_module()

    args = module.build_parser().parse_args(
        [
            "--checkpoint",
            "outputs/molformer_head/head.pt",
            "--config",
            "outputs/molformer_head/config.json",
            "--smiles",
            "CCO",
        ]
    )

    assert args.checkpoint == Path("outputs/molformer_head/head.pt")
    assert args.config == Path("outputs/molformer_head/config.json")
    assert args.smiles == ["CCO"]


def test_predict_checkpoint_loader_disables_weights_only_when_supported(tmp_path) -> None:
    module = load_predict_module()
    calls: dict[str, object] = {}

    class FakeTorch:
        @staticmethod
        def load(
            path: Path,
            *,
            map_location: str,
            weights_only: bool,
        ) -> dict[str, object]:
            calls["path"] = path
            calls["map_location"] = map_location
            calls["weights_only"] = weights_only
            return {
                "state_dict": {},
                "input_dim": 4,
                "hidden_dim": 8,
                "output_dim": 3,
                "dropout": 0.1,
                "feature_mean": [],
                "feature_std": [],
                "label_names": [],
                "config": {},
            }

    checkpoint = module.load_head_checkpoint(tmp_path / "head.pt", FakeTorch)

    assert checkpoint["input_dim"] == 4
    assert calls["map_location"] == "cpu"
    assert calls["weights_only"] is False


def test_greedy_multilabel_kfold_covers_each_sample_once() -> None:
    module = load_train_module()
    labels = np.asarray(
        [
            [1, 0],
            [0, 1],
            [1, 1],
            [0, 0],
            [1, 0],
            [0, 1],
        ],
        dtype=np.float32,
    )

    folds = module.greedy_multilabel_kfold(labels, n_splits=3, seed=7)

    all_indices = sorted(index for fold in folds for index in fold.tolist())
    assert all_indices == list(range(len(labels)))
    assert all(len(fold) > 0 for fold in folds)


def test_run_trains_final_head_and_writes_bundle(tmp_path, monkeypatch) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("sklearn")
    module = load_train_module()

    embeddings_path = tmp_path / "embeddings.npz"
    output_dir = tmp_path / "head"
    write_embedding_npz(embeddings_path)
    monkeypatch.setattr(module, "detect_git_commit", lambda: "abc123")

    exit_code = module.run(
        embeddings_path=embeddings_path,
        output_dir=output_dir,
        epochs=2,
        batch_size=2,
        lr=1e-3,
        weight_decay=0.0,
        dropout=0.1,
        hidden_dim=8,
        folds=2,
        seed=1,
        device="cpu",
        pos_weight_max=10.0,
        patience=0,
        skip_cv=True,
        command="python scripts/train_frozen_lm_head.py --test",
        final_epochs=1,
    )

    assert exit_code == 0
    assert (output_dir / "head.pt").exists()
    assert (output_dir / "training_history.csv").exists()
    assert json.loads((output_dir / "label_names.json").read_text()) == [
        "floral",
        "woody",
        "green",
    ]
    config = json.loads((output_dir / "config.json").read_text())
    assert config["git_commit"] == "abc123"
    assert config["input_dim"] == 4
    assert config["output_dim"] == 3
    assert config["hidden_dim"] == 8
    assert config["final_epochs"] == 1
    assert config["skip_cv"] is True

    metrics = json.loads((output_dir / "metrics.json").read_text())
    assert "final_train" in metrics
    assert "macro_average_precision" in metrics["final_train"]
    assert len(metrics["final_train"]["history"]) == 1


def test_run_writes_cv_epoch_history(tmp_path, monkeypatch) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("sklearn")
    module = load_train_module()

    embeddings_path = tmp_path / "embeddings.npz"
    output_dir = tmp_path / "head"
    write_embedding_npz(embeddings_path)
    monkeypatch.setattr(module, "detect_git_commit", lambda: "abc123")

    exit_code = module.run(
        embeddings_path=embeddings_path,
        output_dir=output_dir,
        epochs=2,
        batch_size=2,
        lr=1e-3,
        weight_decay=0.0,
        dropout=0.1,
        hidden_dim=8,
        folds=2,
        seed=1,
        device="cpu",
        pos_weight_max=10.0,
        patience=0,
        skip_cv=False,
        command="python scripts/train_frozen_lm_head.py --test",
    )

    assert exit_code == 0
    metrics = json.loads((output_dir / "metrics.json").read_text())
    fold = metrics["cv"]["folds"][0]
    assert len(fold["history"]) == 2
    assert fold["history"][0]["epoch"] == 1
    assert fold["history"][1]["epoch"] == 2
    assert isinstance(fold["history"][0]["train_loss"], float)
    assert isinstance(fold["history"][0]["val_loss"], float)
    history_csv = (output_dir / "training_history.csv").read_text(encoding="utf-8")
    assert "phase,fold,epoch,train_loss,val_loss,best_epoch" in history_csv
    assert "cv,1,1," in history_csv
