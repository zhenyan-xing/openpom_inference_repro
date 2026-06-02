from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import types
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts/export_lm_embeddings.py"


def load_export_module():
    spec = importlib.util.spec_from_file_location(
        "export_lm_embeddings_for_tests",
        SCRIPT_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "nonStereoSMILES,descriptors,floral,woody",
                "CCO,desc-a,1,0",
                "C,desc-b,0,1",
                "N,desc-c,1,1",
                "",
            ]
        ),
        encoding="utf-8",
    )


def test_import_does_not_require_transformers_or_torch(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "transformers", None)
    monkeypatch.setitem(sys.modules, "torch", None)

    module = load_export_module()

    assert module.DEFAULT_MODEL_ID == "ibm-research/MoLFormer-XL-both-10pct"


def test_parser_defaults() -> None:
    module = load_export_module()

    args = module.build_parser().parse_args([])

    assert args.data == Path(
        "/home/xing/openpom/openpom/data/curated_datasets/"
        "curated_GS_LF_merged_4983.csv"
    )
    assert args.model_path == Path("models/MoLFormer-XL-both-10pct")
    assert args.model_id == "ibm-research/MoLFormer-XL-both-10pct"
    assert args.output == Path("artifacts/lm_embeddings/openpom_4983_molformer.npz")
    assert args.manifest == Path(
        "artifacts/lm_embeddings/openpom_4983_molformer_manifest.json"
    )
    assert args.batch_size == 64
    assert args.device is None
    assert args.max_length is None
    assert args.smiles_column == "nonStereoSMILES"


def test_run_exports_pooler_embeddings_npz_and_manifest(tmp_path, monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    module = load_export_module()

    csv_path = tmp_path / "openpom.csv"
    model_path = tmp_path / "molformer"
    output_path = tmp_path / "embeddings.npz"
    manifest_path = tmp_path / "manifest.json"
    write_csv(csv_path)
    model_path.mkdir()

    calls: dict[str, object] = {"tokenize": []}
    parameters = [
        torch.nn.Parameter(torch.tensor(1.0)),
        torch.nn.Parameter(torch.tensor(2.0)),
    ]

    class FakeTokenizer:
        def __call__(self, smiles: list[str], **kwargs: object) -> dict[str, object]:
            calls["tokenize"].append({"smiles": list(smiles), "kwargs": kwargs})
            return {
                "input_ids": torch.arange(len(smiles) * 4).reshape(len(smiles), 4),
                "attention_mask": torch.ones((len(smiles), 4), dtype=torch.long),
            }

    class FakeModel:
        def __init__(self) -> None:
            self.call_index = 0

        def to(self, device: str) -> "FakeModel":
            calls["model_to"] = device
            return self

        def eval(self) -> None:
            calls["model_eval"] = True

        def parameters(self):
            return parameters

        def __call__(self, **kwargs: object) -> object:
            calls["model_kwargs"] = kwargs
            batch_size = kwargs["input_ids"].shape[0]
            base = self.call_index * 100
            self.call_index += 1
            pooler = torch.arange(
                base,
                base + batch_size * 3,
                dtype=torch.float32,
            ).reshape(batch_size, 3)
            return types.SimpleNamespace(pooler_output=pooler)

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(path: str, *, trust_remote_code: bool) -> FakeTokenizer:
            calls["tokenizer_from_pretrained"] = {
                "path": path,
                "trust_remote_code": trust_remote_code,
            }
            return FakeTokenizer()

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(
            path: str,
            *,
            trust_remote_code: bool,
            deterministic_eval: bool,
        ) -> FakeModel:
            calls["model_from_pretrained"] = {
                "path": path,
                "trust_remote_code": trust_remote_code,
                "deterministic_eval": deterministic_eval,
            }
            return FakeModel()

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(AutoModel=FakeAutoModel, AutoTokenizer=FakeAutoTokenizer),
    )
    monkeypatch.setattr(module, "detect_git_commit", lambda: "abc123")

    exit_code = module.run(
        data=csv_path,
        model_path=model_path,
        model_id="example/molformer",
        output=output_path,
        manifest=manifest_path,
        batch_size=2,
        device="cpu",
        max_length=8,
        smiles_column="nonStereoSMILES",
        command="python scripts/export_lm_embeddings.py --example",
    )

    assert exit_code == 0
    assert calls["tokenizer_from_pretrained"] == {
        "path": str(model_path.resolve()),
        "trust_remote_code": True,
    }
    assert calls["model_from_pretrained"] == {
        "path": str(model_path.resolve()),
        "trust_remote_code": True,
        "deterministic_eval": True,
    }
    assert calls["model_to"] == "cpu"
    assert calls["model_eval"] is True
    assert all(parameter.requires_grad is False for parameter in parameters)
    assert calls["tokenize"] == [
        {
            "smiles": ["CCO", "C"],
            "kwargs": {
                "padding": True,
                "truncation": True,
                "return_tensors": "pt",
                "max_length": 8,
            },
        },
        {
            "smiles": ["N"],
            "kwargs": {
                "padding": True,
                "truncation": True,
                "return_tensors": "pt",
                "max_length": 8,
            },
        },
    ]

    npz = np.load(output_path)
    assert set(npz.files) == {
        "smiles",
        "embeddings",
        "labels",
        "label_names",
        "model_id",
        "model_path",
    }
    assert npz["smiles"].tolist() == ["CCO", "C", "N"]
    np.testing.assert_array_equal(
        npz["embeddings"],
        np.asarray([[0, 1, 2], [3, 4, 5], [100, 101, 102]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        npz["labels"],
        np.asarray([[1, 0], [0, 1], [1, 1]], dtype=np.float32),
    )
    assert npz["label_names"].tolist() == ["floral", "woody"]
    assert npz["model_id"].item() == "example/molformer"
    assert npz["model_path"].item() == str(model_path.resolve())

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["data_csv"] == str(csv_path.resolve())
    assert manifest["data_sha256"] == sha256_file(csv_path)
    assert manifest["model_id"] == "example/molformer"
    assert manifest["model_path"] == str(model_path.resolve())
    assert manifest["output_path"] == str(output_path.resolve())
    assert manifest["num_samples"] == 3
    assert manifest["num_labels"] == 2
    assert manifest["embedding_shape"] == [3, 3]
    assert manifest["dtype"] == "float32"
    assert manifest["batch_size"] == 2
    assert manifest["device"] == "cpu"
    assert manifest["pooling"] == "pooler_output"
    assert manifest["git_commit"] == "abc123"
    assert manifest["command"] == "python scripts/export_lm_embeddings.py --example"
    created_at = datetime.fromisoformat(manifest["created_at"])
    assert created_at.tzinfo is not None


def test_run_uses_masked_mean_when_pooler_output_is_absent(
    tmp_path,
    monkeypatch,
) -> None:
    torch = pytest.importorskip("torch")
    module = load_export_module()

    csv_path = tmp_path / "openpom.csv"
    model_path = tmp_path / "molformer"
    output_path = tmp_path / "embeddings.npz"
    manifest_path = tmp_path / "manifest.json"
    write_csv(csv_path)
    model_path.mkdir()

    class FakeTokenizer:
        def __call__(self, smiles: list[str], **kwargs: object) -> dict[str, object]:
            del smiles, kwargs
            return {
                "input_ids": torch.zeros((3, 3), dtype=torch.long),
                "attention_mask": torch.tensor(
                    [[1, 1, 0], [1, 0, 0], [1, 1, 1]],
                    dtype=torch.long,
                ),
            }

    class FakeModel:
        def to(self, device: str) -> "FakeModel":
            del device
            return self

        def eval(self) -> None:
            return None

        def parameters(self):
            return []

        def __call__(self, **kwargs: object) -> object:
            del kwargs
            hidden = torch.tensor(
                [
                    [[1.0, 3.0], [5.0, 7.0], [99.0, 99.0]],
                    [[2.0, 4.0], [99.0, 99.0], [99.0, 99.0]],
                    [[1.0, 1.0], [3.0, 5.0], [5.0, 9.0]],
                ],
                dtype=torch.float32,
            )
            return {"last_hidden_state": hidden}

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(path: str, *, trust_remote_code: bool) -> FakeTokenizer:
            del path, trust_remote_code
            return FakeTokenizer()

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(
            path: str,
            *,
            trust_remote_code: bool,
            deterministic_eval: bool,
        ) -> FakeModel:
            del path, trust_remote_code, deterministic_eval
            return FakeModel()

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(AutoModel=FakeAutoModel, AutoTokenizer=FakeAutoTokenizer),
    )
    monkeypatch.setattr(module, "detect_git_commit", lambda: "abc123")

    exit_code = module.run(
        data=csv_path,
        model_path=model_path,
        model_id="example/molformer",
        output=output_path,
        manifest=manifest_path,
        batch_size=10,
        device="cpu",
        max_length=None,
        smiles_column="nonStereoSMILES",
        command="python scripts/export_lm_embeddings.py --example",
    )

    assert exit_code == 0
    npz = np.load(output_path)
    np.testing.assert_array_equal(
        npz["embeddings"],
        np.asarray([[3, 5], [2, 4], [3, 5]], dtype=np.float32),
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["pooling"] == "masked_mean"
