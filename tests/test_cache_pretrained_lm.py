from __future__ import annotations

import importlib.util
import json
import sys
import types
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts/cache_pretrained_lm.py"


def load_cache_module():
    spec = importlib.util.spec_from_file_location(
        "cache_pretrained_lm_for_tests",
        SCRIPT_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_import_does_not_require_transformers_or_torch(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "transformers", None)
    monkeypatch.setitem(sys.modules, "torch", None)

    module = load_cache_module()

    assert module.DEFAULT_MODEL_ID == "ibm-research/MoLFormer-XL-both-10pct"


def test_parser_defaults_and_boolean_flags() -> None:
    module = load_cache_module()

    args = module.build_parser().parse_args([])

    assert args.model_id == "ibm-research/MoLFormer-XL-both-10pct"
    assert args.output_dir == Path("models/MoLFormer-XL-both-10pct")
    assert args.trust_remote_code is True
    assert args.deterministic_eval is True

    args = module.build_parser().parse_args(
        ["--no-trust-remote-code", "--no-deterministic-eval"]
    )
    assert args.trust_remote_code is False
    assert args.deterministic_eval is False


def test_run_caches_model_and_writes_manifest(tmp_path, monkeypatch) -> None:
    module = load_cache_module()
    calls: dict[str, object] = {}

    class FakeTokenizer:
        def save_pretrained(self, output_dir: Path) -> None:
            calls["tokenizer_saved_to"] = output_dir

    class FakeModel:
        def save_pretrained(self, output_dir: Path) -> None:
            calls["model_saved_to"] = output_dir

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(model_id: str, *, trust_remote_code: bool) -> FakeTokenizer:
            calls["tokenizer_from_pretrained"] = {
                "model_id": model_id,
                "trust_remote_code": trust_remote_code,
            }
            return FakeTokenizer()

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(
            model_id: str,
            *,
            trust_remote_code: bool,
            deterministic_eval: bool,
        ) -> FakeModel:
            calls["model_from_pretrained"] = {
                "model_id": model_id,
                "trust_remote_code": trust_remote_code,
                "deterministic_eval": deterministic_eval,
            }
            return FakeModel()

    fake_transformers = types.SimpleNamespace(
        AutoModel=FakeAutoModel,
        AutoTokenizer=FakeAutoTokenizer,
        __version__="9.8.7",
    )
    fake_torch = types.SimpleNamespace(__version__="2.1.0")
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    output_dir = tmp_path / "molformer-cache"
    exit_code = module.run(
        model_id="example/model",
        output_dir=output_dir,
        trust_remote_code=False,
        deterministic_eval=False,
    )

    assert exit_code == 0
    assert calls["tokenizer_from_pretrained"] == {
        "model_id": "example/model",
        "trust_remote_code": False,
    }
    assert calls["model_from_pretrained"] == {
        "model_id": "example/model",
        "trust_remote_code": False,
        "deterministic_eval": False,
    }
    assert calls["tokenizer_saved_to"] == output_dir.resolve()
    assert calls["model_saved_to"] == output_dir.resolve()

    manifest = json.loads((output_dir / "cache_manifest.json").read_text())
    assert manifest["model_id"] == "example/model"
    assert manifest["output_dir"] == str(output_dir.resolve())
    assert manifest["transformers_version"] == "9.8.7"
    assert manifest["torch_version"] == "2.1.0"
    created_at = datetime.fromisoformat(manifest["created_at_utc"])
    assert created_at.tzinfo is not None
