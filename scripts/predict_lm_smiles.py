#!/usr/bin/env python
"""Predict odor labels with a frozen molecular LM and trained MLP head."""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

DEFAULT_CHECKPOINT = Path("outputs/molformer_head/head.pt")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Predict odor probabilities with a frozen chemical LM head."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help=f"Trained head checkpoint. Default: {DEFAULT_CHECKPOINT}",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=f"Optional head config JSON. Default: config embedded in {DEFAULT_CHECKPOINT}",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Override local pretrained LM path from the head checkpoint.",
    )
    parser.add_argument(
        "--smiles",
        action="append",
        default=None,
        help="SMILES string to predict. Repeat for multiple molecules.",
    )
    parser.add_argument(
        "--smiles-file",
        type=Path,
        default=None,
        help="Optional newline-delimited SMILES file. Overrides --smiles.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of top labels to return. Default: 10.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="LM inference batch size. Default: 64.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device. Default: cuda if available, otherwise cpu.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="Optional tokenizer max_length.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path. JSON is also written to stdout.",
    )
    return parser


def resolve_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (Path.cwd() / candidate).resolve()


def resolve_device(device: str | None, torch_module: Any) -> str:
    if device:
        return device
    return "cuda" if torch_module.cuda.is_available() else "cpu"


def load_smiles(args: argparse.Namespace, parser: argparse.ArgumentParser) -> list[str]:
    if args.smiles_file is not None:
        lines = args.smiles_file.read_text(encoding="utf-8").splitlines()
        smiles = [line.strip() for line in lines if line.strip()]
    elif args.smiles is not None:
        smiles = [item.strip() for item in args.smiles if item.strip()]
    else:
        parser.error("Provide at least one --smiles or a --smiles-file.")

    if not smiles:
        parser.error("SMILES input is empty.")
    return smiles


def get_output_value(outputs: Any, key: str) -> Any:
    if isinstance(outputs, dict):
        return outputs.get(key)
    return getattr(outputs, key, None)


def move_batch_to_device(batch: Any, device: str) -> Any:
    if hasattr(batch, "to"):
        return batch.to(device)
    return {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in batch.items()
    }


def masked_mean_pool(
    *,
    last_hidden_state: Any,
    attention_mask: Any,
    torch_module: Any,
) -> Any:
    if attention_mask is None:
        raise ValueError("attention_mask is required for masked mean pooling.")
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(
        min=torch_module.finfo(last_hidden_state.dtype).eps
    )
    return summed / counts


def pool_outputs(outputs: Any, batch: Any, torch_module: Any) -> Any:
    pooler_output = get_output_value(outputs, "pooler_output")
    if pooler_output is not None:
        return pooler_output

    last_hidden_state = get_output_value(outputs, "last_hidden_state")
    if last_hidden_state is None:
        raise ValueError(
            "Model output has neither pooler_output nor last_hidden_state."
        )
    return masked_mean_pool(
        last_hidden_state=last_hidden_state,
        attention_mask=batch.get("attention_mask"),
        torch_module=torch_module,
    )


def batched(items: list[str], batch_size: int) -> list[list[str]]:
    return [
        items[start : start + batch_size]
        for start in range(0, len(items), batch_size)
    ]


def export_runtime_embeddings(
    *,
    smiles: list[str],
    tokenizer: Any,
    model: Any,
    batch_size: int,
    device: str,
    max_length: int | None,
    torch_module: Any,
) -> np.ndarray:
    tokenizer_kwargs: dict[str, Any] = {
        "padding": True,
        "truncation": True,
        "return_tensors": "pt",
    }
    if max_length is not None:
        tokenizer_kwargs["max_length"] = max_length

    embeddings: list[np.ndarray] = []
    with torch_module.no_grad():
        for smiles_batch in batched(smiles, batch_size):
            tokenized = tokenizer(smiles_batch, **tokenizer_kwargs)
            tokenized = move_batch_to_device(tokenized, device)
            outputs = model(**tokenized)
            pooled = pool_outputs(outputs, tokenized, torch_module)
            embeddings.append(pooled.detach().cpu().numpy())
    return np.concatenate(embeddings, axis=0).astype(np.float32)


def load_head_checkpoint(path: Path, torch_module: Any) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch_module.load).parameters:
        kwargs["weights_only"] = False
    checkpoint = torch_module.load(path, **kwargs)
    required = {
        "state_dict",
        "input_dim",
        "hidden_dim",
        "output_dim",
        "dropout",
        "feature_mean",
        "feature_std",
        "label_names",
        "config",
    }
    missing = sorted(required - set(checkpoint))
    if missing:
        raise KeyError(f"Head checkpoint is missing required keys: {missing}")
    return checkpoint


def load_config_override(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"Config JSON must contain an object: {path}")
    return config


def json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [json_ready(item) for item in value]
    return value


def run(
    *,
    checkpoint_path: str | Path,
    model_path: str | Path | None,
    smiles: list[str],
    top_k: int,
    batch_size: int,
    device: str | None,
    max_length: int | None,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    import torch
    from transformers import AutoModel, AutoTokenizer

    from pom_repro.lm_head import FrozenLMHead, top_k_predictions

    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if top_k <= 0:
        raise ValueError("top_k must be positive.")

    checkpoint_path = resolve_path(checkpoint_path)
    checkpoint = load_head_checkpoint(checkpoint_path, torch)
    config = dict(checkpoint["config"])
    if config_path is not None:
        resolved_config_path = resolve_path(config_path)
        config.update(load_config_override(resolved_config_path))
    else:
        resolved_config_path = None
    resolved_model_path = (
        resolve_path(model_path)
        if model_path is not None
        else resolve_path(config["model_path"])
    )
    resolved_device = resolve_device(device, torch)

    tokenizer = AutoTokenizer.from_pretrained(
        str(resolved_model_path),
        trust_remote_code=True,
    )
    encoder = AutoModel.from_pretrained(
        str(resolved_model_path),
        trust_remote_code=True,
        deterministic_eval=True,
    )
    encoder.to(resolved_device)
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)

    head = FrozenLMHead(
        input_dim=int(checkpoint["input_dim"]),
        output_dim=int(checkpoint["output_dim"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
        dropout=float(checkpoint["dropout"]),
    )
    head.load_state_dict(checkpoint["state_dict"])
    head.to(resolved_device)
    head.eval()

    raw_embeddings = export_runtime_embeddings(
        smiles=smiles,
        tokenizer=tokenizer,
        model=encoder,
        batch_size=batch_size,
        device=resolved_device,
        max_length=max_length,
        torch_module=torch,
    )
    feature_mean = np.asarray(checkpoint["feature_mean"], dtype=np.float32)
    feature_std = np.asarray(checkpoint["feature_std"], dtype=np.float32)
    embeddings = ((raw_embeddings - feature_mean) / feature_std).astype(np.float32)

    with torch.no_grad():
        logits = head(torch.from_numpy(embeddings).to(resolved_device))
        probs = torch.sigmoid(logits).detach().cpu().numpy()

    labels = [str(item) for item in checkpoint["label_names"]]
    return {
        "smiles": smiles,
        "labels": labels,
        "probs": probs,
        "top_k": top_k_predictions(probs, labels, top_k),
        "checkpoint": str(checkpoint_path),
        "config": str(resolved_config_path) if resolved_config_path is not None else None,
        "model_path": str(resolved_model_path),
        "device": resolved_device,
    }


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    smiles = load_smiles(args, parser)
    result = run(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        model_path=args.model_path,
        smiles=smiles,
        top_k=args.top_k,
        batch_size=args.batch_size,
        device=args.device,
        max_length=args.max_length,
    )
    json_text = json.dumps(
        json_ready(result),
        allow_nan=False,
        indent=2,
        sort_keys=True,
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json_text + "\n", encoding="utf-8")
    print(json_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
