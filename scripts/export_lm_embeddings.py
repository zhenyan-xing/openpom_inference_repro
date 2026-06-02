#!/usr/bin/env python
"""Export frozen MoLFormer embeddings for OpenPOM SMILES rows."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_DATA = Path(
    "/home/xing/openpom/openpom/data/curated_datasets/"
    "curated_GS_LF_merged_4983.csv"
)
DEFAULT_MODEL_PATH = Path("models/MoLFormer-XL-both-10pct")
DEFAULT_MODEL_ID = "ibm-research/MoLFormer-XL-both-10pct"
DEFAULT_OUTPUT = Path("artifacts/lm_embeddings/openpom_4983_molformer.npz")
DEFAULT_MANIFEST = Path(
    "artifacts/lm_embeddings/openpom_4983_molformer_manifest.json"
)
DEFAULT_SMILES_COLUMN = "nonStereoSMILES"
DESCRIPTOR_COLUMN = "descriptors"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export frozen MoLFormer embeddings for an OpenPOM CSV."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA,
        help=f"Input OpenPOM CSV. Default: {DEFAULT_DATA}",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=f"Local tokenizer/model directory. Default: {DEFAULT_MODEL_PATH}",
    )
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help=f"Model id recorded in manifest only. Default: {DEFAULT_MODEL_ID}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output .npz path. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help=f"Output manifest JSON path. Default: {DEFAULT_MANIFEST}",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Inference batch size. Default: 64.",
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
        "--smiles-column",
        default=DEFAULT_SMILES_COLUMN,
        help=f"CSV SMILES column. Default: {DEFAULT_SMILES_COLUMN}",
    )
    return parser


def resolve_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (Path.cwd() / candidate).resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def detect_git_commit(repo_root: Path | None = None) -> str:
    repo_root = repo_root or Path(__file__).resolve().parents[1]
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return "unknown"

    commit = result.stdout.strip()
    if result.returncode != 0 or not commit:
        return "unknown"
    return commit


def format_command(argv: list[str] | None = None) -> str:
    return shlex.join(argv if argv is not None else sys.argv)


def read_openpom_csv(
    data_path: Path,
    smiles_column: str,
) -> tuple[list[str], np.ndarray, list[str]]:
    with data_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        if fieldnames is None:
            raise ValueError(f"CSV has no header: {data_path}")

        missing = [
            column
            for column in (smiles_column, DESCRIPTOR_COLUMN)
            if column not in fieldnames
        ]
        if missing:
            raise ValueError(
                f"CSV is missing required column(s): {', '.join(missing)}"
            )

        label_names = [
            column
            for column in fieldnames
            if column not in {smiles_column, DESCRIPTOR_COLUMN}
        ]
        smiles: list[str] = []
        labels: list[list[float]] = []

        for row_index, row in enumerate(reader, start=2):
            smile = (row.get(smiles_column) or "").strip()
            if not smile:
                raise ValueError(f"Missing SMILES value in CSV row {row_index}.")

            try:
                label_row = [float(row[label_name]) for label_name in label_names]
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid label value in CSV row {row_index}."
                ) from exc

            smiles.append(smile)
            labels.append(label_row)

    if not smiles:
        raise ValueError(f"CSV contains no molecule rows: {data_path}")

    return smiles, np.asarray(labels, dtype=np.float32), label_names


def resolve_device(device: str | None, torch_module: Any) -> str:
    if device:
        return device
    return "cuda" if torch_module.cuda.is_available() else "cpu"


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


def pool_outputs(outputs: Any, batch: Any, torch_module: Any) -> tuple[Any, str]:
    pooler_output = get_output_value(outputs, "pooler_output")
    if pooler_output is not None:
        return pooler_output, "pooler_output"

    last_hidden_state = get_output_value(outputs, "last_hidden_state")
    if last_hidden_state is None:
        raise ValueError(
            "Model output has neither pooler_output nor last_hidden_state."
        )

    return (
        masked_mean_pool(
            last_hidden_state=last_hidden_state,
            attention_mask=batch.get("attention_mask"),
            torch_module=torch_module,
        ),
        "masked_mean",
    )


def batched(items: list[str], batch_size: int) -> list[list[str]]:
    return [
        items[start : start + batch_size]
        for start in range(0, len(items), batch_size)
    ]


def export_embeddings(
    *,
    smiles: list[str],
    tokenizer: Any,
    model: Any,
    batch_size: int,
    device: str,
    max_length: int | None,
    torch_module: Any,
) -> tuple[np.ndarray, str]:
    tokenizer_kwargs: dict[str, Any] = {
        "padding": True,
        "truncation": True,
        "return_tensors": "pt",
    }
    if max_length is not None:
        tokenizer_kwargs["max_length"] = max_length

    embeddings: list[np.ndarray] = []
    pooling: str | None = None

    with torch_module.no_grad():
        for smiles_batch in batched(smiles, batch_size):
            tokenized = tokenizer(smiles_batch, **tokenizer_kwargs)
            tokenized = move_batch_to_device(tokenized, device)
            outputs = model(**tokenized)
            pooled, batch_pooling = pool_outputs(outputs, tokenized, torch_module)
            if pooling is None:
                pooling = batch_pooling
            elif pooling != batch_pooling:
                raise ValueError("Model pooling output changed between batches.")

            embeddings.append(pooled.detach().cpu().numpy())

    if not embeddings:
        raise ValueError("No embeddings were produced.")

    return np.concatenate(embeddings, axis=0), pooling or "unknown"


def write_npz(
    *,
    output_path: Path,
    smiles: list[str],
    embeddings: np.ndarray,
    labels: np.ndarray,
    label_names: list[str],
    model_id: str,
    model_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        smiles=np.asarray(smiles, dtype=str),
        embeddings=embeddings,
        labels=labels,
        label_names=np.asarray(label_names, dtype=str),
        model_id=np.asarray(model_id),
        model_path=np.asarray(str(model_path)),
    )


def write_manifest(
    *,
    manifest_path: Path,
    data_path: Path,
    data_sha256: str,
    model_id: str,
    model_path: Path,
    output_path: Path,
    num_samples: int,
    num_labels: int,
    embedding_shape: tuple[int, ...],
    dtype: str,
    batch_size: int,
    device: str,
    pooling: str,
    git_commit: str,
    command: str,
) -> None:
    created_at = datetime.now(timezone.utc).isoformat()
    manifest = {
        "data_csv": str(data_path),
        "data_path": str(data_path),
        "data_sha256": data_sha256,
        "model_id": model_id,
        "model_path": str(model_path),
        "output_path": str(output_path),
        "num_samples": num_samples,
        "num_labels": num_labels,
        "embedding_shape": list(embedding_shape),
        "dtype": dtype,
        "batch_size": batch_size,
        "device": device,
        "git_commit": git_commit,
        "created_at": created_at,
        "created_at_utc": created_at,
        "command": command,
        "pooling": pooling,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def run(
    *,
    data: str | Path,
    model_path: str | Path,
    model_id: str,
    output: str | Path,
    manifest: str | Path,
    batch_size: int,
    device: str | None,
    max_length: int | None,
    smiles_column: str,
    command: str | None = None,
) -> int:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if max_length is not None and max_length <= 0:
        raise ValueError("max_length must be positive when provided.")

    data_path = resolve_path(data)
    model_path = resolve_path(model_path)
    output_path = resolve_path(output)
    manifest_path = resolve_path(manifest)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Local model path does not exist: {model_path}. "
            "--model-id is manifest metadata only and is not used as a download fallback."
        )

    import torch
    from transformers import AutoModel, AutoTokenizer

    resolved_device = resolve_device(device, torch)
    smiles, labels, label_names = read_openpom_csv(data_path, smiles_column)

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        trust_remote_code=True,
    )
    model = AutoModel.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        deterministic_eval=True,
    )
    model.to(resolved_device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    embeddings, pooling = export_embeddings(
        smiles=smiles,
        tokenizer=tokenizer,
        model=model,
        batch_size=batch_size,
        device=resolved_device,
        max_length=max_length,
        torch_module=torch,
    )

    write_npz(
        output_path=output_path,
        smiles=smiles,
        embeddings=embeddings,
        labels=labels,
        label_names=label_names,
        model_id=model_id,
        model_path=model_path,
    )
    write_manifest(
        manifest_path=manifest_path,
        data_path=data_path,
        data_sha256=sha256_file(data_path),
        model_id=model_id,
        model_path=model_path,
        output_path=output_path,
        num_samples=len(smiles),
        num_labels=len(label_names),
        embedding_shape=tuple(embeddings.shape),
        dtype=str(embeddings.dtype),
        batch_size=batch_size,
        device=resolved_device,
        pooling=pooling,
        git_commit=detect_git_commit(),
        command=command or format_command(),
    )

    print(f"Wrote embeddings to {output_path}")
    print(f"Wrote manifest to {manifest_path}")
    return 0


def main() -> int:
    args = build_parser().parse_args()
    return run(
        data=args.data,
        model_path=args.model_path,
        model_id=args.model_id,
        output=args.output,
        manifest=args.manifest,
        batch_size=args.batch_size,
        device=args.device,
        max_length=args.max_length,
        smiles_column=args.smiles_column,
        command=format_command(),
    )


if __name__ == "__main__":
    raise SystemExit(main())
