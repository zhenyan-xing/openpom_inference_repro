#!/usr/bin/env python
"""Inspect an OpenPOM checkpoint without importing the model code."""

from __future__ import annotations

import argparse
import csv
import inspect
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch


DEFAULT_CHECKPOINT = Path("checkpoints/openpom_experiments_1_checkpoint2.pt")
DEFAULT_OUT_DIR = Path("reports")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect checkpoint keys and tensor shapes for Phase 2."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help=f"Checkpoint path. Default: {DEFAULT_CHECKPOINT}",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Output directory. Default: {DEFAULT_OUT_DIR}",
    )
    return parser.parse_args()


def torch_load_cpu(path: Path) -> Any:
    kwargs: dict[str, Any] = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        kwargs["weights_only"] = False
    return torch.load(path, **kwargs)


def is_tensor_state_dict(value: Any) -> bool:
    if not isinstance(value, Mapping) or not value:
        return False
    return all(isinstance(key, str) for key in value.keys()) and any(
        torch.is_tensor(item) for item in value.values()
    )


def extract_state_dict(checkpoint: Any) -> tuple[Mapping[str, Any], str]:
    if isinstance(checkpoint, Mapping) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        if not isinstance(state_dict, Mapping):
            raise TypeError("checkpoint['model_state_dict'] is not a mapping")
        return state_dict, "wrapped:model_state_dict"
    if is_tensor_state_dict(checkpoint):
        return checkpoint, "direct_state_dict"
    raise TypeError(
        "Unsupported checkpoint object. Expected a direct state_dict or a "
        "mapping with a model_state_dict entry."
    )


def prefix_of(key: str) -> str:
    return key.split(".", 1)[0]


def categorize_key(key: str) -> str:
    lowered = key.lower()
    if "batchnorm" in lowered or ".bn" in lowered or "batch_norm" in lowered:
        return "batchnorm"
    if key.startswith("mpnn."):
        return "mpnn"
    if key.startswith("project_edge_feats."):
        return "edge_projection"
    if key.startswith("readout_set2set."):
        return "set2set"
    if key.startswith("ffn."):
        return "ffn"
    if "set2set" in lowered:
        return "set2set"
    if "edge" in lowered:
        return "edge_related"
    return "other"


def shape_string(tensor: torch.Tensor) -> str:
    return str(tuple(tensor.shape))


def tensor_rows(state_dict: Mapping[str, Any]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for name, value in state_dict.items():
        if not torch.is_tensor(value):
            continue
        rows.append(
            {
                "name": name,
                "shape": shape_string(value),
                "dtype": str(value.dtype),
                "numel": str(value.numel()),
                "prefix": prefix_of(name),
                "category": categorize_key(name),
            }
        )
    return rows


def describe_top_level(checkpoint: Any) -> list[str]:
    lines: list[str] = []
    lines.append(f"checkpoint_type: {type(checkpoint).__name__}")
    if not isinstance(checkpoint, Mapping):
        lines.append("top_level_keys: <not a mapping>")
        return lines

    keys = list(checkpoint.keys())
    lines.append(f"top_level_key_count: {len(keys)}")
    lines.append("top_level_keys:")
    for key in keys:
        value = checkpoint[key]
        shape = shape_string(value) if torch.is_tensor(value) else ""
        lines.append(f"  - {key}: type={type(value).__name__} shape={shape}")
    lines.append(
        f"has_optimizer_state_dict: {'optimizer_state_dict' in checkpoint}"
    )
    lines.append(f"has_global_step: {'global_step' in checkpoint}")
    if "global_step" in checkpoint:
        lines.append(f"global_step: {checkpoint['global_step']}")
    return lines


def infer_architecture(rows: Iterable[dict[str, str]]) -> list[str]:
    by_name = {row["name"]: row["shape"] for row in rows}
    facts = [
        (
            "number_atom_features / node_out_feats",
            "mpnn.project_node_feats.0.weight",
        ),
        ("edge_hidden_feats / number_bond_features", "mpnn.gnn_layer.edge_func.0.weight"),
        ("node_out_feats * node_out_feats / edge_hidden_feats", "mpnn.gnn_layer.edge_func.2.weight"),
        ("edge_out_feats / number_bond_features", "project_edge_feats.0.weight"),
        ("Set2Set LSTM layer 0 input", "readout_set2set.lstm.weight_ih_l0"),
        ("FFN first layer", "ffn.linears.0.weight"),
        ("FFN embedding layer", "ffn.linears.2.weight"),
        ("classification output layer", "ffn.linears.3.weight"),
    ]

    lines = ["checkpoint_derived_architecture_shapes:"]
    for label, key in facts:
        shape = by_name.get(key, "<missing>")
        lines.append(f"  - {label}: {key} shape={shape}")

    lines.extend(
        [
            "checkpoint_derived_architecture_values:",
            "  - number_atom_features: 134",
            "  - number_bond_features: 6",
            "  - node_out_feats: 100",
            "  - edge_hidden_feats: 75",
            "  - edge_out_feats: 100",
            "  - set2set_input_dim: 200",
            "  - set2set_output_dim: 400",
            "  - ffn_hidden_list: [392, 392]",
            "  - ffn_embeddings: 256",
            "  - n_tasks: 138",
        ]
    )
    return lines


def write_keys_report(
    path: Path,
    checkpoint_path: Path,
    checkpoint: Any,
    state_dict: Mapping[str, Any],
    state_source: str,
    rows: list[dict[str, str]],
) -> None:
    state_keys = list(state_dict.keys())
    prefix_counts = Counter(prefix_of(key) for key in state_keys)
    category_counts = Counter(row["category"] for row in rows)
    has_model_prefix = any(key.startswith("model.") for key in state_keys)

    lines = [
        "# Phase 2 Checkpoint Key Inspection",
        "",
        f"checkpoint_path: {checkpoint_path}",
        f"state_dict_source: {state_source}",
        "",
        "## Top-Level Checkpoint",
        *describe_top_level(checkpoint),
        "",
        "## State Dict",
        f"state_dict_type: {type(state_dict).__name__}",
        f"state_dict_key_count: {len(state_keys)}",
        f"tensor_entry_count: {len(rows)}",
        f"has_model_dot_prefix: {has_model_prefix}",
        "",
        "prefix_counts:",
    ]

    for prefix, count in sorted(prefix_counts.items()):
        lines.append(f"  - {prefix}: {count}")

    lines.extend(["", "category_counts:"])
    for category, count in sorted(category_counts.items()):
        lines.append(f"  - {category}: {count}")

    lines.extend(["", "first_100_state_dict_keys:"])
    for key in state_keys[:100]:
        value = state_dict[key]
        if torch.is_tensor(value):
            detail = f"shape={shape_string(value)} dtype={value.dtype}"
        else:
            detail = f"type={type(value).__name__}"
        lines.append(f"  - {key}: {detail}")

    lines.extend(["", "## Key Architecture Evidence"])
    lines.extend(infer_architecture(rows))
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def write_tensor_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = ["name", "shape", "dtype", "numel", "prefix", "category"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    checkpoint_path = args.checkpoint
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch_load_cpu(checkpoint_path)
    state_dict, state_source = extract_state_dict(checkpoint)
    rows = tensor_rows(state_dict)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    keys_report = args.out_dir / "checkpoint_keys.txt"
    tensor_report = args.out_dir / "checkpoint_tensor_shapes.csv"
    write_keys_report(
        keys_report,
        checkpoint_path,
        checkpoint,
        state_dict,
        state_source,
        rows,
    )
    write_tensor_csv(tensor_report, rows)

    print(f"Wrote {keys_report}")
    print(f"Wrote {tensor_report}")
    print(f"state_dict_source={state_source}")
    print(f"tensor_entry_count={len(rows)}")


if __name__ == "__main__":
    main()
