#!/usr/bin/env python
"""Export fixed reference outputs from the original OpenPOM implementation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_OPENPOM_ROOT = Path("/home/xing/openpom")
DEFAULT_OPENPOM_CHECKPOINT = (
    DEFAULT_OPENPOM_ROOT
    / "models/ensemble_models/experiments_1/checkpoint2.pt"
)
DEFAULT_LOCAL_CHECKPOINT = Path("checkpoints/openpom_experiments_1_checkpoint2.pt")
DEFAULT_OUTPUT_DIR = Path("reference_outputs")
DEFAULT_SMILES = [
    "C",
    "CCO",
    "CC(=O)O",
    "c1ccccc1",
    "CC(C)O",
    "C1=CC=CC=C1O",
]

MODEL_KWARGS = {
    "node_out_feats": 100,
    "edge_hidden_feats": 75,
    "edge_out_feats": 100,
    "num_step_message_passing": 5,
    "mpnn_residual": True,
    "message_aggregator_type": "sum",
    "mode": "classification",
    "n_classes": 1,
    "readout_type": "set2set",
    "num_step_set2set": 3,
    "num_layer_set2set": 2,
    "ffn_hidden_list": [392, 392],
    "ffn_embeddings": 256,
    "ffn_activation": "relu",
    "ffn_dropout_p": 0.12,
    "ffn_dropout_at_input_no_act": False,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run original OpenPOM on a fixed SMILES set and export reference "
            "probs/logits/embeddings plus graph feature summaries."
        )
    )
    parser.add_argument(
        "--openpom-root",
        type=Path,
        default=DEFAULT_OPENPOM_ROOT,
        help=f"Read-only OpenPOM source tree. Default: {DEFAULT_OPENPOM_ROOT}",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help=(
            "Checkpoint to load. Default: checkpoints/openpom_experiments_1_"
            "checkpoint2.pt if present, otherwise OpenPOM experiments_1/checkpoint2.pt."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for JSON/NPZ outputs. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--json-name",
        default="openpom_reference.json",
        help="JSON output filename inside --output-dir.",
    )
    parser.add_argument(
        "--npz-name",
        default="openpom_reference.npz",
        help="NPZ output filename inside --output-dir.",
    )
    parser.add_argument(
        "--smiles",
        action="append",
        default=None,
        help="SMILES to export. Repeat to override the default fixed list.",
    )
    parser.add_argument(
        "--smiles-file",
        type=Path,
        default=None,
        help="Optional newline-delimited SMILES file. Overrides defaults.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device for inference. Default: cpu.",
    )
    parser.add_argument(
        "--allow-pointer-checkpoint",
        action="store_true",
        help="Disable the Git LFS pointer guard. This is only for debugging.",
    )
    return parser.parse_args()


def configure_runtime_env() -> None:
    tmp_root = Path("/tmp/openpom_reference_runtime")
    tmp_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("WANDB_DISABLED", "true")
    os.environ.setdefault("WANDB_MODE", "disabled")
    os.environ.setdefault("WANDB_DIR", str(tmp_root / "wandb"))
    os.environ.setdefault("XDG_CONFIG_HOME", str(tmp_root / "config"))
    os.environ.setdefault("MPLBACKEND", "Agg")


def resolve_checkpoint(explicit_checkpoint: Path | None) -> Path:
    if explicit_checkpoint is not None:
        return explicit_checkpoint
    if DEFAULT_LOCAL_CHECKPOINT.exists():
        return DEFAULT_LOCAL_CHECKPOINT
    return DEFAULT_OPENPOM_CHECKPOINT


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    arr = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(arr.shape).encode("utf-8"))
    digest.update(str(arr.dtype).encode("utf-8"))
    digest.update(arr.tobytes())
    return digest.hexdigest()


def read_lfs_pointer(path: Path) -> dict[str, str] | None:
    if not path.exists() or path.stat().st_size > 1024:
        return None
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return None
    if not text.startswith("version https://git-lfs.github.com/spec/v1"):
        return None

    pointer: dict[str, str] = {}
    for line in text.splitlines():
        if line.startswith("version "):
            pointer["version"] = line.removeprefix("version ")
        elif line.startswith("oid "):
            pointer["oid"] = line.removeprefix("oid ")
        elif line.startswith("size "):
            pointer["size"] = line.removeprefix("size ")
    return pointer


def require_real_checkpoint(path: Path, allow_pointer: bool) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")

    pointer = read_lfs_pointer(path)
    if pointer and not allow_pointer:
        raise SystemExit(
            "Checkpoint is a Git LFS pointer, not real weights:\n"
            f"  path: {path}\n"
            f"  oid: {pointer.get('oid', 'unknown')}\n"
            f"  expected size: {pointer.get('size', 'unknown')} bytes\n"
            "Fetch or copy the real checkpoint, then rerun with --checkpoint "
            "if it is stored outside the default location."
        )

    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "is_lfs_pointer": pointer is not None,
        "lfs_pointer": pointer,
    }


def git_commit(path: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip() or None


def load_smiles(args: argparse.Namespace) -> list[str]:
    if args.smiles_file is not None:
        lines = args.smiles_file.read_text(encoding="utf-8").splitlines()
        smiles = [line.strip() for line in lines if line.strip()]
    elif args.smiles is not None:
        smiles = args.smiles
    else:
        smiles = DEFAULT_SMILES

    if not smiles:
        raise ValueError("SMILES list is empty.")
    return smiles


def load_task_labels(openpom_root: Path) -> list[str]:
    dataset = (
        openpom_root
        / "openpom/data/curated_datasets/curated_GS_LF_merged_4983.csv"
    )
    with dataset.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader)
    return header[2:]


def import_openpom(openpom_root: Path) -> dict[str, Any]:
    sys.path.insert(0, str(openpom_root))

    import dgl  # noqa: PLC0415
    import deepchem  # noqa: PLC0415
    import rdkit  # noqa: PLC0415
    import torch  # noqa: PLC0415
    from openpom.feat.graph_featurizer import (  # noqa: PLC0415
        GraphConvConstants,
        GraphFeaturizer,
    )
    from openpom.models.mpnn_pom import MPNNPOM  # noqa: PLC0415

    return {
        "dgl": dgl,
        "deepchem": deepchem,
        "rdkit": rdkit,
        "torch": torch,
        "GraphConvConstants": GraphConvConstants,
        "GraphFeaturizer": GraphFeaturizer,
        "MPNNPOM": MPNNPOM,
    }


def load_checkpoint(torch_module: Any, path: Path, device: str) -> dict[str, Any]:
    kwargs = {"map_location": device}
    if "weights_only" in inspect.signature(torch_module.load).parameters:
        kwargs["weights_only"] = False
    checkpoint = torch_module.load(path, **kwargs)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    if isinstance(checkpoint, dict):
        return checkpoint
    raise TypeError(f"Unsupported checkpoint object type: {type(checkpoint)!r}")


def summarize_graph(smiles: str, graph: Any) -> dict[str, Any]:
    node_features = np.asarray(graph.node_features)
    edge_features = np.asarray(graph.edge_features)
    edge_index = np.asarray(graph.edge_index)
    return {
        "smiles": smiles,
        "num_nodes": int(node_features.shape[0]),
        "num_directed_edges": int(edge_features.shape[0]),
        "node_feature_shape": list(node_features.shape),
        "edge_feature_shape": list(edge_features.shape),
        "edge_index_shape": list(edge_index.shape),
        "node_feature_sum": float(node_features.sum()),
        "edge_feature_sum": float(edge_features.sum()),
        "edge_index_sum": float(edge_index.sum()),
        "node_feature_sha256": sha256_array(node_features),
        "edge_feature_sha256": sha256_array(edge_features),
        "edge_index_sha256": sha256_array(edge_index),
    }


def graph_summary_arrays(summaries: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    return {
        "graph_num_nodes": np.asarray(
            [item["num_nodes"] for item in summaries], dtype=np.int64
        ),
        "graph_num_directed_edges": np.asarray(
            [item["num_directed_edges"] for item in summaries], dtype=np.int64
        ),
        "graph_node_feature_shapes": np.asarray(
            [item["node_feature_shape"] for item in summaries], dtype=np.int64
        ),
        "graph_edge_feature_shapes": np.asarray(
            [item["edge_feature_shape"] for item in summaries], dtype=np.int64
        ),
        "graph_edge_index_shapes": np.asarray(
            [item["edge_index_shape"] for item in summaries], dtype=np.int64
        ),
        "graph_node_feature_sums": np.asarray(
            [item["node_feature_sum"] for item in summaries], dtype=np.float64
        ),
        "graph_edge_feature_sums": np.asarray(
            [item["edge_feature_sum"] for item in summaries], dtype=np.float64
        ),
        "graph_edge_index_sums": np.asarray(
            [item["edge_index_sum"] for item in summaries], dtype=np.float64
        ),
        "graph_node_feature_sha256": np.asarray(
            [item["node_feature_sha256"] for item in summaries]
        ),
        "graph_edge_feature_sha256": np.asarray(
            [item["edge_feature_sha256"] for item in summaries]
        ),
        "graph_edge_index_sha256": np.asarray(
            [item["edge_index_sha256"] for item in summaries]
        ),
    }


def tensor_to_numpy(tensor: Any) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def main() -> int:
    args = parse_args()
    configure_runtime_env()

    openpom_root = args.openpom_root.resolve()
    checkpoint = resolve_checkpoint(args.checkpoint).resolve()
    checkpoint_info = require_real_checkpoint(
        checkpoint, allow_pointer=args.allow_pointer_checkpoint
    )
    smiles = load_smiles(args)
    task_labels = load_task_labels(openpom_root)
    modules = import_openpom(openpom_root)

    dgl = modules["dgl"]
    torch = modules["torch"]
    GraphConvConstants = modules["GraphConvConstants"]
    GraphFeaturizer = modules["GraphFeaturizer"]
    MPNNPOM = modules["MPNNPOM"]

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested --device cuda, but torch.cuda.is_available() is false.")

    featurizer = GraphFeaturizer()
    graphs = list(featurizer.featurize(smiles))
    if len(graphs) != len(smiles):
        raise RuntimeError(
            f"Expected {len(smiles)} featurized graphs, got {len(graphs)}."
        )

    graph_summaries = [
        summarize_graph(smile, graph) for smile, graph in zip(smiles, graphs)
    ]
    dgl_graphs = [graph.to_dgl_graph(self_loop=False) for graph in graphs]
    batched_graph = dgl.batch(dgl_graphs).to(args.device)

    model_kwargs = {
        **MODEL_KWARGS,
        "n_tasks": len(task_labels),
        "number_atom_features": GraphConvConstants.ATOM_FDIM,
        "number_bond_features": GraphConvConstants.BOND_FDIM,
    }
    model = MPNNPOM(**model_kwargs)
    state_dict = load_checkpoint(torch, checkpoint, args.device)
    model.load_state_dict(state_dict)
    model.to(args.device)
    model.eval()

    with torch.no_grad():
        probs_tensor, logits_tensor, embeddings_tensor = model(batched_graph)

    probs = tensor_to_numpy(probs_tensor)
    logits = tensor_to_numpy(logits_tensor)
    embeddings = tensor_to_numpy(embeddings_tensor)

    if list(probs.shape) != [len(smiles), len(task_labels)]:
        raise RuntimeError(f"Unexpected probs shape: {probs.shape}")
    if list(logits.shape) != [len(smiles), len(task_labels), 1]:
        raise RuntimeError(f"Unexpected logits shape: {logits.shape}")
    if list(embeddings.shape) != [len(smiles), MODEL_KWARGS["ffn_embeddings"]]:
        raise RuntimeError(f"Unexpected embeddings shape: {embeddings.shape}")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / args.json_name
    npz_path = output_dir / args.npz_name

    version_info = {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "dgl": modules["dgl"].__version__,
        "deepchem": modules["deepchem"].__version__,
        "rdkit": modules["rdkit"].__version__,
    }

    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "openpom_root": str(openpom_root),
        "openpom_commit": git_commit(openpom_root),
        "checkpoint": checkpoint_info,
        "device": args.device,
        "versions": version_info,
        "model_kwargs": model_kwargs,
        "task_labels": task_labels,
        "output_shapes": {
            "probs": list(probs.shape),
            "logits": list(logits.shape),
            "embeddings": list(embeddings.shape),
        },
    }
    samples = []
    for index, smile in enumerate(smiles):
        samples.append(
            {
                "smiles": smile,
                "graph_summary": graph_summaries[index],
                "probs": probs[index].tolist(),
                "logits": logits[index].tolist(),
                "embedding": embeddings[index].tolist(),
            }
        )

    json_payload = {"metadata": metadata, "samples": samples}
    json_path.write_text(
        json.dumps(json_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    np.savez_compressed(
        npz_path,
        smiles=np.asarray(smiles),
        probs=probs,
        logits=logits,
        embeddings=embeddings,
        task_labels=np.asarray(task_labels),
        **graph_summary_arrays(graph_summaries),
    )

    print(f"Wrote {json_path}")
    print(f"Wrote {npz_path}")
    print(f"probs shape: {probs.shape}")
    print(f"logits shape: {logits.shape}")
    print(f"embeddings shape: {embeddings.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
