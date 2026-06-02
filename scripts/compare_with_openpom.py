#!/usr/bin/env python
"""Compare local MPNNPOM outputs against exported OpenPOM reference outputs."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

DEFAULT_REQUESTED_CHECKPOINT = Path(
    "/home/xing/openpom/models/ensemble_models/experiments_1/checkpoint2.pt"
)
DEFAULT_LOCAL_CHECKPOINT = REPO_ROOT / "checkpoints/openpom_experiments_1_checkpoint2.pt"
DEFAULT_REFERENCE_NPZ = REPO_ROOT / "reference_outputs/openpom_reference.npz"
DEFAULT_REPORT = REPO_ROOT / "reports/output_parity_single_checkpoint.json"
DEBUG_HINTS = [
    "featurizer value/order",
    "graph batching and edge order",
    "checkpoint state_dict loading",
    "atom+bond readout implementation",
    "Set2Set configuration",
    "FFN embedding/output behavior",
]


@dataclass(frozen=True)
class ResolvedCheckpoint:
    requested_path: str
    resolved_path: str
    requested_is_lfs_pointer: bool
    resolution: str
    sha256: str
    size_bytes: int
    lfs_pointer: dict[str, str] | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the local checkpoint-compatible MPNNPOM implementation "
            "against Phase 1 OpenPOM reference outputs."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_REQUESTED_CHECKPOINT,
        help=f"Checkpoint path. Default: {DEFAULT_REQUESTED_CHECKPOINT}",
    )
    parser.add_argument(
        "--reference-npz",
        type=Path,
        default=DEFAULT_REFERENCE_NPZ,
        help=f"OpenPOM reference NPZ. Default: {DEFAULT_REFERENCE_NPZ}",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT,
        help=f"JSON report path. Default: {DEFAULT_REPORT}",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device for local inference. Default: cpu.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of top probability labels to compare. Default: 10.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
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


def _pointer_sha256(pointer: dict[str, str]) -> str | None:
    oid = pointer.get("oid")
    if oid is None:
        return None
    if oid.startswith("sha256:"):
        return oid.removeprefix("sha256:")
    return None


def resolve_checkpoint(
    requested_path: Path,
    local_checkpoint: Path = DEFAULT_LOCAL_CHECKPOINT,
) -> ResolvedCheckpoint:
    requested_path = requested_path.expanduser()
    if not requested_path.is_absolute():
        requested_path = (REPO_ROOT / requested_path).resolve()
    else:
        requested_path = requested_path.resolve()

    if not requested_path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {requested_path}")

    pointer = read_lfs_pointer(requested_path)
    if pointer is None:
        return ResolvedCheckpoint(
            requested_path=str(requested_path),
            resolved_path=str(requested_path),
            requested_is_lfs_pointer=False,
            resolution="requested_checkpoint",
            sha256=sha256_file(requested_path),
            size_bytes=requested_path.stat().st_size,
            lfs_pointer=None,
        )

    pointer_sha = _pointer_sha256(pointer)
    if not local_checkpoint.is_absolute():
        local_checkpoint = (REPO_ROOT / local_checkpoint).resolve()
    else:
        local_checkpoint = local_checkpoint.resolve()

    if not local_checkpoint.exists():
        raise FileNotFoundError(
            "Requested checkpoint is a Git LFS pointer, and the local real "
            f"checkpoint copy is missing: {local_checkpoint}"
        )

    local_sha = sha256_file(local_checkpoint)
    if pointer_sha is not None and local_sha != pointer_sha:
        raise ValueError(
            "Requested checkpoint is a Git LFS pointer, but the local checkpoint "
            "copy does not match its sha256 oid:\n"
            f"  pointer oid: {pointer_sha}\n"
            f"  local sha:   {local_sha}\n"
            f"  local path:  {local_checkpoint}"
        )

    return ResolvedCheckpoint(
        requested_path=str(requested_path),
        resolved_path=str(local_checkpoint),
        requested_is_lfs_pointer=True,
        resolution="lfs_pointer_resolved_to_local_copy",
        sha256=local_sha,
        size_bytes=local_checkpoint.stat().st_size,
        lfs_pointer=pointer,
    )


def load_reference_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Reference NPZ does not exist: {path}")
    data = np.load(path, allow_pickle=True)
    required_keys = {"smiles", "probs", "logits", "embeddings", "task_labels"}
    missing = sorted(required_keys - set(data.files))
    if missing:
        raise KeyError(f"Reference NPZ is missing required keys: {missing}")
    return {key: np.asarray(data[key]) for key in required_keys}


def max_abs_diff(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        return float("inf")
    if left.size == 0 and right.size == 0:
        return 0.0
    return float(np.max(np.abs(left - right)))


def mean_abs_diff(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        return float("inf")
    if left.size == 0 and right.size == 0:
        return 0.0
    return float(np.mean(np.abs(left - right)))


def embedding_cosine_similarity(
    local_embeddings: np.ndarray,
    reference_embeddings: np.ndarray,
) -> np.ndarray:
    if local_embeddings.shape != reference_embeddings.shape:
        raise ValueError(
            "Embedding shape mismatch: "
            f"local={local_embeddings.shape}, reference={reference_embeddings.shape}"
        )
    numerator = np.sum(local_embeddings * reference_embeddings, axis=1)
    local_norm = np.linalg.norm(local_embeddings, axis=1)
    reference_norm = np.linalg.norm(reference_embeddings, axis=1)
    denominator = local_norm * reference_norm
    if np.any(denominator == 0):
        raise ValueError("Cannot compute cosine similarity for zero-norm embeddings.")
    return numerator / denominator


def top_k_indices(probs: np.ndarray, top_k: int) -> np.ndarray:
    if probs.ndim != 2:
        raise ValueError(f"Expected probability array to be 2-D, got {probs.ndim}-D.")
    if top_k <= 0:
        raise ValueError("--top-k must be positive.")
    if top_k > probs.shape[1]:
        raise ValueError(
            f"--top-k={top_k} exceeds number of labels ({probs.shape[1]})."
        )
    return np.argsort(-probs, axis=1, kind="stable")[:, :top_k]


def compare_top_k(
    local_probs: np.ndarray,
    reference_probs: np.ndarray,
    labels: list[str],
    top_k: int,
) -> list[dict[str, Any]]:
    local_top = top_k_indices(local_probs, top_k)
    reference_top = top_k_indices(reference_probs, top_k)

    comparisons: list[dict[str, Any]] = []
    for local_indices, reference_indices in zip(local_top, reference_top, strict=True):
        local_index_list = [int(index) for index in local_indices]
        reference_index_list = [int(index) for index in reference_indices]
        overlap_count = len(set(local_index_list) & set(reference_index_list))
        comparisons.append(
            {
                "reference_indices": reference_index_list,
                "local_indices": local_index_list,
                "reference_labels": [labels[index] for index in reference_index_list],
                "local_labels": [labels[index] for index in local_index_list],
                "overlap_count": overlap_count,
                "overlap_fraction": float(overlap_count / top_k),
                "order_match": bool(local_index_list == reference_index_list),
            }
        )
    return comparisons


def validate_output_arrays(
    local: dict[str, np.ndarray],
    reference: dict[str, np.ndarray],
) -> None:
    for key in ["probs", "logits", "embeddings"]:
        if local[key].shape != reference[key].shape:
            raise ValueError(
                f"{key} shape mismatch: local={local[key].shape}, "
                f"reference={reference[key].shape}"
            )
        if not np.isfinite(reference[key]).all():
            raise ValueError(f"Reference {key} contains NaN or Inf.")
        if not np.isfinite(local[key]).all():
            raise ValueError(f"Local {key} contains NaN or Inf.")


def compute_metrics(
    local: dict[str, np.ndarray],
    reference: dict[str, np.ndarray],
    labels: list[str],
    top_k: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    validate_output_arrays(local, reference)
    cosine = embedding_cosine_similarity(local["embeddings"], reference["embeddings"])
    top_k_comparisons = compare_top_k(
        local_probs=local["probs"],
        reference_probs=reference["probs"],
        labels=labels,
        top_k=top_k,
    )
    overlap_counts = [item["overlap_count"] for item in top_k_comparisons]
    order_matches = [item["order_match"] for item in top_k_comparisons]

    metrics = {
        "probs_max_abs_diff": max_abs_diff(local["probs"], reference["probs"]),
        "probs_mean_abs_diff": mean_abs_diff(local["probs"], reference["probs"]),
        "logits_max_abs_diff": max_abs_diff(local["logits"], reference["logits"]),
        "logits_mean_abs_diff": mean_abs_diff(local["logits"], reference["logits"]),
        "embedding_cosine_min": float(np.min(cosine)),
        "embedding_cosine_mean": float(np.mean(cosine)),
        "top_k": int(top_k),
        "top_k_min_overlap": int(min(overlap_counts)),
        "top_k_all_order_match": bool(all(order_matches)),
        # Backward-compatible aliases for reports/tests created when top_k was fixed at 10.
        "top10_min_overlap": int(min(overlap_counts)),
        "top10_all_order_match": bool(all(order_matches)),
    }
    return metrics, top_k_comparisons


def classify_status(metrics: dict[str, Any], top_k: int) -> str:
    top_k_order_match = metrics.get(
        "top_k_all_order_match",
        metrics["top10_all_order_match"],
    )
    top_k_min_overlap = metrics.get("top_k_min_overlap", metrics["top10_min_overlap"])
    if (
        metrics["probs_max_abs_diff"] <= 1e-4
        and top_k_order_match
        and metrics["embedding_cosine_min"] >= 0.999999
    ):
        return "ideal"

    required_overlap = min(8, top_k)
    if (
        metrics["probs_max_abs_diff"] <= 1e-3
        and top_k_min_overlap >= required_overlap
        and metrics["embedding_cosine_min"] >= 0.999
    ):
        return "acceptable"

    return "fail"


def _tensor_to_numpy(tensor: Any) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def _torch_load_checkpoint(path: Path, map_location: str) -> Any:
    import torch

    kwargs: dict[str, Any] = {"map_location": map_location}
    if "weights_only" in inspect.signature(torch.load).parameters:
        kwargs["weights_only"] = False
    return torch.load(path, **kwargs)


def run_local_model(
    smiles: list[str],
    checkpoint_path: Path,
    device: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    import torch

    from pom_repro.checkpoint import load_checkpoint_strict
    from pom_repro.featurizer import GraphFeaturizer
    from pom_repro.graph_batch import batch_graphs
    from pom_repro.model import MPNNPOM

    torch_device = torch.device(device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"Requested device {device!r}, but torch.cuda.is_available() is false."
        )

    graphs = GraphFeaturizer().featurize(smiles)
    if len(graphs) != len(smiles):
        raise RuntimeError(f"Expected {len(smiles)} graphs, got {len(graphs)}.")
    for smile, graph in zip(smiles, graphs, strict=True):
        if isinstance(graph, np.ndarray) and graph.size == 0:
            raise ValueError(f"Failed to featurize SMILES: {smile}")

    batched_graph = batch_graphs(graphs, device=torch_device)
    model = MPNNPOM()
    checkpoint_report = load_checkpoint_strict(
        model=model,
        checkpoint_path=checkpoint_path,
        map_location="cpu",
    )
    model.to(torch_device)
    model.eval()

    with torch.no_grad():
        probs_tensor, logits_tensor, embeddings_tensor = model(batched_graph)

    outputs = {
        "probs": _tensor_to_numpy(probs_tensor),
        "logits": _tensor_to_numpy(logits_tensor),
        "embeddings": _tensor_to_numpy(embeddings_tensor),
    }
    return outputs, asdict(checkpoint_report)


def build_sample_reports(
    smiles: list[str],
    cosine: np.ndarray,
    top_k_comparisons: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    samples = []
    for index, smile in enumerate(smiles):
        comparison = top_k_comparisons[index]
        samples.append(
            {
                "smiles": smile,
                "embedding_cosine_similarity": float(cosine[index]),
                "reference_top_k_indices": comparison["reference_indices"],
                "local_top_k_indices": comparison["local_indices"],
                "reference_top_k_labels": comparison["reference_labels"],
                "local_top_k_labels": comparison["local_labels"],
                "top_k_overlap_count": comparison["overlap_count"],
                "top_k_overlap_fraction": comparison["overlap_fraction"],
                "top_k_order_match": comparison["order_match"],
                # Backward-compatible aliases for existing top-10 reports.
                "reference_top10_indices": comparison["reference_indices"],
                "local_top10_indices": comparison["local_indices"],
                "reference_top10_labels": comparison["reference_labels"],
                "local_top10_labels": comparison["local_labels"],
                "top10_overlap_count": comparison["overlap_count"],
                "top10_overlap_fraction": comparison["overlap_fraction"],
                "top10_order_match": comparison["order_match"],
            }
        )
    return samples


def write_report(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()

    reference_path = args.reference_npz.resolve()
    reference = load_reference_npz(reference_path)
    smiles = [str(item) for item in reference["smiles"].tolist()]
    labels = [str(item) for item in reference["task_labels"].tolist()]
    if reference["probs"].shape[1] != len(labels):
        raise ValueError(
            "Reference label count does not match probs width: "
            f"{len(labels)} labels vs {reference['probs'].shape[1]} probabilities."
        )

    resolved_checkpoint = resolve_checkpoint(args.checkpoint)
    local_outputs, checkpoint_report = run_local_model(
        smiles=smiles,
        checkpoint_path=Path(resolved_checkpoint.resolved_path),
        device=args.device,
    )
    metrics, top_k_comparisons = compute_metrics(
        local=local_outputs,
        reference=reference,
        labels=labels,
        top_k=args.top_k,
    )
    cosine = embedding_cosine_similarity(
        local_outputs["embeddings"],
        reference["embeddings"],
    )
    status = classify_status(metrics, args.top_k)

    report = {
        "status": status,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "metadata": {
            "reference_npz": str(reference_path),
            "checkpoint": asdict(resolved_checkpoint),
            "device": args.device,
            "num_smiles": len(smiles),
            "num_labels": len(labels),
            "top_k": args.top_k,
            "output_shapes": {
                "reference": {
                    "probs": list(reference["probs"].shape),
                    "logits": list(reference["logits"].shape),
                    "embeddings": list(reference["embeddings"].shape),
                },
                "local": {
                    "probs": list(local_outputs["probs"].shape),
                    "logits": list(local_outputs["logits"].shape),
                    "embeddings": list(local_outputs["embeddings"].shape),
                },
            },
            "checkpoint_load_report": checkpoint_report,
        },
        "metrics": metrics,
        "samples": build_sample_reports(
            smiles=smiles,
            cosine=cosine,
            top_k_comparisons=top_k_comparisons,
        ),
        "debug_hints": [] if status in {"ideal", "acceptable"} else DEBUG_HINTS,
    }
    write_report(report, args.report)

    print(f"Wrote {args.report}")
    print(f"status: {status}")
    print(f"probs max_abs_diff: {metrics['probs_max_abs_diff']:.8g}")
    print(f"probs mean_abs_diff: {metrics['probs_mean_abs_diff']:.8g}")
    print(f"logits max_abs_diff: {metrics['logits_max_abs_diff']:.8g}")
    print(f"logits mean_abs_diff: {metrics['logits_mean_abs_diff']:.8g}")
    print(f"embedding cosine min: {metrics['embedding_cosine_min']:.8g}")
    print(f"top-{args.top_k} min overlap: {metrics['top_k_min_overlap']}")
    print(f"top-{args.top_k} all order match: {metrics['top_k_all_order_match']}")
    return 0 if status in {"ideal", "acceptable"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
