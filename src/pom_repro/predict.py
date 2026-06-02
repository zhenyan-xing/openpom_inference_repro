"""User-facing OpenPOM-compatible SMILES prediction helpers."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np


ODOR_LABELS: tuple[str, ...] = (
    "alcoholic",
    "aldehydic",
    "alliaceous",
    "almond",
    "amber",
    "animal",
    "anisic",
    "apple",
    "apricot",
    "aromatic",
    "balsamic",
    "banana",
    "beefy",
    "bergamot",
    "berry",
    "bitter",
    "black currant",
    "brandy",
    "burnt",
    "buttery",
    "cabbage",
    "camphoreous",
    "caramellic",
    "cedar",
    "celery",
    "chamomile",
    "cheesy",
    "cherry",
    "chocolate",
    "cinnamon",
    "citrus",
    "clean",
    "clove",
    "cocoa",
    "coconut",
    "coffee",
    "cognac",
    "cooked",
    "cooling",
    "cortex",
    "coumarinic",
    "creamy",
    "cucumber",
    "dairy",
    "dry",
    "earthy",
    "ethereal",
    "fatty",
    "fermented",
    "fishy",
    "floral",
    "fresh",
    "fruit skin",
    "fruity",
    "garlic",
    "gassy",
    "geranium",
    "grape",
    "grapefruit",
    "grassy",
    "green",
    "hawthorn",
    "hay",
    "hazelnut",
    "herbal",
    "honey",
    "hyacinth",
    "jasmin",
    "juicy",
    "ketonic",
    "lactonic",
    "lavender",
    "leafy",
    "leathery",
    "lemon",
    "lily",
    "malty",
    "meaty",
    "medicinal",
    "melon",
    "metallic",
    "milky",
    "mint",
    "muguet",
    "mushroom",
    "musk",
    "musty",
    "natural",
    "nutty",
    "odorless",
    "oily",
    "onion",
    "orange",
    "orangeflower",
    "orris",
    "ozone",
    "peach",
    "pear",
    "phenolic",
    "pine",
    "pineapple",
    "plum",
    "popcorn",
    "potato",
    "powdery",
    "pungent",
    "radish",
    "raspberry",
    "ripe",
    "roasted",
    "rose",
    "rummy",
    "sandalwood",
    "savory",
    "sharp",
    "smoky",
    "soapy",
    "solvent",
    "sour",
    "spicy",
    "strawberry",
    "sulfurous",
    "sweaty",
    "sweet",
    "tea",
    "terpenic",
    "tobacco",
    "tomato",
    "tropical",
    "vanilla",
    "vegetable",
    "vetiver",
    "violet",
    "warm",
    "waxy",
    "weedy",
    "winey",
    "woody",
)

_LFS_HEADER = "version https://git-lfs.github.com/spec/v1"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_user_path(path: str | Path) -> Path:
    path = Path(path).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (Path.cwd() / path).resolve()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_lfs_pointer(path: Path) -> dict[str, str] | None:
    if not path.exists() or path.stat().st_size > 1024:
        return None
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return None
    if not text.startswith(_LFS_HEADER):
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


def _checkpoint_search_roots() -> list[Path]:
    roots = [_repo_root() / "checkpoints", Path.cwd() / "checkpoints"]
    unique_roots: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        resolved = root.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique_roots.append(resolved)
    return unique_roots


def _find_checkpoint_by_sha256(expected_sha256: str) -> Path | None:
    for root in _checkpoint_search_roots():
        if not root.exists():
            continue
        for candidate in sorted(root.rglob("*")):
            if not candidate.is_file():
                continue
            if _read_lfs_pointer(candidate) is not None:
                continue
            if _sha256_file(candidate) == expected_sha256:
                return candidate.resolve()
    return None


def resolve_checkpoint_path(checkpoint_path: str | Path) -> Path:
    """Resolve a real checkpoint, rejecting unresolved Git LFS pointers."""

    requested_path = _resolve_user_path(checkpoint_path)
    if not requested_path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {requested_path}")

    pointer = _read_lfs_pointer(requested_path)
    if pointer is None:
        return requested_path

    pointer_sha = _pointer_sha256(pointer)
    if pointer_sha is None:
        raise ValueError(
            f"Checkpoint is a Git LFS pointer without a sha256 oid: {requested_path}"
        )

    resolved = _find_checkpoint_by_sha256(pointer_sha)
    if resolved is None:
        searched = ", ".join(str(root) for root in _checkpoint_search_roots())
        raise ValueError(
            "Checkpoint is a Git LFS pointer, but no matching real checkpoint "
            "was found under checkpoints/.\n"
            f"  requested: {requested_path}\n"
            f"  sha256:    {pointer_sha}\n"
            f"  searched:  {searched}"
        )
    return resolved


def resolve_checkpoint_paths(checkpoint_paths: list[str] | list[Path]) -> list[Path]:
    """Resolve checkpoint paths while preserving caller-provided order."""

    if not checkpoint_paths:
        raise ValueError("checkpoint_paths must contain at least one checkpoint.")
    return [resolve_checkpoint_path(path) for path in checkpoint_paths]


def _validate_smiles(smiles: list[str]) -> None:
    if not smiles:
        raise ValueError("smiles must contain at least one SMILES string.")
    for index, value in enumerate(smiles):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"smiles[{index}] must be a non-empty string.")


def _validate_top_k(top_k: int) -> None:
    if top_k <= 0:
        raise ValueError("top_k must be positive.")
    if top_k > len(ODOR_LABELS):
        raise ValueError(
            f"top_k={top_k} exceeds number of odor labels ({len(ODOR_LABELS)})."
        )


def _featurize_smiles(smiles: list[str]):
    from pom_repro.featurizer import GraphFeaturizer

    graphs = GraphFeaturizer().featurize(smiles)
    if len(graphs) != len(smiles):
        raise RuntimeError(f"Expected {len(smiles)} graphs, got {len(graphs)}.")
    for smile, graph in zip(smiles, graphs, strict=True):
        if isinstance(graph, np.ndarray) and graph.size == 0:
            raise ValueError(f"Failed to featurize SMILES: {smile}")
    return graphs


def _tensor_to_numpy(tensor: Any) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def _validate_output_arrays(probs: np.ndarray, embedding: np.ndarray | None) -> None:
    if not np.isfinite(probs).all():
        raise ValueError("Predicted probabilities contain NaN or Inf.")
    if np.any(probs < 0.0) or np.any(probs > 1.0):
        raise ValueError("Predicted probabilities are outside [0, 1].")
    if embedding is not None and not np.isfinite(embedding).all():
        raise ValueError("Predicted embeddings contain NaN or Inf.")


def _top_k_predictions(probs: np.ndarray, top_k: int) -> list[list[dict[str, Any]]]:
    top_indices = np.argsort(-probs, axis=1, kind="stable")[:, :top_k]
    predictions: list[list[dict[str, Any]]] = []
    for row_probs, row_indices in zip(probs, top_indices, strict=True):
        predictions.append(
            [
                {
                    "index": int(index),
                    "label": ODOR_LABELS[int(index)],
                    "prob": float(row_probs[int(index)]),
                }
                for index in row_indices
            ]
        )
    return predictions


def predict_smiles(
    smiles: list[str],
    checkpoint_paths: list[str],
    top_k: int = 10,
    device: str = "cpu",
    return_embedding: bool = True,
) -> dict[str, Any]:
    """Predict odor probabilities for SMILES with one or more checkpoints.

    Probabilities are averaged across checkpoints, matching OpenPOM's ensemble
    demo behavior. When requested, embeddings are averaged across checkpoints.
    """

    _validate_smiles(smiles)
    _validate_top_k(top_k)
    resolved_checkpoints = resolve_checkpoint_paths(checkpoint_paths)

    import torch

    torch_device = torch.device(device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"Requested device {device!r}, but torch.cuda.is_available() is false."
        )

    graphs = _featurize_smiles(smiles)

    from pom_repro.checkpoint import load_checkpoint_strict
    from pom_repro.graph_batch import batch_graphs
    from pom_repro.model import MPNNPOM

    batched_graph = batch_graphs(graphs, device=torch_device)

    all_probs: list[np.ndarray] = []
    all_embeddings: list[np.ndarray] = []

    with torch.no_grad():
        for checkpoint_path in resolved_checkpoints:
            model = MPNNPOM()
            load_checkpoint_strict(
                model=model,
                checkpoint_path=checkpoint_path,
                map_location="cpu",
            )
            model.to(torch_device)
            model.eval()

            probs_tensor, _, embeddings_tensor = model(batched_graph)
            all_probs.append(_tensor_to_numpy(probs_tensor))
            if return_embedding:
                all_embeddings.append(_tensor_to_numpy(embeddings_tensor))

    probs = np.mean(np.stack(all_probs, axis=0), axis=0)
    embedding = None
    if return_embedding:
        embedding = np.mean(np.stack(all_embeddings, axis=0), axis=0)

    _validate_output_arrays(probs=probs, embedding=embedding)

    return {
        "smiles": list(smiles),
        "labels": list(ODOR_LABELS),
        "probs": probs,
        "top_k": _top_k_predictions(probs, top_k),
        "checkpoint_paths": [str(path) for path in resolved_checkpoints],
        "embedding": embedding,
    }


__all__ = [
    "ODOR_LABELS",
    "predict_smiles",
    "resolve_checkpoint_path",
    "resolve_checkpoint_paths",
]
