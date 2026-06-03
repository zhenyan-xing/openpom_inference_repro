#!/usr/bin/env python
"""Train an MLP odor head from frozen molecular LM embeddings."""

from __future__ import annotations

import argparse
import json
import math
import random
import shlex
import subprocess
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

DEFAULT_EMBEDDINGS = Path("artifacts/lm_embeddings/openpom_4983_molformer.npz")
DEFAULT_OUTPUT_DIR = Path("outputs/molformer_head")


@dataclass(frozen=True)
class FoldResult:
    fold: int
    train_size: int
    val_size: int
    best_epoch: int
    best_val_loss: float
    final_train_loss: float
    metrics: dict[str, Any]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train a 138-label MLP head from frozen molecular language model "
            "embeddings exported by scripts/export_lm_embeddings.py."
        )
    )
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=DEFAULT_EMBEDDINGS,
        help=f"Input embeddings .npz. Default: {DEFAULT_EMBEDDINGS}",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help=(
            "Original OpenPOM CSV path to record in config. The training input "
            "is still --embeddings."
        ),
    )
    parser.add_argument(
        "--model-id",
        default=None,
        help="Pretrained LM id to record in config, overriding the embedding metadata.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Local pretrained LM path to record in config, overriding the embedding metadata.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for config/head/metrics. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=80,
        help="Training epochs for each CV fold and final head. Default: 80.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="MLP training batch size. Default: 32.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="AdamW learning rate. Default: 1e-3.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="AdamW weight decay. Default: 1e-4.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help="MLP dropout. Default: 0.2.",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=512,
        help="Hidden layer width. Default: 512.",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=5,
        help="Number of greedy multilabel CV folds. Default: 5.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed. Default: 42.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device. Default: cuda if available, otherwise cpu.",
    )
    parser.add_argument(
        "--pos-weight-max",
        type=float,
        default=50.0,
        help="Clip BCE positive class weights to this value. Default: 50.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=20,
        help="Early stopping patience for CV folds. Use 0 to disable. Default: 20.",
    )
    parser.add_argument(
        "--skip-cv",
        action="store_true",
        help="Skip CV and only train the final head on all embeddings.",
    )
    return parser


def resolve_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (Path.cwd() / candidate).resolve()


def detect_git_commit(repo_root: Path | None = None) -> str:
    repo_root = repo_root or REPO_ROOT
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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)

    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str | None, torch_module: Any) -> str:
    if device:
        return device
    return "cuda" if torch_module.cuda.is_available() else "cpu"


def load_embedding_npz(path: Path) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Embeddings file does not exist: {path}")
    data = np.load(path, allow_pickle=False)
    required = {"embeddings", "labels", "label_names", "smiles", "model_id", "model_path"}
    missing = sorted(required - set(data.files))
    if missing:
        raise KeyError(f"Embedding NPZ is missing required keys: {missing}")

    embeddings = np.asarray(data["embeddings"], dtype=np.float32)
    labels = np.asarray(data["labels"], dtype=np.float32)
    label_names = [str(item) for item in data["label_names"].tolist()]
    smiles = [str(item) for item in data["smiles"].tolist()]
    model_id = str(np.asarray(data["model_id"]).item())
    model_path = str(np.asarray(data["model_path"]).item())

    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must be 2-D, got {embeddings.shape}.")
    if labels.ndim != 2:
        raise ValueError(f"labels must be 2-D, got {labels.shape}.")
    if embeddings.shape[0] != labels.shape[0]:
        raise ValueError(
            f"Sample count mismatch: embeddings={embeddings.shape[0]}, "
            f"labels={labels.shape[0]}."
        )
    if labels.shape[1] != len(label_names):
        raise ValueError(
            f"Label width mismatch: labels={labels.shape[1]}, "
            f"label_names={len(label_names)}."
        )
    if len(smiles) != embeddings.shape[0]:
        raise ValueError(
            f"SMILES count mismatch: smiles={len(smiles)}, "
            f"embeddings={embeddings.shape[0]}."
        )
    if not np.isfinite(embeddings).all():
        raise ValueError("embeddings contain NaN or Inf.")
    if not np.isfinite(labels).all():
        raise ValueError("labels contain NaN or Inf.")
    if not np.all((labels == 0.0) | (labels == 1.0)):
        raise ValueError("labels must be binary 0/1 values.")

    metadata = {
        "smiles_count": len(smiles),
        "model_id": model_id,
        "model_path": model_path,
    }
    return embeddings, labels, label_names, metadata


def compute_standardization(
    embeddings: np.ndarray,
    eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    mean = embeddings.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = embeddings.std(axis=0, dtype=np.float64).astype(np.float32)
    std = np.where(std < eps, 1.0, std).astype(np.float32)
    return mean, std


def standardize(
    embeddings: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    return ((embeddings - mean) / std).astype(np.float32)


def compute_pos_weight(
    labels: np.ndarray,
    max_value: float,
) -> np.ndarray:
    if max_value <= 0:
        raise ValueError("pos_weight_max must be positive.")
    positives = labels.sum(axis=0)
    negatives = labels.shape[0] - positives
    weights = np.ones(labels.shape[1], dtype=np.float32)
    mask = positives > 0
    weights[mask] = (negatives[mask] / positives[mask]).astype(np.float32)
    weights = np.clip(weights, 1.0, float(max_value)).astype(np.float32)
    return weights


def greedy_multilabel_kfold(
    labels: np.ndarray,
    n_splits: int,
    seed: int,
) -> list[np.ndarray]:
    """Greedy multilabel fold assignment balancing rare positive labels."""

    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")
    if n_splits > labels.shape[0]:
        raise ValueError("n_splits cannot exceed number of samples.")

    rng = np.random.default_rng(seed)
    binary = labels.astype(np.int32)
    label_counts = binary.sum(axis=0)
    rarity = np.zeros(binary.shape[1], dtype=np.float64)
    positive_label_mask = label_counts > 0
    rarity[positive_label_mask] = 1.0 / label_counts[positive_label_mask]
    sample_scores = binary @ rarity
    order = np.lexsort((rng.random(binary.shape[0]), -binary.sum(axis=1), -sample_scores))

    fold_indices: list[list[int]] = [[] for _ in range(n_splits)]
    fold_pos_counts = np.zeros((n_splits, binary.shape[1]), dtype=np.float64)
    fold_sizes = np.zeros(n_splits, dtype=np.float64)
    desired_pos_per_fold = label_counts / n_splits
    desired_size = binary.shape[0] / n_splits

    for index in order:
        positives = np.flatnonzero(binary[index] > 0)
        if len(positives) == 0:
            candidates = np.flatnonzero(fold_sizes == fold_sizes.min())
            chosen = int(rng.choice(candidates))
        else:
            label_deficit = desired_pos_per_fold[positives] - fold_pos_counts[:, positives]
            score = label_deficit.sum(axis=1) - 0.05 * (fold_sizes / desired_size)
            candidates = np.flatnonzero(score == score.max())
            if len(candidates) > 1:
                min_size = fold_sizes[candidates].min()
                candidates = candidates[fold_sizes[candidates] == min_size]
            chosen = int(rng.choice(candidates))

        fold_indices[chosen].append(int(index))
        fold_pos_counts[chosen] += binary[index]
        fold_sizes[chosen] += 1

    return [np.asarray(sorted(items), dtype=np.int64) for items in fold_indices]


def iter_batches(
    num_items: int,
    batch_size: int,
    rng: np.random.Generator | None,
) -> list[np.ndarray]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    indices = np.arange(num_items)
    if rng is not None:
        rng.shuffle(indices)
    return [
        indices[start : start + batch_size]
        for start in range(0, num_items, batch_size)
    ]


def tensor_from_numpy(array: np.ndarray, device: str, torch_module: Any) -> Any:
    return torch_module.from_numpy(array).to(device)


def train_one_epoch(
    *,
    model: Any,
    embeddings: Any,
    labels: Any,
    optimizer: Any,
    loss_fn: Any,
    batch_size: int,
    rng: np.random.Generator,
) -> float:
    import torch

    model.train()
    total_loss = 0.0
    total_seen = 0
    for batch_indices in iter_batches(labels.shape[0], batch_size, rng):
        index_tensor = torch.as_tensor(batch_indices, device=labels.device)
        batch_x = embeddings.index_select(0, index_tensor)
        batch_y = labels.index_select(0, index_tensor)

        optimizer.zero_grad(set_to_none=True)
        with torch.enable_grad():
            logits = model(batch_x)
            loss = loss_fn(logits, batch_y)
            loss.backward()
            optimizer.step()

        batch_size_seen = int(batch_y.shape[0])
        total_loss += float(loss.detach().cpu()) * batch_size_seen
        total_seen += batch_size_seen

    return total_loss / max(total_seen, 1)


def predict_logits(
    *,
    model: Any,
    embeddings: Any,
    batch_size: int,
) -> np.ndarray:
    import torch

    model.eval()
    logits: list[np.ndarray] = []
    with torch.no_grad():
        for batch_indices in iter_batches(embeddings.shape[0], batch_size, rng=None):
            index_tensor = torch.as_tensor(batch_indices, device=embeddings.device)
            batch_logits = model(embeddings.index_select(0, index_tensor))
            logits.append(batch_logits.detach().cpu().numpy())
    return np.concatenate(logits, axis=0)


def evaluate_loss(
    *,
    model: Any,
    embeddings: Any,
    labels: Any,
    loss_fn: Any,
    batch_size: int,
) -> float:
    import torch

    model.eval()
    total_loss = 0.0
    total_seen = 0
    with torch.no_grad():
        for batch_indices in iter_batches(labels.shape[0], batch_size, rng=None):
            index_tensor = torch.as_tensor(batch_indices, device=labels.device)
            batch_x = embeddings.index_select(0, index_tensor)
            batch_y = labels.index_select(0, index_tensor)
            loss = loss_fn(model(batch_x), batch_y)
            batch_size_seen = int(batch_y.shape[0])
            total_loss += float(loss.detach().cpu()) * batch_size_seen
            total_seen += batch_size_seen
    return total_loss / max(total_seen, 1)


def sigmoid_numpy(logits: np.ndarray) -> np.ndarray:
    logits64 = logits.astype(np.float64)
    return (1.0 / (1.0 + np.exp(-logits64))).astype(np.float32)


def safe_metric_summary(labels: np.ndarray, probs: np.ndarray) -> dict[str, Any]:
    from sklearn.metrics import average_precision_score, roc_auc_score

    per_label_roc: list[float | None] = []
    per_label_ap: list[float | None] = []
    for label_index in range(labels.shape[1]):
        y_true = labels[:, label_index]
        y_score = probs[:, label_index]
        if len(np.unique(y_true)) < 2:
            per_label_roc.append(None)
        else:
            per_label_roc.append(float(roc_auc_score(y_true, y_score)))

        if float(y_true.sum()) == 0.0:
            per_label_ap.append(None)
        else:
            per_label_ap.append(float(average_precision_score(y_true, y_score)))

    valid_roc = np.asarray([value for value in per_label_roc if value is not None], dtype=np.float64)
    valid_ap = np.asarray([value for value in per_label_ap if value is not None], dtype=np.float64)

    if len(np.unique(labels.reshape(-1))) < 2:
        micro_roc = None
    else:
        micro_roc = float(roc_auc_score(labels.reshape(-1), probs.reshape(-1)))

    if float(labels.sum()) == 0.0:
        micro_ap = None
    else:
        micro_ap = float(average_precision_score(labels.reshape(-1), probs.reshape(-1)))

    return {
        "macro_roc_auc": float(np.mean(valid_roc)) if valid_roc.size else None,
        "macro_average_precision": float(np.mean(valid_ap)) if valid_ap.size else None,
        "micro_roc_auc": micro_roc,
        "micro_average_precision": micro_ap,
        "valid_roc_auc_label_count": int(valid_roc.size),
        "valid_ap_label_count": int(valid_ap.size),
        "per_label_roc_auc": per_label_roc,
        "per_label_average_precision": per_label_ap,
    }


def train_head(
    *,
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    val_embeddings: np.ndarray | None,
    val_labels: np.ndarray | None,
    hidden_dim: int,
    dropout: float,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    pos_weight_max: float,
    patience: int,
    seed: int,
    device: str,
) -> tuple[Any, dict[str, Any]]:
    import torch
    from pom_repro.lm_head import FrozenLMHead

    if epochs <= 0:
        raise ValueError("epochs must be positive.")
    input_dim = int(train_embeddings.shape[1])
    output_dim = int(train_labels.shape[1])
    model = FrozenLMHead(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
    ).to(device)

    pos_weight = compute_pos_weight(train_labels, max_value=pos_weight_max)
    loss_fn = torch.nn.BCEWithLogitsLoss(
        pos_weight=tensor_from_numpy(pos_weight, device, torch)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    train_x = tensor_from_numpy(train_embeddings, device, torch)
    train_y = tensor_from_numpy(train_labels, device, torch)
    val_x = tensor_from_numpy(val_embeddings, device, torch) if val_embeddings is not None else None
    val_y = tensor_from_numpy(val_labels, device, torch) if val_labels is not None else None
    rng = np.random.default_rng(seed)

    best_state = {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }
    best_epoch = 0
    best_val_loss = math.inf
    stale_epochs = 0
    history: list[dict[str, float | int]] = []
    final_train_loss = math.inf

    for epoch in range(1, epochs + 1):
        final_train_loss = train_one_epoch(
            model=model,
            embeddings=train_x,
            labels=train_y,
            optimizer=optimizer,
            loss_fn=loss_fn,
            batch_size=batch_size,
            rng=rng,
        )
        if val_x is not None and val_y is not None:
            val_loss = evaluate_loss(
                model=model,
                embeddings=val_x,
                labels=val_y,
                loss_fn=loss_fn,
                batch_size=batch_size,
            )
        else:
            val_loss = final_train_loss

        history.append(
            {
                "epoch": epoch,
                "train_loss": float(final_train_loss),
                "val_loss": float(val_loss),
            }
        )

        if val_loss < best_val_loss:
            best_val_loss = float(val_loss)
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1

        if patience > 0 and val_x is not None and stale_epochs >= patience:
            break

    model.load_state_dict(best_state)
    return model, {
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val_loss),
        "final_train_loss": float(final_train_loss),
        "history": history,
        "pos_weight": pos_weight.tolist(),
    }


def run_cv(
    *,
    embeddings: np.ndarray,
    labels: np.ndarray,
    folds: int,
    hidden_dim: int,
    dropout: float,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    pos_weight_max: float,
    patience: int,
    seed: int,
    device: str,
) -> list[FoldResult]:
    all_folds = greedy_multilabel_kfold(labels, n_splits=folds, seed=seed)
    fold_results: list[FoldResult] = []

    for fold_number, val_indices in enumerate(all_folds, start=1):
        train_indices = np.setdiff1d(
            np.arange(labels.shape[0], dtype=np.int64),
            val_indices,
            assume_unique=True,
        )
        train_mean, train_std = compute_standardization(embeddings[train_indices])
        train_x = standardize(embeddings[train_indices], train_mean, train_std)
        val_x = standardize(embeddings[val_indices], train_mean, train_std)
        train_y = labels[train_indices]
        val_y = labels[val_indices]

        model, train_report = train_head(
            train_embeddings=train_x,
            train_labels=train_y,
            val_embeddings=val_x,
            val_labels=val_y,
            hidden_dim=hidden_dim,
            dropout=dropout,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            pos_weight_max=pos_weight_max,
            patience=patience,
            seed=seed + fold_number,
            device=device,
        )

        import torch

        val_logits = predict_logits(
            model=model,
            embeddings=tensor_from_numpy(val_x, device, torch),
            batch_size=batch_size,
        )
        metrics = safe_metric_summary(val_y, sigmoid_numpy(val_logits))
        metrics["val_loss"] = train_report["best_val_loss"]
        metrics["train_loss"] = train_report["final_train_loss"]

        fold_results.append(
            FoldResult(
                fold=fold_number,
                train_size=int(len(train_indices)),
                val_size=int(len(val_indices)),
                best_epoch=int(train_report["best_epoch"]),
                best_val_loss=float(train_report["best_val_loss"]),
                final_train_loss=float(train_report["final_train_loss"]),
                metrics=metrics,
            )
        )
        print(
            f"fold {fold_number}/{folds}: "
            f"macro_roc_auc={metrics['macro_roc_auc']} "
            f"macro_ap={metrics['macro_average_precision']} "
            f"best_epoch={train_report['best_epoch']}"
        )

    return fold_results


def summarize_cv(fold_results: list[FoldResult]) -> dict[str, Any]:
    metric_names = [
        "macro_roc_auc",
        "macro_average_precision",
        "micro_roc_auc",
        "micro_average_precision",
        "val_loss",
        "train_loss",
    ]
    summary: dict[str, Any] = {}
    for metric_name in metric_names:
        values = [
            result.metrics.get(metric_name)
            for result in fold_results
            if result.metrics.get(metric_name) is not None
        ]
        if not values:
            summary[metric_name] = {"mean": None, "std": None}
        else:
            array = np.asarray(values, dtype=np.float64)
            summary[metric_name] = {
                "mean": float(np.mean(array)),
                "std": float(np.std(array)),
            }
    return summary


def json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_safe(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def save_final_bundle(
    *,
    output_dir: Path,
    model: Any,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    label_names: list[str],
    config: dict[str, Any],
    metrics: dict[str, Any],
) -> None:
    import torch

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": config["input_dim"],
            "hidden_dim": config["hidden_dim"],
            "output_dim": config["output_dim"],
            "dropout": config["dropout"],
            "feature_mean": feature_mean.astype(np.float32),
            "feature_std": feature_std.astype(np.float32),
            "label_names": label_names,
            "config": config,
        },
        output_dir / "head.pt",
    )
    write_json(output_dir / "config.json", config)
    write_json(output_dir / "label_names.json", label_names)
    write_json(output_dir / "metrics.json", metrics)


def run(
    *,
    embeddings_path: str | Path,
    output_dir: str | Path,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    dropout: float,
    hidden_dim: int,
    folds: int,
    seed: int,
    device: str | None,
    pos_weight_max: float,
    patience: int,
    skip_cv: bool,
    command: str | None = None,
    data_path: str | Path | None = None,
    model_id: str | None = None,
    model_path: str | Path | None = None,
) -> int:
    import torch

    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if lr <= 0:
        raise ValueError("lr must be positive.")
    if weight_decay < 0:
        raise ValueError("weight_decay cannot be negative.")
    if patience < 0:
        raise ValueError("patience cannot be negative.")

    set_seed(seed)
    resolved_device = resolve_device(device, torch)
    embeddings_path = resolve_path(embeddings_path)
    output_dir = resolve_path(output_dir)
    resolved_data_path = resolve_path(data_path) if data_path is not None else None
    resolved_model_path = resolve_path(model_path) if model_path is not None else None
    embeddings, labels, label_names, source_metadata = load_embedding_npz(embeddings_path)

    cv_results: list[FoldResult] = []
    if not skip_cv:
        cv_results = run_cv(
            embeddings=embeddings,
            labels=labels,
            folds=folds,
            hidden_dim=hidden_dim,
            dropout=dropout,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            pos_weight_max=pos_weight_max,
            patience=patience,
            seed=seed,
            device=resolved_device,
        )

    feature_mean, feature_std = compute_standardization(embeddings)
    final_x = standardize(embeddings, feature_mean, feature_std)
    final_model, final_report = train_head(
        train_embeddings=final_x,
        train_labels=labels,
        val_embeddings=None,
        val_labels=None,
        hidden_dim=hidden_dim,
        dropout=dropout,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        pos_weight_max=pos_weight_max,
        patience=0,
        seed=seed + 10_000,
        device=resolved_device,
    )
    final_logits = predict_logits(
        model=final_model,
        embeddings=tensor_from_numpy(final_x, resolved_device, torch),
        batch_size=batch_size,
    )
    final_metrics = safe_metric_summary(labels, sigmoid_numpy(final_logits))
    final_metrics["train_loss"] = final_report["final_train_loss"]
    final_metrics["best_epoch"] = final_report["best_epoch"]

    config = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": command or format_command(),
        "git_commit": detect_git_commit(),
        "data_path": str(resolved_data_path) if resolved_data_path is not None else None,
        "embeddings_path": str(embeddings_path),
        "model_id": model_id or source_metadata["model_id"],
        "model_path": (
            str(resolved_model_path)
            if resolved_model_path is not None
            else source_metadata["model_path"]
        ),
        "num_samples": int(embeddings.shape[0]),
        "input_dim": int(embeddings.shape[1]),
        "output_dim": int(labels.shape[1]),
        "hidden_dim": int(hidden_dim),
        "dropout": float(dropout),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "lr": float(lr),
        "weight_decay": float(weight_decay),
        "folds": int(folds),
        "seed": int(seed),
        "device": resolved_device,
        "pos_weight_max": float(pos_weight_max),
        "patience": int(patience),
        "skip_cv": bool(skip_cv),
        "standardization": "zscore_fit_on_training_embeddings",
    }
    metrics = {
        "cv": {
            "folds": [asdict(result) for result in cv_results],
            "summary": summarize_cv(cv_results) if cv_results else {},
        },
        "final_train": final_metrics,
    }
    save_final_bundle(
        output_dir=output_dir,
        model=final_model,
        feature_mean=feature_mean,
        feature_std=feature_std,
        label_names=label_names,
        config=config,
        metrics=metrics,
    )

    print(f"Wrote {output_dir / 'head.pt'}")
    print(f"Wrote {output_dir / 'config.json'}")
    print(f"Wrote {output_dir / 'label_names.json'}")
    print(f"Wrote {output_dir / 'metrics.json'}")
    if cv_results:
        cv_summary = metrics["cv"]["summary"]
        print(
            "CV summary: "
            f"macro_roc_auc={cv_summary['macro_roc_auc']['mean']} "
            f"macro_ap={cv_summary['macro_average_precision']['mean']}"
        )
    print(
        "Final train metrics: "
        f"macro_roc_auc={final_metrics['macro_roc_auc']} "
        f"macro_ap={final_metrics['macro_average_precision']}"
    )
    return 0


def main() -> int:
    args = build_parser().parse_args()
    return run(
        embeddings_path=args.embeddings,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        hidden_dim=args.hidden_dim,
        folds=args.folds,
        seed=args.seed,
        device=args.device,
        pos_weight_max=args.pos_weight_max,
        patience=args.patience,
        skip_cv=args.skip_cv,
        command=format_command(),
        data_path=args.data,
        model_id=args.model_id,
        model_path=args.model_path,
    )


if __name__ == "__main__":
    raise SystemExit(main())
