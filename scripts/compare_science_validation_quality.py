#!/usr/bin/env python
"""Compare local inference against POM Science validation supplement data.

This script evaluates the current single-checkpoint OpenPOM-compatible
inference path on the subset of the Science POM prospective-validation
supplement for which we have:

- SMILES, from Data S7;
- human panel descriptor profiles, from Data S4;
- paper-provided GNN descriptor predictions, from Data S5.

It is intentionally a subset analysis, not a full reproduction of the original
400-molecule paper benchmark.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
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

DEFAULT_DATA_DIR = REPO_ROOT / "tests/science.ade4401_data_s1_to_s7"
DEFAULT_CHECKPOINT = REPO_ROOT / "checkpoints/openpom_experiments_1_checkpoint2.pt"
DEFAULT_JSON_REPORT = REPO_ROOT / "reports/science_validation_quality.json"
DEFAULT_SUMMARY = REPO_ROOT / "reports/science_validation_quality_summary.md"
DEFAULT_PER_SAMPLE = REPO_ROOT / "reports/science_validation_per_sample.csv"
DEFAULT_PER_LABEL = REPO_ROOT / "reports/science_validation_per_label.csv"
SCIENCE_LABEL_SPECIAL_CASES = {
    "jasmine": "jasmin",
}


@dataclass(frozen=True)
class PairMetrics:
    pearson: float
    spearman: float
    cosine: float
    mae: float
    rmse: float


@dataclass(frozen=True)
class TopKMetrics:
    k: int
    mean_overlap_count: float
    median_overlap_count: float
    mean_overlap_fraction: float
    exact_order_match_fraction: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare current OpenPOM-compatible inference against Science "
            "POM validation supplement data on the available SMILES subset."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Directory containing Science Data S1-S7 CSV files. Default: {DEFAULT_DATA_DIR}",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help=f"Single checkpoint to evaluate. Default: {DEFAULT_CHECKPOINT}",
    )
    parser.add_argument(
        "--json-report",
        type=Path,
        default=DEFAULT_JSON_REPORT,
        help=f"JSON report path. Default: {DEFAULT_JSON_REPORT}",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=DEFAULT_SUMMARY,
        help=f"Markdown summary path. Default: {DEFAULT_SUMMARY}",
    )
    parser.add_argument(
        "--per-sample-csv",
        type=Path,
        default=DEFAULT_PER_SAMPLE,
        help=f"Per-sample CSV path. Default: {DEFAULT_PER_SAMPLE}",
    )
    parser.add_argument(
        "--per-label-csv",
        type=Path,
        default=DEFAULT_PER_LABEL,
        help=f"Per-label CSV path. Default: {DEFAULT_PER_LABEL}",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device for inference. Default: cpu.",
    )
    return parser.parse_args()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing CSV: {path}")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def read_csv_header(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing CSV: {path}")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return next(csv.reader(handle))


def by_sample(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        sample = row.get("Sample", "").strip()
        if sample:
            result[sample] = row
    return result


def science_label_to_openpom(label: str, openpom_labels: set[str]) -> str:
    normalized = label.strip().lower()
    normalized = SCIENCE_LABEL_SPECIAL_CASES.get(normalized, normalized)
    if normalized not in openpom_labels:
        raise KeyError(f"Science label {label!r} does not map to OpenPOM labels.")
    return normalized


def build_label_mapping(science_labels: list[str]) -> list[dict[str, Any]]:
    from pom_repro.predict import ODOR_LABELS

    openpom_label_set = set(ODOR_LABELS)
    mapping: list[dict[str, Any]] = []
    for science_label in science_labels:
        openpom_label = science_label_to_openpom(science_label, openpom_label_set)
        mapping.append(
            {
                "science_label": science_label,
                "openpom_label": openpom_label,
                "openpom_index": int(ODOR_LABELS.index(openpom_label)),
            }
        )
    return mapping


def matrix_for_samples(
    rows_by_sample: dict[str, dict[str, str]],
    samples: list[str],
    labels: list[str],
) -> np.ndarray:
    values = []
    for sample in samples:
        row = rows_by_sample[sample]
        values.append([float(row[label]) for label in labels])
    return np.asarray(values, dtype=np.float64)


def rankdata_average_ties(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        average_rank = 0.5 * (start + 1 + end)
        ranks[order[start:end]] = average_rank
        start = end
    return ranks


def finite_pair(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch: {left.shape} vs {right.shape}")
    mask = np.isfinite(left) & np.isfinite(right)
    return left[mask], right[mask]


def pearson(left: np.ndarray, right: np.ndarray) -> float:
    left, right = finite_pair(left, right)
    if len(left) < 2:
        return float("nan")
    left_centered = left - np.mean(left)
    right_centered = right - np.mean(right)
    denominator = np.linalg.norm(left_centered) * np.linalg.norm(right_centered)
    if denominator == 0:
        return float("nan")
    return float(np.dot(left_centered, right_centered) / denominator)


def spearman(left: np.ndarray, right: np.ndarray) -> float:
    left, right = finite_pair(left, right)
    if len(left) < 2:
        return float("nan")
    return pearson(rankdata_average_ties(left), rankdata_average_ties(right))


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    left, right = finite_pair(left, right)
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    if denominator == 0:
        return float("nan")
    return float(np.dot(left, right) / denominator)


def pair_metrics(left: np.ndarray, right: np.ndarray) -> PairMetrics:
    left_flat, right_flat = finite_pair(left, right)
    diff = left_flat - right_flat
    return PairMetrics(
        pearson=pearson(left_flat, right_flat),
        spearman=spearman(left_flat, right_flat),
        cosine=cosine(left_flat, right_flat),
        mae=float(np.mean(np.abs(diff))) if len(diff) else float("nan"),
        rmse=float(np.sqrt(np.mean(diff * diff))) if len(diff) else float("nan"),
    )


def summary_stats(values: list[float] | np.ndarray) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return {
            "count": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "q1": float("nan"),
            "q3": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }
    return {
        "count": int(len(array)),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "q1": float(np.quantile(array, 0.25)),
        "q3": float(np.quantile(array, 0.75)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def top_k_indices(matrix: np.ndarray, k: int) -> np.ndarray:
    if k <= 0:
        raise ValueError("k must be positive")
    if k > matrix.shape[1]:
        raise ValueError(f"k={k} exceeds number of labels={matrix.shape[1]}")
    return np.argsort(-matrix, axis=1, kind="stable")[:, :k]


def top_k_metrics(left: np.ndarray, right: np.ndarray, k: int) -> TopKMetrics:
    left_top = top_k_indices(left, k)
    right_top = top_k_indices(right, k)
    overlaps: list[int] = []
    order_matches: list[bool] = []
    for left_indices, right_indices in zip(left_top, right_top, strict=True):
        left_list = [int(index) for index in left_indices]
        right_list = [int(index) for index in right_indices]
        overlaps.append(len(set(left_list) & set(right_list)))
        order_matches.append(left_list == right_list)
    return TopKMetrics(
        k=k,
        mean_overlap_count=float(np.mean(overlaps)),
        median_overlap_count=float(np.median(overlaps)),
        mean_overlap_fraction=float(np.mean(overlaps) / k),
        exact_order_match_fraction=float(np.mean(order_matches)),
    )


def predict_science_labels(
    smiles: list[str],
    checkpoint: Path,
    label_mapping: list[dict[str, Any]],
    device: str,
) -> np.ndarray:
    from pom_repro.predict import predict_smiles

    result = predict_smiles(
        smiles=smiles,
        checkpoint_paths=[str(checkpoint)],
        top_k=10,
        device=device,
        return_embedding=False,
    )
    probs = np.asarray(result["probs"], dtype=np.float64)
    label_indices = [int(item["openpom_index"]) for item in label_mapping]
    return probs[:, label_indices]


def nan_safe(value: Any) -> Any:
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, dict):
        return {key: nan_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [nan_safe(item) for item in value]
    return value


def subset_metrics(
    subset_name: str,
    samples: list[str],
    smiles_by_sample: dict[str, str],
    human: np.ndarray,
    paper: np.ndarray,
    ours: np.ndarray,
    labels: list[str],
    s1_by_sample: dict[str, dict[str, str]],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    pairs = {
        "ours_vs_human": (ours, human),
        "paper_gnn_vs_human": (paper, human),
        "ours_vs_paper_gnn": (ours, paper),
    }

    global_metrics = {
        name: asdict(pair_metrics(left, right))
        for name, (left, right) in pairs.items()
    }
    top_k = {
        name: {
            "top5": asdict(top_k_metrics(left, right, k=5)),
            "top10": asdict(top_k_metrics(left, right, k=10)),
        }
        for name, (left, right) in pairs.items()
    }

    per_sample_rows: list[dict[str, Any]] = []
    per_sample_values: dict[str, dict[str, list[float]]] = {
        name: {"pearson": [], "spearman": [], "cosine": []}
        for name in pairs
    }
    for row_index, sample in enumerate(samples):
        sample_row: dict[str, Any] = {
            "subset": subset_name,
            "sample": sample,
            "smiles": smiles_by_sample[sample],
            "disqualification_reason": s1_by_sample.get(sample, {}).get(
                "Disqualification reason", ""
            ),
        }
        for name, (left, right) in pairs.items():
            metrics = pair_metrics(left[row_index], right[row_index])
            sample_row[f"{name}_pearson"] = metrics.pearson
            sample_row[f"{name}_spearman"] = metrics.spearman
            sample_row[f"{name}_cosine"] = metrics.cosine
            sample_row[f"{name}_mae"] = metrics.mae
            for key in ["pearson", "spearman", "cosine"]:
                per_sample_values[name][key].append(getattr(metrics, key))
        per_sample_rows.append(sample_row)

    per_label_rows: list[dict[str, Any]] = []
    per_label_values: dict[str, dict[str, list[float]]] = {
        name: {"pearson": [], "spearman": [], "cosine": []}
        for name in pairs
    }
    for label_index, science_label in enumerate(labels):
        label_row: dict[str, Any] = {
            "subset": subset_name,
            "science_label": science_label,
        }
        for name, (left, right) in pairs.items():
            metrics = pair_metrics(left[:, label_index], right[:, label_index])
            label_row[f"{name}_pearson"] = metrics.pearson
            label_row[f"{name}_spearman"] = metrics.spearman
            label_row[f"{name}_cosine"] = metrics.cosine
            label_row[f"{name}_mae"] = metrics.mae
            for key in ["pearson", "spearman", "cosine"]:
                per_label_values[name][key].append(getattr(metrics, key))
        per_label_rows.append(label_row)

    metrics = {
        "num_samples": len(samples),
        "num_labels": len(labels),
        "global": global_metrics,
        "top_k": top_k,
        "per_sample_summary": {
            name: {key: summary_stats(values) for key, values in metric_values.items()}
            for name, metric_values in per_sample_values.items()
        },
        "per_label_summary": {
            name: {key: summary_stats(values) for key, values in metric_values.items()}
            for name, metric_values in per_label_values.items()
        },
    }
    return metrics, per_sample_rows, per_label_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([{key: nan_safe(value) for key, value in row.items()} for row in rows])


def fmt(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{value:.4f}"


def render_summary(report: dict[str, Any]) -> str:
    lines = [
        "# Science POM Validation Subset Quality Report",
        "",
        "## Scope",
        "",
        "This report uses only the supplement rows that currently have SMILES,",
        "human panel descriptor profiles, and paper-provided GNN prediction",
        "profiles. It is not a full 400-molecule Science benchmark and it is",
        "not a 10-model ensemble evaluation.",
        "",
        "Current model under test:",
        "",
        f"- checkpoint: `{report['metadata']['checkpoint']}`",
        "- inference: local OpenPOM-compatible single-checkpoint model",
        "- label space compared: 55 Science validation descriptors mapped into the 138 OpenPOM labels",
        "",
        "## Dataset Counts",
        "",
    ]
    counts = report["metadata"]["counts"]
    for key, value in counts.items():
        lines.append(f"- {key}: {value}")

    lines.extend(["", "## Main Metrics", ""])
    for subset_name, subset in report["subsets"].items():
        lines.extend([f"### {subset_name}", ""])
        lines.append(f"Samples: {subset['num_samples']}; labels: {subset['num_labels']}")
        lines.append("")
        lines.append("| Pair | Global Pearson | Global Spearman | Global Cosine | Top-5 overlap | Top-10 overlap | Median per-sample Pearson |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
        for pair_name in ["ours_vs_human", "paper_gnn_vs_human", "ours_vs_paper_gnn"]:
            global_metrics = subset["global"][pair_name]
            top5 = subset["top_k"][pair_name]["top5"]["mean_overlap_fraction"]
            top10 = subset["top_k"][pair_name]["top10"]["mean_overlap_fraction"]
            sample_median = subset["per_sample_summary"][pair_name]["pearson"]["median"]
            lines.append(
                "| "
                f"{pair_name} | "
                f"{fmt(global_metrics['pearson'])} | "
                f"{fmt(global_metrics['spearman'])} | "
                f"{fmt(global_metrics['cosine'])} | "
                f"{fmt(top5)} | "
                f"{fmt(top10)} | "
                f"{fmt(sample_median)} |"
            )
        lines.append("")

    lines.extend(
        [
            "## Interpretation Notes",
            "",
            "- `ours_vs_human` is the current OpenPOM single-checkpoint inference quality on the available supplement subset.",
            "- `paper_gnn_vs_human` is the paper-provided GNN prediction profile against the same human panel profiles, useful as a reference baseline.",
            "- `ours_vs_paper_gnn` shows how similar our OpenPOM checkpoint behavior is to the paper-provided GNN prediction profiles on this subset.",
            "- Differences can come from using OpenPOM rather than the original closed Science model, single checkpoint rather than ensemble, training set differences, label normalization, and the subset restriction to rows with SMILES.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    checkpoint = args.checkpoint.resolve()

    s1_rows = read_csv_rows(data_dir / "science.ade4401_data_s1.csv")
    s4_rows = read_csv_rows(data_dir / "science.ade4401_data_s4.csv")
    s5_rows = read_csv_rows(data_dir / "science.ade4401_data_s5.csv")
    s7_rows = read_csv_rows(data_dir / "science.ade4401_data_s7.csv")

    s1_by_sample = by_sample(s1_rows)
    s4_by_sample = by_sample(s4_rows)
    s5_by_sample = by_sample(s5_rows)
    s7_by_sample = by_sample(s7_rows)
    smiles_by_sample = {
        sample: row["SMILES"].strip()
        for sample, row in s7_by_sample.items()
        if row.get("SMILES", "").strip()
    }

    science_labels = read_csv_header(data_dir / "science.ade4401_data_s4.csv")[1:]
    label_mapping = build_label_mapping(science_labels)

    s4_samples = set(s4_by_sample)
    s5_samples = set(s5_by_sample)
    s7_samples = set(smiles_by_sample)
    s1_samples = set(s1_by_sample)
    clean_samples = {
        sample
        for sample, row in s1_by_sample.items()
        if not row.get("Disqualification reason", "").strip()
    }
    all_overlap = sorted(s4_samples & s5_samples & s7_samples)
    clean_overlap = sorted(set(all_overlap) & clean_samples)

    subset_definitions = {
        "all_overlap": all_overlap,
        "clean_overlap": clean_overlap,
    }

    all_samples_for_inference = sorted(set(all_overlap) | set(clean_overlap))
    all_smiles_for_inference = [smiles_by_sample[sample] for sample in all_samples_for_inference]
    all_predictions = predict_science_labels(
        smiles=all_smiles_for_inference,
        checkpoint=checkpoint,
        label_mapping=label_mapping,
        device=args.device,
    )
    prediction_by_sample = {
        sample: all_predictions[index]
        for index, sample in enumerate(all_samples_for_inference)
    }

    subsets: dict[str, Any] = {}
    per_sample_rows: list[dict[str, Any]] = []
    per_label_rows: list[dict[str, Any]] = []
    for subset_name, samples in subset_definitions.items():
        human = matrix_for_samples(s4_by_sample, samples, science_labels)
        paper = matrix_for_samples(s5_by_sample, samples, science_labels)
        ours = np.asarray([prediction_by_sample[sample] for sample in samples])

        if human.shape != paper.shape or human.shape != ours.shape:
            raise RuntimeError(
                f"Shape mismatch for {subset_name}: human={human.shape}, "
                f"paper={paper.shape}, ours={ours.shape}"
            )
        if not np.isfinite(human).all() or not np.isfinite(paper).all() or not np.isfinite(ours).all():
            raise RuntimeError(f"Non-finite values found in subset {subset_name}")

        subset_report, subset_sample_rows, subset_label_rows = subset_metrics(
            subset_name=subset_name,
            samples=samples,
            smiles_by_sample=smiles_by_sample,
            human=human,
            paper=paper,
            ours=ours,
            labels=science_labels,
            s1_by_sample=s1_by_sample,
        )
        subsets[subset_name] = subset_report
        per_sample_rows.extend(subset_sample_rows)
        per_label_rows.extend(subset_label_rows)

    report = {
        "metadata": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "data_dir": str(data_dir),
            "checkpoint": str(checkpoint),
            "device": args.device,
            "counts": {
                "data_s1_samples": len(s1_samples),
                "data_s4_samples": len(s4_samples),
                "data_s5_samples": len(s5_samples),
                "data_s7_smiles_samples": len(s7_samples),
                "all_overlap_samples": len(all_overlap),
                "clean_overlap_samples": len(clean_overlap),
                "science_labels": len(science_labels),
                "mapped_labels": len(label_mapping),
            },
            "label_mapping": label_mapping,
            "limitations": [
                "Subset analysis only: not all Science validation samples have SMILES in Data S7.",
                "Current evaluation uses one OpenPOM checkpoint, not the paper's original closed model.",
                "Current evaluation is not a 10-model ensemble because only one real checkpoint is present locally.",
            ],
        },
        "subsets": subsets,
    }

    args.json_report.parent.mkdir(parents=True, exist_ok=True)
    args.json_report.write_text(
        json.dumps(nan_safe(report), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_csv(args.per_sample_csv, per_sample_rows)
    write_csv(args.per_label_csv, per_label_rows)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(render_summary(nan_safe(report)), encoding="utf-8")

    print(f"Wrote {args.json_report}")
    print(f"Wrote {args.summary}")
    print(f"Wrote {args.per_sample_csv}")
    print(f"Wrote {args.per_label_csv}")
    for subset_name, subset in subsets.items():
        ours = subset["global"]["ours_vs_human"]
        paper = subset["global"]["paper_gnn_vs_human"]
        print(
            f"{subset_name}: n={subset['num_samples']} "
            f"ours_vs_human Pearson={ours['pearson']:.4f} "
            f"Spearman={ours['spearman']:.4f}; "
            f"paper_vs_human Pearson={paper['pearson']:.4f} "
            f"Spearman={paper['spearman']:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
