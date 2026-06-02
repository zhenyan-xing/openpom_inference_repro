"""Strict OpenPOM checkpoint loading utilities."""

from __future__ import annotations

import argparse
import inspect
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn


DEFAULT_CHECKPOINT = Path("checkpoints/openpom_experiments_1_checkpoint2.pt")
DEFAULT_REPORT = Path("reports/checkpoint_load_report.txt")


@dataclass(frozen=True)
class ShapeMismatch:
    """Shape difference for a key present in both model and checkpoint."""

    key: str
    model_shape: tuple[int, ...] | str
    checkpoint_shape: tuple[int, ...] | str


@dataclass
class CheckpointLoadReport:
    """Structured result for a strict checkpoint load attempt."""

    checkpoint_path: str
    state_dict_source: str
    model_key_count: int
    checkpoint_key_count: int
    direct_strict_success: bool = False
    strict_load_success: bool = False
    remap_used: bool = False
    missing_keys: list[str] = field(default_factory=list)
    unexpected_keys: list[str] = field(default_factory=list)
    shape_mismatches: list[ShapeMismatch] = field(default_factory=list)
    error_message: str | None = None


class CheckpointLoadError(RuntimeError):
    """Raised when strict checkpoint loading fails."""

    def __init__(self, message: str, report: CheckpointLoadReport) -> None:
        super().__init__(message)
        self.report = report


KeyRemapper = Callable[[Mapping[str, Any]], Mapping[str, Any]]


def torch_load_checkpoint(path: Path, map_location: str | torch.device = "cpu") -> Any:
    """Load a checkpoint with a PyTorch-version-compatible signature."""

    kwargs: dict[str, Any] = {"map_location": map_location}
    if "weights_only" in inspect.signature(torch.load).parameters:
        kwargs["weights_only"] = False
    return torch.load(path, **kwargs)


def is_tensor_state_dict(value: Any) -> bool:
    """Return True for direct state-dict-like mappings."""

    if not isinstance(value, Mapping) or not value:
        return False
    return all(isinstance(key, str) for key in value.keys()) and any(
        torch.is_tensor(item) for item in value.values()
    )


def extract_state_dict(checkpoint: Any) -> tuple[Mapping[str, Any], str]:
    """Extract model weights from an OpenPOM checkpoint object."""

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


def _shape_of(value: Any) -> tuple[int, ...] | str:
    if torch.is_tensor(value):
        return tuple(int(dim) for dim in value.shape)
    return f"<non_tensor:{type(value).__name__}>"


def inspect_state_dict_compatibility(
    model: nn.Module,
    state_dict: Mapping[str, Any],
    checkpoint_path: str | Path = "<memory>",
    state_dict_source: str = "unknown",
) -> CheckpointLoadReport:
    """Compare model and checkpoint keys without mutating model weights."""

    model_state = model.state_dict()
    model_keys = set(model_state.keys())
    checkpoint_keys = set(state_dict.keys())

    shape_mismatches = [
        ShapeMismatch(
            key=key,
            model_shape=_shape_of(model_state[key]),
            checkpoint_shape=_shape_of(state_dict[key]),
        )
        for key in sorted(model_keys & checkpoint_keys)
        if _shape_of(model_state[key]) != _shape_of(state_dict[key])
    ]

    return CheckpointLoadReport(
        checkpoint_path=str(checkpoint_path),
        state_dict_source=state_dict_source,
        model_key_count=len(model_state),
        checkpoint_key_count=len(state_dict),
        missing_keys=sorted(model_keys - checkpoint_keys),
        unexpected_keys=sorted(checkpoint_keys - model_keys),
        shape_mismatches=shape_mismatches,
    )


def _ensure_mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{context} did not return a mapping")
    return value


def _raise_load_error(
    message: str,
    report: CheckpointLoadReport,
    error: BaseException | None = None,
) -> None:
    if error is not None:
        report.error_message = str(error)
    raise CheckpointLoadError(message, report) from error


def load_checkpoint_strict(
    model: nn.Module,
    checkpoint_path: str | Path,
    map_location: str | torch.device = "cpu",
    key_remapper: KeyRemapper | None = None,
) -> CheckpointLoadReport:
    """Load an OpenPOM checkpoint with strict key and tensor-shape validation."""

    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch_load_checkpoint(checkpoint_path, map_location=map_location)
    state_dict, state_source = extract_state_dict(checkpoint)

    report = inspect_state_dict_compatibility(
        model=model,
        state_dict=state_dict,
        checkpoint_path=checkpoint_path,
        state_dict_source=state_source,
    )

    try:
        incompatible = model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        if key_remapper is None:
            _raise_load_error(
                "Direct strict checkpoint loading failed.",
                report,
                exc,
            )
    else:
        report.direct_strict_success = True
        report.strict_load_success = True
        report.missing_keys = list(incompatible.missing_keys)
        report.unexpected_keys = list(incompatible.unexpected_keys)
        return report

    remapped_state_dict = _ensure_mapping(key_remapper(state_dict), "key_remapper")
    remapped_report = inspect_state_dict_compatibility(
        model=model,
        state_dict=remapped_state_dict,
        checkpoint_path=checkpoint_path,
        state_dict_source=f"{state_source}:remapped",
    )
    remapped_report.remap_used = True

    try:
        incompatible = model.load_state_dict(remapped_state_dict, strict=True)
    except RuntimeError as exc:
        _raise_load_error(
            "Strict checkpoint loading failed after key remapping.",
            remapped_report,
            exc,
        )

    remapped_report.strict_load_success = True
    remapped_report.missing_keys = list(incompatible.missing_keys)
    remapped_report.unexpected_keys = list(incompatible.unexpected_keys)
    return remapped_report


def format_checkpoint_load_report(report: CheckpointLoadReport) -> str:
    """Render a strict load report for humans and future agents."""

    lines = [
        "# Phase 6 Checkpoint Load Report",
        "",
        f"checkpoint_path: {report.checkpoint_path}",
        f"state_dict_source: {report.state_dict_source}",
        f"model_key_count: {report.model_key_count}",
        f"checkpoint_key_count: {report.checkpoint_key_count}",
        f"direct_strict_success: {report.direct_strict_success}",
        f"strict_load_success: {report.strict_load_success}",
        f"remap_used: {report.remap_used}",
        "",
        "## Compatibility",
        f"missing_key_count: {len(report.missing_keys)}",
        f"unexpected_key_count: {len(report.unexpected_keys)}",
        f"shape_mismatch_count: {len(report.shape_mismatches)}",
        "",
        "missing_keys:",
    ]

    lines.extend(f"  - {key}" for key in report.missing_keys)
    if not report.missing_keys:
        lines.append("  - <none>")

    lines.extend(["", "unexpected_keys:"])
    lines.extend(f"  - {key}" for key in report.unexpected_keys)
    if not report.unexpected_keys:
        lines.append("  - <none>")

    lines.extend(["", "shape_mismatches:"])
    lines.extend(
        "  - "
        f"{mismatch.key}: model={mismatch.model_shape} "
        f"checkpoint={mismatch.checkpoint_shape}"
        for mismatch in report.shape_mismatches
    )
    if not report.shape_mismatches:
        lines.append("  - <none>")

    lines.extend(["", "error_message:"])
    lines.append(f"  {report.error_message or '<none>'}")

    compatible = (
        report.strict_load_success
        and not report.missing_keys
        and not report.unexpected_keys
        and not report.shape_mismatches
    )
    if compatible:
        if report.remap_used:
            notes = "Strict loading succeeded after explicit key remapping."
        else:
            notes = "Direct strict loading succeeded. No key remapping was used."
        lines.extend(["", "result: PASS", f"notes: {notes}"])
    else:
        lines.extend(["", "result: FAIL"])

    lines.append("")
    return "\n".join(lines)


def write_checkpoint_load_report(
    report: CheckpointLoadReport,
    path: str | Path = DEFAULT_REPORT,
) -> Path:
    """Write a strict checkpoint load report to disk."""

    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(format_checkpoint_load_report(report), encoding="utf-8")
    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Strictly load the OpenPOM checkpoint into the local model."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help=f"Checkpoint path. Default: {DEFAULT_CHECKPOINT}",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT,
        help=f"Report path. Default: {DEFAULT_REPORT}",
    )
    parser.add_argument(
        "--map-location",
        default="cpu",
        help="torch.load map_location. Default: cpu",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from pom_repro.model import MPNNPOM

    model = MPNNPOM()
    try:
        report = load_checkpoint_strict(
            model=model,
            checkpoint_path=args.checkpoint,
            map_location=args.map_location,
        )
    except CheckpointLoadError as exc:
        write_checkpoint_load_report(exc.report, args.report)
        print(format_checkpoint_load_report(exc.report), end="")
        raise SystemExit(1) from exc

    write_checkpoint_load_report(report, args.report)
    print(format_checkpoint_load_report(report), end="")


if __name__ == "__main__":
    main()
