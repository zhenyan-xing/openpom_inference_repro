#!/usr/bin/env python
"""Predict OpenPOM odor labels for one or more SMILES strings."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

ENV_CHECKPOINTS = "POM_REPRO_ENSEMBLE_CHECKPOINTS"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Predict odor probabilities from SMILES with OpenPOM checkpoints."
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
        "--checkpoint",
        action="append",
        default=None,
        help=(
            "Checkpoint path. Repeat for an ensemble. If omitted, reads "
            f"{ENV_CHECKPOINTS} using the OS path separator."
        ),
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of top odor labels to return. Default: 10.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device for inference. Default: cpu.",
    )
    parser.add_argument(
        "--no-embedding",
        action="store_true",
        help="Do not return averaged embeddings in the JSON output.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path. JSON is also written to stdout.",
    )
    return parser


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


def load_checkpoints(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> list[str]:
    if args.checkpoint is not None:
        checkpoints = [item.strip() for item in args.checkpoint if item.strip()]
    else:
        raw_env = os.environ.get(ENV_CHECKPOINTS, "")
        checkpoints = [item.strip() for item in raw_env.split(os.pathsep) if item.strip()]

    if not checkpoints:
        parser.error(
            "Provide at least one --checkpoint or set "
            f"{ENV_CHECKPOINTS}."
        )
    return checkpoints


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


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    smiles = load_smiles(args, parser)
    checkpoints = load_checkpoints(args, parser)

    from pom_repro.predict import predict_smiles

    result = predict_smiles(
        smiles=smiles,
        checkpoint_paths=checkpoints,
        top_k=args.top_k,
        device=args.device,
        return_embedding=not args.no_embedding,
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
