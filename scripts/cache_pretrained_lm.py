#!/usr/bin/env python
"""Cache a HuggingFace chemical language model for offline embedding export."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_MODEL_ID = "ibm-research/MoLFormer-XL-both-10pct"
DEFAULT_OUTPUT_DIR = Path("models/MoLFormer-XL-both-10pct")
MANIFEST_NAME = "cache_manifest.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download and cache a pretrained HuggingFace chemical LM."
    )
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help=f"HuggingFace model id. Default: {DEFAULT_MODEL_ID}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Local directory for the cached tokenizer/model. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow custom model code from HuggingFace. Default: true.",
    )
    parser.add_argument(
        "--deterministic-eval",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pass deterministic_eval to AutoModel.from_pretrained. Default: true.",
    )
    return parser


def write_manifest(
    *,
    model_id: str,
    output_dir: Path,
    transformers_version: str,
    torch_version: str,
) -> Path:
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": model_id,
        "output_dir": str(output_dir),
        "torch_version": torch_version,
        "transformers_version": transformers_version,
    }
    manifest_path = output_dir / MANIFEST_NAME
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def run(
    *,
    model_id: str,
    output_dir: Path,
    trust_remote_code: bool,
    deterministic_eval: bool,
) -> int:
    from transformers import AutoModel, AutoTokenizer
    import torch
    import transformers

    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=trust_remote_code,
    )
    model = AutoModel.from_pretrained(
        model_id,
        trust_remote_code=trust_remote_code,
        deterministic_eval=deterministic_eval,
    )

    tokenizer.save_pretrained(output_dir)
    model.save_pretrained(output_dir)
    manifest_path = write_manifest(
        model_id=model_id,
        output_dir=output_dir,
        transformers_version=transformers.__version__,
        torch_version=torch.__version__,
    )

    print(f"Saved tokenizer and model to {output_dir}")
    print(f"Wrote manifest to {manifest_path}")
    return 0


def main() -> int:
    args = build_parser().parse_args()
    return run(
        model_id=args.model_id,
        output_dir=args.output_dir,
        trust_remote_code=args.trust_remote_code,
        deterministic_eval=args.deterministic_eval,
    )


if __name__ == "__main__":
    raise SystemExit(main())
