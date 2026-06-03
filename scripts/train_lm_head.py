#!/usr/bin/env python
"""Compatibility entrypoint for training the frozen LM MLP head."""

from __future__ import annotations

import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from train_frozen_lm_head import main


if __name__ == "__main__":
    raise SystemExit(main())
