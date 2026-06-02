# Phase 6: Strict Checkpoint Loading

## Purpose

Phase 6 formalized checkpoint loading for the local OpenPOM-compatible model.
The goal was to make checkpoint compatibility an explicit, testable contract
instead of relying on ad hoc `load_state_dict` calls.

## What Was Added

New implementation:

```text
src/pom_repro/checkpoint.py
```

New tests:

```text
tests/test_checkpoint_load.py
```

Generated report:

```text
reports/checkpoint_load_report.txt
```

Compatibility note:

```text
docs/checkpoint_compatibility.md
```

## Loader Behavior

The public loading API is:

```python
from pom_repro.checkpoint import load_checkpoint_strict

report = load_checkpoint_strict(model, checkpoint_path)
```

The loader:

- loads checkpoints with `torch.load(..., map_location="cpu")`;
- passes `weights_only=False` when the installed PyTorch version supports it;
- extracts weights from `checkpoint["model_state_dict"]`;
- tries `model.load_state_dict(state_dict, strict=True)` first;
- reports missing keys, unexpected keys, and tensor shape mismatches;
- never uses `strict=False`;
- only invokes an optional key remapper after direct strict loading fails.

For the current target checkpoint, no key remapping is needed.

## Target Checkpoint

The Phase 6 source of truth is the real local checkpoint copy:

```text
checkpoints/openpom_experiments_1_checkpoint2.pt
```

The Git LFS pointer files under `/home/xing/openpom/models/ensemble_models`
are not used as model weights.

## Validation

Strict loading result:

```text
model_key_count = 44
checkpoint_key_count = 44
missing_keys = []
unexpected_keys = []
shape_mismatches = []
direct_strict_success = True
remap_used = False
```

Commands:

```bash
PYTHONPATH=src /home/xing/miniconda3/envs/pom_openpom_ref/bin/python \
  -m pytest tests/test_checkpoint_load.py -q

PYTHONPATH=src /home/xing/miniconda3/envs/pom_openpom_ref/bin/python \
  -m pytest tests/test_model_shapes.py tests/test_featurizer.py -q

PYTHONPATH=src /home/xing/miniconda3/envs/pom_openpom_ref/bin/python \
  -m pom_repro.checkpoint \
  --checkpoint checkpoints/openpom_experiments_1_checkpoint2.pt \
  --report reports/checkpoint_load_report.txt
```

Observed checkpoint test result:

```text
5 passed
```

Observed model and featurizer regression result:

```text
8 passed
```

The strict-loaded model also produced finite outputs for `["C", "CCO"]`:

```text
proba:      [2, 138]
logits:     [2, 138, 1]
embeddings: [2, 256]
```

## Notes For Phase 7

Phase 7 can assume the local architecture and primary checkpoint are key- and
shape-compatible. If output parity fails, debug graph features, atom+bond
readout, Set2Set behavior, or FFN behavior before changing checkpoint loading.
