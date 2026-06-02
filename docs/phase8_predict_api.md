# Phase 8: Predict API And Ensemble

## Purpose

Phase 8 added the user-facing prediction layer after Phase 7 established
single-checkpoint parity. The goal was to expose a stable API and CLI for
predicting top-k OpenPOM odor labels from SMILES, while preserving the same
local inference path:

```text
GraphFeaturizer -> batch_graphs -> MPNNPOM -> load_checkpoint_strict
```

## What Was Added

New prediction API:

```text
src/pom_repro/predict.py
```

New command-line wrapper:

```text
scripts/predict_smiles.py
```

New tests:

```text
tests/test_predict_smiles.py
```

New user documentation:

```text
README.md
```

The package root also now exports:

```python
from pom_repro import predict_smiles
```

## API Contract

The public function is:

```python
def predict_smiles(
    smiles: list[str],
    checkpoint_paths: list[str],
    top_k: int = 10,
    device: str = "cpu",
    return_embedding: bool = True,
) -> dict[str, Any]:
    ...
```

Return keys:

```text
smiles:           original SMILES list
labels:           138 OpenPOM odor labels in checkpoint/reference order
probs:            averaged probability array, [num_smiles, 138]
top_k:            per-SMILES sorted top-k label/probability records
checkpoint_paths: resolved real checkpoint paths
embedding:        averaged embedding array, [num_smiles, 256], or None
```

The odor label order is embedded in `pom_repro.predict.ODOR_LABELS` and matches
the OpenPOM task order used by the Phase 1 reference export.

## Ensemble Behavior

For each checkpoint, Phase 8 runs the checkpoint-compatible model in eval mode
with gradients disabled. It stacks and averages checkpoint outputs as:

```text
all_probs:      [num_checkpoints, num_smiles, 138]
probs:          mean(all_probs, axis=0)
all_embeddings: [num_checkpoints, num_smiles, 256]
embedding:      mean(all_embeddings, axis=0)
```

This follows OpenPOM's user-facing `predict_odors.py` behavior for
probabilities. Embeddings are not exposed by the OpenPOM demo, so this
reproduction returns averaged embeddings only as an optional convenience when
`return_embedding=True`.

## Checkpoint Handling

Real checkpoint files load directly through `load_checkpoint_strict`.

Git LFS pointer files are rejected unless their SHA256 oid can be matched to a
real checkpoint file under a local `checkpoints/` directory. This lets the
known pointer path for experiment 1 resolve to:

```text
checkpoints/openpom_experiments_1_checkpoint2.pt
```

but prevents silently treating unresolved 133-byte pointer files as usable
model weights.

Current local state:

```text
/home/xing/openpom/models/ensemble_models/experiments_1/checkpoint2.pt
...
/home/xing/openpom/models/ensemble_models/experiments_10/checkpoint2.pt
```

are Git LFS pointer files. Only the experiment 1 real checkpoint is currently
present in this repository.

True 10-model ensemble acceptance is therefore gated by:

```text
POM_REPRO_ENSEMBLE_CHECKPOINTS
```

This environment variable should contain 10 unique real checkpoint paths joined
with the OS path separator.

## CLI Behavior

The CLI supports:

```text
--smiles       repeatable SMILES input
--smiles-file  newline-delimited SMILES file
--checkpoint   repeatable checkpoint input
--top-k        number of odor labels to return
--device       torch device
--no-embedding omit averaged embeddings
--output       optional JSON output file
```

If `--checkpoint` is omitted, the CLI reads checkpoint paths from
`POM_REPRO_ENSEMBLE_CHECKPOINTS`.

## Validation

The prediction layer validates:

```text
non-empty SMILES input
non-empty checkpoint list
1 <= top_k <= 138
SMILES featurization success
CUDA availability when a CUDA device is requested
finite probability and embedding outputs
probabilities within [0, 1]
```

## Test Results

Syntax check:

```bash
python -m py_compile \
  src/pom_repro/predict.py \
  scripts/predict_smiles.py \
  tests/test_predict_smiles.py
```

Result:

```text
PASS
```

Phase 8 tests:

```bash
PYTHONPATH=src /home/xing/miniconda3/envs/pom_openpom_ref/bin/python \
  -m pytest tests/test_predict_smiles.py -q
```

Result:

```text
8 passed, 1 skipped
```

Full regression suite:

```bash
PYTHONPATH=src /home/xing/miniconda3/envs/pom_openpom_ref/bin/python \
  -m pytest tests -q
```

Result:

```text
24 passed, 1 skipped
```

The skipped test is the true 10-model ensemble acceptance test, which requires
`POM_REPRO_ENSEMBLE_CHECKPOINTS` to be set to 10 unique real checkpoint files.
