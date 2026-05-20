# Phase 7: Single-Checkpoint Output Parity

## Purpose

Phase 7 formalized the output comparison between the local
OpenPOM-compatible implementation and the original OpenPOM reference outputs.
The goal was to verify that a single checkpoint produces aligned probabilities,
logits, embeddings, and top-k odor labels.

## What Was Added

New comparison script:

```text
scripts/compare_with_openpom.py
```

New test file:

```text
tests/test_output_parity.py
```

Generated report:

```text
reports/output_parity_single_checkpoint.json
```

## Checkpoint Handling

The requested OpenPOM checkpoint path is:

```text
/home/xing/openpom/models/ensemble_models/experiments_1/checkpoint2.pt
```

That file is a 133-byte Git LFS pointer, not real model weights. The Phase 7
script accepts it as the default requested checkpoint, reads its SHA256 oid,
and resolves it to the real local checkpoint copy:

```text
checkpoints/openpom_experiments_1_checkpoint2.pt
```

The resolved checkpoint SHA256 is:

```text
91182014cfa66dc673f7dd58fc77e3597ec345027f60d475abcc9dd18f6531f2
```

The report records both the requested pointer path and the resolved local
checkpoint path.

## Comparison Behavior

The script compares local model outputs against:

```text
reference_outputs/openpom_reference.npz
```

It uses the SMILES and task label order stored in that reference file. The
local path is:

```text
GraphFeaturizer -> batch_graphs -> MPNNPOM -> load_checkpoint_strict
```

The script checks:

```text
probs max_abs_diff
probs mean_abs_diff
logits max_abs_diff
logits mean_abs_diff
embedding cosine similarity
top-10 label overlap
top-10 label order
```

It also validates output shapes and finite values before writing the JSON
report.

## Result

The Phase 7 report result is:

```text
status: ideal
```

Observed metrics:

```text
probs max_abs_diff:      0.0
probs mean_abs_diff:     0.0
logits max_abs_diff:     0.0
logits mean_abs_diff:    0.0
embedding cosine min:    0.9999999403953552
embedding cosine mean:   1.0
top-10 min overlap:      10
top-10 all order match:  true
```

Strict checkpoint loading also remained clean:

```text
model_key_count = 44
checkpoint_key_count = 44
missing_keys = []
unexpected_keys = []
shape_mismatches = []
direct_strict_success = True
remap_used = False
```

## Validation

Commands run:

```bash
/home/xing/miniconda3/envs/pom_openpom_ref/bin/python \
  scripts/compare_with_openpom.py --device cpu
```

Result:

```text
status: ideal
```

```bash
PYTHONPATH=src /home/xing/miniconda3/envs/pom_openpom_ref/bin/python \
  -m pytest tests/test_output_parity.py -q
```

Result:

```text
3 passed
```

Related regression checks:

```bash
PYTHONPATH=src /home/xing/miniconda3/envs/pom_openpom_ref/bin/python \
  -m pytest \
  tests/test_featurizer.py \
  tests/test_model_shapes.py \
  tests/test_checkpoint_load.py \
  tests/test_output_parity.py \
  -q
```

Result:

```text
16 passed
```

## Notes For Phase 8

Single-checkpoint parity is now established. Phase 8 can build user-facing
prediction and ensemble support on top of the existing strict checkpoint loader
and local inference path.
