# Checkpoint Compatibility

## Current Status

The local OpenPOM-compatible `MPNNPOM` implementation directly strict-loads
the primary checkpoint:

```text
checkpoints/openpom_experiments_1_checkpoint2.pt
```

Strict loading succeeds with:

```text
state_dict_source = wrapped:model_state_dict
model_key_count = 44
checkpoint_key_count = 44
missing_keys = 0
unexpected_keys = 0
shape_mismatches = 0
remap_used = False
```

No checkpoint key remapping rules are currently required.

## Loading Policy

Use `src/pom_repro/checkpoint.py` for checkpoint loading. The loader must try:

```python
model.load_state_dict(state_dict, strict=True)
```

before any remapping logic. It must not silently use `strict=False`.

If strict loading fails for a future checkpoint:

1. inspect missing keys;
2. inspect unexpected keys;
3. inspect tensor shape mismatches;
4. add a targeted key remapper only if the mismatch is a naming issue;
5. document every remapping rule in this file.

## Remapping Rules

```text
none
```

## Reproduce The Report

Run:

```bash
PYTHONPATH=src /home/xing/miniconda3/envs/pom_openpom_ref/bin/python \
  -m pom_repro.checkpoint \
  --checkpoint checkpoints/openpom_experiments_1_checkpoint2.pt \
  --report reports/checkpoint_load_report.txt
```

The expected result is:

```text
result: PASS
notes: Direct strict loading succeeded. No key remapping was used.
```
