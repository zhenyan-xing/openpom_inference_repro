# Phase 1: OpenPOM Reference Export

## Purpose

Phase 1 exports a fixed reference answer key from the original OpenPOM
implementation. Later reproduction work should compare local model outputs
against these files before changing architecture or checkpoint-loading logic.

## What Was Added

- Export script:
  - `scripts/export_openpom_reference.py`
- Local checkpoint copy:
  - `checkpoints/openpom_experiments_1_checkpoint2.pt`
- Reference outputs:
  - `reference_outputs/openpom_reference.json`
  - `reference_outputs/openpom_reference.npz`
- Checkpoint note:
  - `checkpoints/README.md`

## Runtime Environment

An isolated conda environment was created for the reference export:

```text
pom_openpom_ref
```

Python executable used:

```text
/home/xing/miniconda3/envs/pom_openpom_ref/bin/python
```

The environment was created separately from existing environments, and
`/home/xing/openpom` was used as read-only source code via `sys.path`. The
OpenPOM source tree was not modified or installed in editable mode.

The smoke test successfully imported:

```text
torch
dgl
deepchem
rdkit
openpom.feat.graph_featurizer.GraphFeaturizer
openpom.models.mpnn_pom.MPNNPOM
```

DeepChem printed warnings about optional TensorFlow, transformers, JAX, and
other optional model dependencies. Those packages are not required for this
OpenPOM inference export.

## Checkpoint

The files under `/home/xing/openpom/models/ensemble_models/*/checkpoint2.pt`
were Git LFS pointer files, not real model weights. The primary checkpoint was
therefore downloaded into this repository:

```text
checkpoints/openpom_experiments_1_checkpoint2.pt
```

Checkpoint identity:

```text
size_bytes: 25106835
sha256: 91182014cfa66dc673f7dd58fc77e3597ec345027f60d475abcc9dd18f6531f2
```

This SHA256 matches the Git LFS oid recorded in the original pointer file.

OpenPOM reference commit:

```text
5434c4b7badebca8e7d180aa34d2f787402121ee
```

## Export Details

The exported SMILES list is:

```text
C
CCO
CC(=O)O
c1ccccc1
CC(C)O
C1=CC=CC=C1O
```

The script uses the original OpenPOM `GraphFeaturizer`, converts graphs to DGL,
batches them, loads the checkpoint into original OpenPOM `MPNNPOM`, runs
`model.eval()` on CPU, and saves:

- SMILES
- probabilities
- logits
- POM embeddings
- graph feature summaries
- checkpoint metadata
- OpenPOM commit
- environment versions

The model hyperparameters match `/home/xing/openpom/predict_odors.py`, including:

```text
node_out_feats = 100
edge_hidden_feats = 75
edge_out_feats = 100
num_step_message_passing = 5
readout_type = set2set
num_step_set2set = 3
num_layer_set2set = 2
ffn_hidden_list = [392, 392]
ffn_embeddings = 256
ffn_dropout_p = 0.12
ffn_dropout_at_input_no_act = False
```

## Validation

The export command completed successfully:

```bash
/home/xing/miniconda3/envs/pom_openpom_ref/bin/python scripts/export_openpom_reference.py --device cpu
```

Generated output shapes:

```text
probs:      [6, 138]
logits:     [6, 138, 1]
embeddings: [6, 256]
```

Both `openpom_reference.json` and `openpom_reference.npz` were loaded after
writing. The SMILES order and array shapes matched the expected Phase 1
acceptance criteria.

## Re-run Command

To regenerate the reference outputs:

```bash
/home/xing/miniconda3/envs/pom_openpom_ref/bin/python scripts/export_openpom_reference.py --device cpu
```

To use a checkpoint stored elsewhere:

```bash
/home/xing/miniconda3/envs/pom_openpom_ref/bin/python scripts/export_openpom_reference.py --checkpoint /path/to/checkpoint2.pt --device cpu
```
