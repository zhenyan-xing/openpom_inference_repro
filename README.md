# POM Inference Reproduction

Checkpoint-compatible POM inference utilities for predicting multilabel
odor probabilities from SMILES strings.

This repo mirrors the Phase 7 parity path:

```text
GraphFeaturizer -> batch_graphs -> MPNNPOM -> load_checkpoint_strict
```

## Python API

```python
from pom_repro.predict import predict_smiles

result = predict_smiles(
    smiles=["C", "CCO"],
    checkpoint_paths=["checkpoints/openpom_experiments_1_checkpoint2.pt"],
    top_k=10,
    device="cpu",
    return_embedding=True,
)

print(result["top_k"][0])
print(result["probs"].shape)      # (2, 138)
print(result["embedding"].shape)  # (2, 256)
```

`probs` are averaged across all checkpoint paths. `embedding` is also averaged
when `return_embedding=True`; set `return_embedding=False` to omit it.

## CLI

```bash
PYTHONPATH=src python scripts/predict_smiles.py \
  --smiles C \
  --smiles CCO \
  --checkpoint checkpoints/openpom_experiments_1_checkpoint2.pt \
  --top-k 10 \
  --device cpu
```

The CLI writes JSON to stdout. Use `--output predictions.json` to also write the
same JSON to a file.

## True 10-Model Ensemble

OpenPOM ensembles probabilities across 10 independently trained checkpoints.
Provide those real checkpoint files either with repeated `--checkpoint` flags or
with `POM_REPRO_ENSEMBLE_CHECKPOINTS` using the OS path separator:

```bash
export POM_REPRO_ENSEMBLE_CHECKPOINTS="/path/to/exp1/checkpoint2.pt:/path/to/exp2/checkpoint2.pt:/path/to/exp3/checkpoint2.pt:/path/to/exp4/checkpoint2.pt:/path/to/exp5/checkpoint2.pt:/path/to/exp6/checkpoint2.pt:/path/to/exp7/checkpoint2.pt:/path/to/exp8/checkpoint2.pt:/path/to/exp9/checkpoint2.pt:/path/to/exp10/checkpoint2.pt"

PYTHONPATH=src python scripts/predict_smiles.py --smiles CCO --top-k 10
```

Current local note: the files under
`/home/xing/openpom/models/ensemble_models/experiments_*/checkpoint2.pt` are Git
LFS pointers, not usable weights. The API rejects unresolved pointers. A pointer
is accepted only when a real matching SHA256 checkpoint exists under
`checkpoints/`.

The bundled real checkpoint is:

```text
checkpoints/openpom_experiments_1_checkpoint2.pt
```
