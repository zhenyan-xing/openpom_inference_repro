# POM/OpenPOM Checkpoint-Compatible Reproduction Plan

## Project Goal

This project is a **checkpoint-compatible architecture reproduction** of the
OpenPOM/POM-style model.

The goal is not to train a new model, not to build a GUI, and not merely to
make an inference demo. The goal is to reimplement the core OpenPOM
`MPNNPOM` architecture from scratch in this repository, load public OpenPOM
checkpoints, and verify that the same SMILES produce outputs aligned with the
original OpenPOM implementation.

In short:

```text
Reproduce the core architecture -> load OpenPOM checkpoint -> compare outputs.
```

OpenPOM is used only as the reference implementation. Do not modify the local
OpenPOM source tree.

## Local Reference Setup

Current local project path:

```text
/home/xing/pom_inference_repro
```

Local OpenPOM reference path:

```text
/home/xing/openpom
```

Reference OpenPOM commit observed at plan creation:

```text
5434c4b7badebca8e7d180aa34d2f787402121ee
```

Primary single-checkpoint target:

```text
/home/xing/openpom/models/ensemble_models/experiments_1/checkpoint2.pt
```

Available ensemble checkpoints:

```text
/home/xing/openpom/models/ensemble_models/experiments_1/checkpoint2.pt
/home/xing/openpom/models/ensemble_models/experiments_2/checkpoint2.pt
/home/xing/openpom/models/ensemble_models/experiments_3/checkpoint2.pt
/home/xing/openpom/models/ensemble_models/experiments_4/checkpoint2.pt
/home/xing/openpom/models/ensemble_models/experiments_5/checkpoint2.pt
/home/xing/openpom/models/ensemble_models/experiments_6/checkpoint2.pt
/home/xing/openpom/models/ensemble_models/experiments_7/checkpoint2.pt
/home/xing/openpom/models/ensemble_models/experiments_8/checkpoint2.pt
/home/xing/openpom/models/ensemble_models/experiments_9/checkpoint2.pt
/home/xing/openpom/models/ensemble_models/experiments_10/checkpoint2.pt
```

Important OpenPOM source files to consult:

```text
/home/xing/openpom/openpom/models/mpnn_pom.py
/home/xing/openpom/openpom/feat/graph_featurizer.py
/home/xing/openpom/openpom/layers/pom_ffn.py
/home/xing/openpom/openpom/layers/pom_mpnn_gnn.py
```

## Proposed Repository Layout

```text
pom_inference_repro/
  README.md
  docs/
    reproduction_plan.md
    architecture_spec.md
    checkpoint_compatibility.md

  src/
    pom_repro/
      __init__.py
      constants.py
      labels.py
      featurizer.py
      graph_batch.py
      mpnn.py
      readout.py
      ffn.py
      model.py
      checkpoint.py
      predict.py

  scripts/
    export_openpom_reference.py
    inspect_checkpoint.py
    compare_graph_features.py
    compare_with_openpom.py
    predict_smiles.py

  tests/
    test_featurizer.py
    test_model_shapes.py
    test_checkpoint_load.py
    test_predict_smiles.py

  reports/
  reference_outputs/
  checkpoints/
    README.md
```

## Core Model Contract To Reproduce

These values were confirmed from the Phase 1 OpenPOM reference export and
Phase 2 checkpoint inspection. They are the checkpoint-compatible OpenPOM
contract for `checkpoints/openpom_experiments_1_checkpoint2.pt`:

```text
atom feature dim: 134
bond feature dim: 6
n_tasks: 138
node_out_feats: 100
edge_hidden_feats: 75
edge_out_feats: 100
num_step_message_passing: 5
readout: Set2Set
num_step_set2set: 3
num_layer_set2set: 2
ffn_hidden_list: [392, 392]
POM embedding dim: 256
forward return: proba, logits, embeddings
```

Module names should stay close to OpenPOM where possible:

```text
mpnn
project_edge_feats
readout_set2set
ffn
```

This reduces the amount of checkpoint key remapping needed later.

## Phase 0: Lock Reference Baseline

### Purpose

Make the reference target explicit so future comparisons are reproducible.

### Tasks

- Record local OpenPOM path.
- Record OpenPOM commit hash or tag.
- Record checkpoint path.
- Record Python, PyTorch, DGL, DeepChem, RDKit, and CUDA versions if relevant.

### Expected Outputs

```text
docs/checkpoint_compatibility.md
reports/environment.txt
```

### Acceptance Criteria

- The exact OpenPOM reference version is written down.
- The exact checkpoint path is written down.
- Future agents can tell which implementation and checkpoint all comparisons
  refer to.

## Phase 1: Export OpenPOM Reference Outputs

### Purpose

Create a fixed "answer key" from the original OpenPOM implementation before
writing the reproduction.

### Tasks

- Implement:

```text
scripts/export_openpom_reference.py
```

- Use original OpenPOM from:

```text
/home/xing/openpom
```

- Load the primary single checkpoint:

```text
/home/xing/openpom/models/ensemble_models/experiments_1/checkpoint2.pt
```

- Run a fixed SMILES list, for example:

```text
C
CCO
CC(=O)O
c1ccccc1
CC(C)O
C1=CC=CC=C1O
```

- Save probabilities, logits, embeddings, and graph feature summaries.

### Expected Outputs

```text
reference_outputs/openpom_reference.json
reference_outputs/openpom_reference.npz
```

### Acceptance Criteria

- The original OpenPOM implementation can be run locally.
- Reference outputs are saved and can be loaded by comparison scripts.
- The file records the checkpoint and OpenPOM commit used to produce it.

## Phase 2: Inspect Checkpoint Structure

### Purpose

Use the checkpoint to drive implementation details instead of guessing module
names or tensor shapes.

### Tasks

- Implement:

```text
scripts/inspect_checkpoint.py
```

- The script should:
  - load a checkpoint with `torch.load(..., map_location="cpu")`;
  - print and save top-level checkpoint keys;
  - detect whether the checkpoint is a direct `state_dict` or wrapped dict;
  - print and save the first 100 state_dict keys;
  - save every tensor name, shape, and dtype.

### Expected Outputs

```text
reports/checkpoint_keys.txt
reports/checkpoint_tensor_shapes.csv
```

### Acceptance Criteria

- It is clear whether checkpoint keys have prefixes such as `model.`.
- It is clear which keys correspond to MPNN, Set2Set, edge projection, FFN,
  BatchNorm, or other layers.
- Tensor shapes can be used to verify the architecture spec.

## Phase 3: Write Architecture Spec

### Purpose

Create a precise technical contract for what the reproduction must implement.

### Tasks

- Implement:

```text
docs/architecture_spec.md
```

- Document:
  - SMILES input assumptions;
  - DGL graph input assumptions;
  - node feature name and shape;
  - edge feature name and shape;
  - message passing module;
  - edge projection module;
  - readout module;
  - FFN module;
  - output tensor shapes;
  - checkpoint key compatibility considerations.

### Expected Contents

```text
Input:
  list[str] SMILES

Graph:
  atom feature dim = 134
  bond feature dim = 6

Model:
  MPNN / NNConv-style message passing
  node_out_feats = 100
  edge_hidden_feats = 75
  edge_out_feats = 100
  num_step_message_passing = 5
  readout = Set2Set
  num_step_set2set = 3
  num_layer_set2set = 2
  ffn_hidden_list = [392, 392]
  ffn_embeddings = 256
  n_tasks = 138

Output:
  proba shape = [batch, 138]
  logits shape = [batch, 138, 1]
  embeddings shape = [batch, 256]
```

### Acceptance Criteria

- A new agent can implement the model from the spec without rereading the full
  conversation.
- Any unknowns are marked explicitly and tied back to checkpoint/source
  inspection.

## Phase 4: Reproduce Featurizer

### Purpose

Match OpenPOM graph features exactly. This is critical because the same
checkpoint with a different feature order will produce different outputs.

### Tasks

- Implement:

```text
src/pom_repro/featurizer.py
src/pom_repro/graph_batch.py
scripts/compare_graph_features.py
tests/test_featurizer.py
```

- Use RDKit.
- Reproduce OpenPOM atom feature order.
- Reproduce OpenPOM bond feature order.
- Construct each bond as two directed edges.
- Try to match OpenPOM edge order.

### Acceptance Criteria

Basic shape checks:

```text
C -> node_features [1, 134], edge_features [0, 6]
CCO -> node_features [3, 134], edge_features [4, 6]
```

Parity checks against OpenPOM:

```text
atom feature max_abs_diff = 0
bond feature max_abs_diff = 0
edge order is identical, or any difference is documented and proven harmless
```

If output parity fails later, debug the featurizer before tuning model logic.

## Phase 5: Implement Model Skeleton And Forward Pass

### Purpose

Build the reproduced model structure with OpenPOM-compatible module names and
tensor shapes.

### Tasks

- Implement:

```text
src/pom_repro/mpnn.py
src/pom_repro/readout.py
src/pom_repro/ffn.py
src/pom_repro/model.py
tests/test_model_shapes.py
```

- Keep names close to OpenPOM:

```text
mpnn
project_edge_feats
readout_set2set
ffn
```

- Keep forward output close to OpenPOM:

```python
proba, logits, embeddings = model(g)
```

### Acceptance Criteria

With random initialization and a small graph batch:

```text
proba shape = [batch, 138]
logits shape = [batch, 138, 1]
embeddings shape = [batch, 256]
no NaN values
```

Shape tests are necessary but not sufficient. The real validation is
checkpoint loading and output parity.

## Phase 6: Load Checkpoint Strictly

### Purpose

Verify that the reproduced architecture is compatible with OpenPOM checkpoint
parameters.

### Tasks

- Implement:

```text
src/pom_repro/checkpoint.py
tests/test_checkpoint_load.py
```

- The loader should:
  - extract the correct `state_dict`;
  - try `model.load_state_dict(state_dict, strict=True)` first;
  - only use key remapping if direct strict load fails;
  - report missing keys, unexpected keys, and shape mismatches;
  - never silently use `strict=False`.

### Expected Outputs

```text
reports/checkpoint_load_report.txt
```

### Acceptance Criteria

Priority order:

```text
shape mismatch = 0
missing_keys = 0, or every missing key is explained
unexpected_keys = 0, or every unexpected key is explained
```

If strict loading fails, update `docs/checkpoint_compatibility.md` with the
reason and remapping rules.

## Phase 7: Single-Checkpoint Output Parity

### Purpose

Compare reproduced model outputs with original OpenPOM outputs for the same
SMILES and same checkpoint.

### Tasks

- Implement:

```text
scripts/compare_with_openpom.py
```

- Compare against:

```text
reference_outputs/openpom_reference.npz
```

- Start with only the primary checkpoint:

```text
/home/xing/openpom/models/ensemble_models/experiments_1/checkpoint2.pt
```

### Metrics

```text
probs max_abs_diff
probs mean_abs_diff
logits max_abs_diff
logits mean_abs_diff
embedding cosine similarity
top-10 label overlap
top-10 label order
```

### Expected Outputs

```text
reports/output_parity_single_checkpoint.json
```

### Acceptance Criteria

Ideal:

```text
probs max_abs_diff around 1e-5 to 1e-4
top-10 labels and order match
embedding cosine similarity near 1
```

Acceptable if library or floating point differences exist:

```text
top-k labels mostly match
probabilities show only small drift
```

If differences are large, debug in this order:

```text
1. featurizer value/order
2. graph batching and edge order
3. checkpoint load report
4. readout logic
5. FFN embedding/output logic
```

## Phase 8: Ensemble And Predict API

### Purpose

Only after single-checkpoint parity is working, add the user-facing prediction
API and ensemble support.

### Tasks

- Implement:

```text
src/pom_repro/predict.py
scripts/predict_smiles.py
tests/test_predict_smiles.py
README.md
```

- Suggested API:

```python
def predict_smiles(
    smiles: list[str],
    checkpoint_paths: list[str],
    top_k: int = 10,
    device: str = "cpu",
    return_embedding: bool = True,
):
    ...
```

### Ensemble Behavior

```text
probs = average probabilities across checkpoints
embedding = optional average, or return per-checkpoint embeddings
```

### Acceptance Criteria

```text
single checkpoint prediction works
10-model ensemble prediction works
top-k odor labels are returned
probabilities are in [0, 1]
no NaN values
```

## Recommended Execution Order

Do not skip ahead unless a phase is intentionally marked as deferred.

```text
0. Lock OpenPOM local reference version
1. Export original OpenPOM reference outputs
2. Inspect checkpoint keys and tensor shapes
3. Write architecture_spec.md
4. Reproduce featurizer and pass feature parity checks
5. Implement model shape forward pass
6. Strictly load checkpoint
7. Compare single-checkpoint outputs
8. Add ensemble prediction and README
```

## First-Phase Scope Boundary

The first phase of this project should not include:

```text
training
GUI
advanced pretrained molecular models
AUROC reproduction
hyperparameter search
new model design
```

The first-phase success criteria are:

```text
1. featurizer matches OpenPOM
2. model state_dict can be loaded or remapped with full reporting
3. missing/unexpected/shape mismatch keys are explained
4. single-checkpoint outputs align with OpenPOM reference outputs
```

## Guidance For Future Agents

If you are another agent continuing this work:

1. Treat this repository as the new from-scratch reproduction.
2. Treat `/home/xing/openpom` as read-only reference code.
3. Do not modify OpenPOM source files.
4. Prioritize checkpoint compatibility over aesthetic refactors.
5. Keep module names close to OpenPOM unless there is a strong reason not to.
6. Do not silently use `strict=False` when loading checkpoints.
7. If outputs differ, debug featurizer and state_dict loading before changing
   model math.
8. Update this plan or add notes to `docs/checkpoint_compatibility.md` when a
   phase changes based on real evidence.

The main thread of the project is:

```text
feature parity -> checkpoint load parity -> output parity
```
