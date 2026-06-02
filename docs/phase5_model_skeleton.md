# Phase 5: Model Skeleton And Forward Pass

Phase 5 implemented the checkpoint-shaped OpenPOM model skeleton in this
repository. The goal was to build the core model modules with OpenPOM-compatible
names, tensor shapes, and forward outputs so the next phase can focus on strict
checkpoint loading and parity checks.

## Outputs

Implemented files:

```text
src/pom_repro/mpnn.py
src/pom_repro/readout.py
src/pom_repro/ffn.py
src/pom_repro/model.py
tests/test_model_shapes.py
```

The public model import is:

```python
from pom_repro.model import MPNNPOM
```

`src/pom_repro/__init__.py` was intentionally left DGL-free so the existing
featurizer utilities remain importable in environments without `dgl` or
`dgllife`.

## Implemented Architecture

The default `MPNNPOM` configuration matches the Phase 1 reference export and
Phase 2 checkpoint inspection:

```text
n_tasks = 138
n_classes = 1
number_atom_features = 134
number_bond_features = 6
node_out_feats = 100
edge_hidden_feats = 75
edge_out_feats = 100
num_step_message_passing = 5
mpnn_residual = True
message_aggregator_type = sum
readout_type = set2set
num_step_set2set = 3
num_layer_set2set = 2
ffn_hidden_list = [392, 392]
ffn_embeddings = 256
ffn_activation = relu
ffn_dropout_p = 0.12
ffn_dropout_at_input_no_act = False
```

OpenPOM-compatible module names were preserved:

```text
mpnn
project_edge_feats
readout_set2set
ffn
```

The model forward contract is:

```python
proba, logits, embeddings = model(g)
```

with output shapes:

```text
proba:      [batch_size, 138]
logits:     [batch_size, 138, 1]
embeddings: [batch_size, 256]
```

## Module Details

`src/pom_repro/mpnn.py` defines `CustomMPNNGNN`, an OpenPOM-style subclass of
`dgllife.model.gnn.MPNNGNN`. It keeps the base `project_node_feats` and `gru`
modules, then replaces `gnn_layer` with a DGL `NNConv` using the OpenPOM edge
network:

```text
Linear(6, 75) -> ReLU -> Linear(75, 10000)
```

`src/pom_repro/readout.py` implements the atom+bond readout used before graph
pooling. Raw edge features are projected through `project_edge_feats`, then
concatenated with source node encodings and summed onto destination nodes. The
resulting `[num_nodes, 200]` representation is passed to `readout_set2set`.

`src/pom_repro/ffn.py` implements an OpenPOM-compatible
`CustomPositionwiseFeedForward` with:

```text
400 -> 392 -> 392 -> 256 -> 138
```

The 256-dimensional penultimate linear output is returned as the POM embedding.

`src/pom_repro/model.py` wires the MPNN, edge projection, Set2Set readout, and
FFN into `MPNNPOM`. Classification mode reshapes raw FFN output to
`[batch_size, 138, 1]`, applies sigmoid, and squeezes probabilities to
`[batch_size, 138]`.

## Tests And Validation

Added `tests/test_model_shapes.py` with two checks:

- forward shape and finite-output smoke test on a batched DGL graph from
  `GraphFeaturizer().featurize(["C", "CCO"])`;
- state dict shape contract test for key checkpoint-compatible tensors.

The test file uses:

```python
pytest.importorskip("torch")
pytest.importorskip("dgl")
pytest.importorskip("dgllife")
```

so environments without model dependencies skip these tests instead of failing
during collection.

Validation commands run successfully in the `pom_openpom_ref` environment:

```bash
python -m py_compile \
  src/pom_repro/mpnn.py \
  src/pom_repro/readout.py \
  src/pom_repro/ffn.py \
  src/pom_repro/model.py \
  tests/test_model_shapes.py

PYTHONPATH=src /home/xing/miniconda3/envs/pom_openpom_ref/bin/python \
  -m pytest tests/test_model_shapes.py -q

PYTHONPATH=src /home/xing/miniconda3/envs/pom_openpom_ref/bin/python \
  -m pytest tests/test_featurizer.py -q
```

Observed results:

```text
tests/test_model_shapes.py: 2 passed
tests/test_featurizer.py:   6 passed
```

## Checkpoint Compatibility

The reproduced model state dict was compared against:

```text
checkpoints/openpom_experiments_1_checkpoint2.pt
```

Results:

```text
model_keys = 44
checkpoint_keys = 44
missing_in_model = []
extra_in_model = []
shape_mismatches = []
```

Strict checkpoint loading also succeeded:

```python
model.load_state_dict(checkpoint["model_state_dict"], strict=True)
```

After strict loading, a forward pass on `["C", "CCO"]` produced:

```text
proba      (2, 138)     finite=True
logits     (2, 138, 1)  finite=True
embeddings (2, 256)     finite=True
```

## Reference Output Sanity Check

As an extra check beyond Phase 5 requirements, the strict-loaded local model was
compared against `reference_outputs/openpom_reference.npz` for the six Phase 1
reference SMILES.

Observed maximum absolute differences:

```text
probs:      1.6391277313232422e-07
logits:     1.1920928955078125e-06
embeddings: 7.152557373046875e-07
```

This indicates the Phase 5 skeleton is already numerically aligned with the
original OpenPOM reference to small floating-point tolerance for the checked
samples.

## Notes For Phase 6

Phase 6 can start from direct strict loading of
`checkpoint["model_state_dict"]`; no key remapping was needed during Phase 5
validation.

The base environment at the time of implementation did not include `pytest`,
`dgl`, `dgllife`, or `rdkit`. Full model tests were run in:

```text
/home/xing/miniconda3/envs/pom_openpom_ref
```
