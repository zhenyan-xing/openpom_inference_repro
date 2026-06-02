# POM/OpenPOM Architecture Specification

## Scope

This document is the first-stage architecture contract for the local
checkpoint-compatible OpenPOM/POM reproduction.

The goal is to reimplement the inference-time core of OpenPOM's `MPNNPOM`
architecture so that this repository can:

```text
SMILES -> graph features -> DGL batched graph -> MPNNPOM-style model
       -> proba, logits, embeddings
```

and then load OpenPOM checkpoints with minimal or no key remapping.

This specification focuses on:

```text
featurizer contract
graph batching contract
model module contract
tensor shapes
forward return values
checkpoint-compatible module naming
```

Training loops, cross-validation, AUROC reproduction, and hyperparameter
search are intentionally out of scope for this first-stage spec.

## Reference Implementation

Local OpenPOM path:

```text
/home/xing/openpom
```

Reference OpenPOM commit observed when this project plan was written:

```text
5434c4b7badebca8e7d180aa34d2f787402121ee
```

Primary single-checkpoint target for this repository:

```text
checkpoints/openpom_experiments_1_checkpoint2.pt
```

This is the local real checkpoint copy used by Phase 1 and Phase 2. The
OpenPOM ensemble path that originally identified it is:

```text
/home/xing/openpom/models/ensemble_models/experiments_1/checkpoint2.pt
```

The files under `/home/xing/openpom/models/ensemble_models/*/checkpoint2.pt`
were observed as Git LFS pointer files locally, so use the checkpoint under
`checkpoints/` as the implementation source of truth.

Important reference source files:

```text
/home/xing/openpom/openpom/feat/graph_featurizer.py
/home/xing/openpom/openpom/models/mpnn_pom.py
/home/xing/openpom/openpom/layers/pom_mpnn_gnn.py
/home/xing/openpom/openpom/layers/pom_ffn.py
```

OpenPOM should be treated as read-only reference code. Do not modify it.

Local reproduction evidence:

```text
docs/phase1_openpom_reference.md
docs/phase2_checkpoint_inspection.md
reports/checkpoint_keys.txt
reports/checkpoint_tensor_shapes.csv
```

## High-Level Pipeline

The reproduction should follow this data flow:

```text
SMILES
  -> RDKit Mol
  -> atom feature matrix
  -> directed bond edge index
  -> bond feature matrix
  -> graph object / DGLGraph
  -> dgl.batch(...)
  -> MPNN message passing
  -> atom+bond readout
  -> Set2Set graph readout
  -> FFN
  -> proba, logits, embeddings
```

User-facing inference may accept `list[str]` SMILES, but the model `forward`
method itself should accept a batched DGL graph.

## Input Graph Contract

The model input is a DGL batched graph with:

```text
g.ndata["x"]
g.edata["edge_attr"]
```

Expected shapes:

```text
g.ndata["x"]:         [total_num_atoms_in_batch, 134]
g.edata["edge_attr"]: [total_num_directed_edges_in_batch, 6]
```

Each chemical bond must be represented as two directed edges:

```text
atom_i -> atom_j
atom_j -> atom_i
```

Single-atom molecules are valid. They should have:

```text
node feature shape: [1, 134]
edge feature shape: [0, 6]
```

For a molecule such as `CCO`, expected graph-level shapes are:

```text
node feature shape: [3, 134]
edge feature shape: [4, 6]
```

The reproduction should match OpenPOM edge order if possible. If edge order
differs, the difference must be documented and shown not to affect output
parity.

## Atom Feature Contract

Atom feature dimension:

```text
ATOM_FDIM = 134
```

Atom features are concatenated one-hot encodings based on:

```text
total valence
total degree
total number of hydrogens
formal charge
atomic number
hybridization
```

The expected OpenPOM feature groups are:

```text
total valence:             0, 1, 2, 3, 4, 5, 6, unknown
total degree:              0, 1, 2, 3, 4, 5, unknown
total number of hydrogens: 0, 1, 2, 3, 4, unknown
formal charge:             -1, -2, 1, 2, 0, unknown
atomic number:             0 through 99, unknown
hybridization:             SP, SP2, SP3, SP3D, SP3D2, unknown
```

The feature order must match OpenPOM exactly. Matching only the total dimension
is not sufficient.

OpenPOM uses RDKit-derived implicit hydrogen information by default. The
reference featurizer uses `GraphFeaturizer(is_adding_hs=False)`, so explicit
hydrogens should not be added in the reproduction unless a later parity report
proves OpenPOM did so for a specific run.

## Bond Feature Contract

Bond feature dimension:

```text
BOND_FDIM = 6
```

Bond feature order:

```text
[no_bond_placeholder, is_single, is_double, is_triple, is_aromatic, is_in_ring]
```

For a real bond:

```text
no_bond_placeholder = 0
```

For a missing or placeholder bond, if such a case is used:

```text
[1, 0, 0, 0, 0, 0]
```

For normal molecular graphs, every RDKit bond should produce two directed edges
with identical bond feature values.

## Model Hyperparameter Contract

Checkpoint-compatible OpenPOM `MPNNPOM` values to reproduce for
`checkpoints/openpom_experiments_1_checkpoint2.pt`:

```text
n_tasks: 138
n_classes: 1
number_atom_features: 134
number_bond_features: 6
node_out_feats: 100
edge_hidden_feats: 75
edge_out_feats: 100
num_step_message_passing: 5
mpnn_residual: True
message_aggregator_type: sum
readout_type: set2set
num_step_set2set: 3
num_layer_set2set: 2
ffn_hidden_list: [392, 392]
ffn_embeddings: 256
ffn_activation: relu
ffn_dropout_p: 0.12
ffn_dropout_at_input_no_act: False
```

The default task is multilabel binary classification over 138 odor labels.
These values match the Phase 1 OpenPOM reference export and the Phase 2
checkpoint tensor shapes. Older OpenPOM examples may mention smaller default
model widths; those are not compatible with the local checkpoint.

## Checkpoint-Compatible Module Names

To reduce checkpoint remapping, the reproduced model should keep module names
close to OpenPOM:

```text
mpnn
project_edge_feats
readout_set2set
ffn
```

Do not rename these to unrelated names such as `encoder`, `pooler`, or `head`
unless a checkpoint inspection proves remapping is unavoidable.

## MPNN Message Passing Contract

The message passing module is an MPNN/NNConv-style graph neural network.

Input tensors:

```text
node_feats: [total_num_atoms, 134]
edge_feats: [total_num_directed_edges, 6]
```

Expected hidden node output:

```text
node_encodings: [total_num_atoms, 100]
```

Edge features participate in message passing through an edge network:

```text
edge feature, dim 6
  -> Linear(6, edge_hidden_feats)
  -> ReLU
  -> Linear(edge_hidden_feats, node_out_feats * node_out_feats)
  -> edge-conditioned message transform
```

With default values:

```text
edge feature, dim 6
  -> Linear(6, 75)
  -> ReLU
  -> Linear(75, 100 * 100)
```

Message passing uses:

```text
num_step_message_passing = 5
message_aggregator_type = "sum"
mpnn_residual = True
```

Checkpoint key evidence:

```text
mpnn.project_node_feats.0.weight: (100, 134)
mpnn.gnn_layer.edge_func.0.weight: (75, 6)
mpnn.gnn_layer.edge_func.2.weight: (10000, 75)
mpnn.gru.weight_ih_l0: (300, 100)
mpnn.gru.weight_hh_l0: (300, 100)
```

## Edge Projection And Atom+Bond Readout Contract

OpenPOM uses edge features again after MPNN message passing.

The raw edge feature is projected with:

```text
project_edge_feats: Linear(6, edge_out_feats) + ReLU
```

With default values:

```text
edge_emb: [total_num_directed_edges, 100]
```

For each directed edge, concatenate:

```text
source node encoding: [100]
edge embedding:       [100]
```

to produce a directed message:

```text
src_msg: [200]
```

These directed atom+bond messages are aggregated to the destination nodes,
producing per-node combined features:

```text
node_combined_feats: [total_num_atoms, 200]
```

This readout detail is important. The Set2Set input is not only the MPNN node
embedding; it is the atom+bond combined representation.

Checkpoint key evidence:

```text
project_edge_feats.0.weight: (100, 6)
project_edge_feats.0.bias: (100,)
```

## Graph Readout Contract

Default graph-level readout:

```text
Set2Set
```

Default Set2Set parameters:

```text
num_step_set2set = 3
num_layer_set2set = 2
input_dim = node_out_feats + edge_out_feats = 200
```

Set2Set doubles the input feature dimension, so the graph representation shape
is:

```text
graph_feats: [batch_size, 400]
```

`global_sum_pooling` exists as an optional OpenPOM mode, but it is not the
first-stage target unless checkpoint inspection shows a checkpoint was trained
with that mode.

Checkpoint key evidence:

```text
readout_set2set.lstm.weight_ih_l0: (800, 400)
readout_set2set.lstm.weight_hh_l0: (800, 200)
readout_set2set.lstm.weight_ih_l1: (800, 200)
readout_set2set.lstm.weight_hh_l1: (800, 200)
```

## FFN And Embedding Contract

The FFN receives the graph representation:

```text
graph_feats: [batch_size, 400]
```

Default FFN hidden and embedding settings:

```text
ffn_hidden_list = [392, 392]
ffn_embeddings = 256
ffn_activation = "relu"
ffn_dropout_p = 0.12
ffn_dropout_at_input_no_act = False
```

OpenPOM constructs the hidden stack conceptually as:

```text
[392, 392, 256]
```

The 256-dimensional penultimate output is the POM embedding:

```text
embeddings: [batch_size, 256]
```

The final output dimension for classification is:

```text
n_tasks * n_classes = 138 * 1 = 138
```

Raw FFN output before final reshape:

```text
ffn_output: [batch_size, 138]
```

Checkpoint key evidence:

```text
ffn.linears.0.weight: (392, 400)
ffn.linears.1.weight: (392, 392)
ffn.linears.2.weight: (256, 392)
ffn.linears.3.weight: (138, 256)
ffn.batchnorms.0.weight: (392,)
ffn.batchnorms.1.weight: (392,)
ffn.batchnorms.2.weight: (256,)
```

## Forward Output Contract

For classification, `forward` should return:

```python
proba, logits, embeddings = model(g)
```

Expected final shapes:

```text
proba:      [batch_size, 138]
logits:     [batch_size, 138, 1]
embeddings: [batch_size, 256]
```

Classification is multilabel binary classification:

```text
proba = sigmoid(logits)
```

Because `n_classes = 1`, `proba` is squeezed to:

```text
[batch_size, n_tasks]
```

while `logits` retains:

```text
[batch_size, n_tasks, 1]
```

The external prediction API may later wrap these outputs in dictionaries, but
the core model should first preserve this OpenPOM-style tuple contract.

## Minimal Shape Smoke Tests

The model reproduction should pass these first:

### Featurizer Shapes

```text
C:
  node_features: [1, 134]
  edge_features: [0, 6]

CCO:
  node_features: [3, 134]
  edge_features: [4, 6]
```

### Model Forward Shapes

For a batch with two molecules:

```text
proba.shape == [2, 138]
logits.shape == [2, 138, 1]
embeddings.shape == [2, 256]
```

All outputs must be finite:

```text
no NaN
no inf
```

## Required Parity Checks

Shape checks are necessary but not sufficient.

The reproduction must eventually compare against OpenPOM on the same SMILES
and checkpoint:

```text
atom feature max_abs_diff
bond feature max_abs_diff
edge order
checkpoint missing keys
checkpoint unexpected keys
checkpoint shape mismatches
probs max_abs_diff
probs mean_abs_diff
logits max_abs_diff
logits mean_abs_diff
embedding cosine similarity
top-10 label overlap
top-10 label order
```

If output parity fails, debug in this order:

```text
1. atom/bond feature value and order
2. graph edge order and batching
3. checkpoint state_dict loading
4. atom+bond readout implementation
5. Set2Set configuration
6. FFN embedding/output behavior
```

## Checkpoint Loading Requirements

The local checkpoint is a wrapped dictionary. Load the model weights from:

```text
checkpoint["model_state_dict"]
```

The wrapped checkpoint also contains:

```text
optimizer_state_dict
global_step = 1984
```

The model state dict has no extra `model.` prefix. Expected top-level
parameter prefixes are:

```text
mpnn
project_edge_feats
readout_set2set
ffn
```

Checkpoint loading should be strict by default:

```python
model.load_state_dict(state_dict, strict=True)
```

If strict loading fails:

1. inspect missing keys;
2. inspect unexpected keys;
3. inspect shape mismatches;
4. only then add explicit key remapping;
5. document every remapping rule in `docs/checkpoint_compatibility.md`.

Do not silently use `strict=False`. A checkpoint that "loads" while skipping
core layers does not validate architecture reproduction.

## Out Of Scope For This File

The following topics are useful for a fuller reproduction, but are not part of
this first-stage architecture spec:

```text
training loop
loss implementation
class imbalance weighting
small overfit tests
cross-validation
AUROC reproduction
10-model ensemble benchmark reproduction
hyperparameter search
data curation notebooks
new molecular standardization choices
```

These should be documented later only after the checkpoint-compatible inference
core is working.
