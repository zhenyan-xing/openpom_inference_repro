# Phase 2 Checkpoint Inspection Summary

Phase 2 added a checkpoint inspection script and generated local reports from
the real OpenPOM checkpoint copy:

```text
checkpoints/openpom_experiments_1_checkpoint2.pt
```

## Outputs

Generated artifacts:

```text
scripts/inspect_checkpoint.py
reports/checkpoint_keys.txt
reports/checkpoint_tensor_shapes.csv
docs/phase2_checkpoint_inspection.md
```

Run command:

```bash
python scripts/inspect_checkpoint.py
```

Validation command:

```bash
python -m py_compile scripts/inspect_checkpoint.py
```

## Key Findings

The checkpoint is a wrapped dictionary, not a direct state dict.

Top-level keys:

```text
model_state_dict
optimizer_state_dict
global_step
```

The model weights live in `model_state_dict`. The checkpoint also includes
optimizer state and `global_step = 1984`.

The state dict has 44 tensor entries and no extra `model.` prefix. The main
state dict prefixes are:

```text
mpnn: 11
project_edge_feats: 2
readout_set2set: 8
ffn: 23
```

These names should drive the reproduced model module hierarchy:

```text
mpnn
project_edge_feats
readout_set2set
ffn
```

## Architecture Evidence

Important tensor shapes from `reports/checkpoint_tensor_shapes.csv`:

```text
mpnn.project_node_feats.0.weight: (100, 134)
mpnn.gnn_layer.edge_func.0.weight: (75, 6)
mpnn.gnn_layer.edge_func.2.weight: (10000, 75)
project_edge_feats.0.weight: (100, 6)
readout_set2set.lstm.weight_ih_l0: (800, 400)
readout_set2set.lstm.weight_hh_l0: (800, 200)
ffn.linears.0.weight: (392, 400)
ffn.linears.1.weight: (392, 392)
ffn.linears.2.weight: (256, 392)
ffn.linears.3.weight: (138, 256)
```

Checkpoint-derived architecture values:

```text
number_atom_features = 134
number_bond_features = 6
node_out_feats = 100
edge_hidden_feats = 75
edge_out_feats = 100
Set2Set input dim = 200
Set2Set output dim = 400
ffn_hidden_list = [392, 392]
ffn_embeddings = 256
n_tasks = 138
```

## Phase 3 Note

Phase 3 should use this checkpoint inspection and the Phase 1 reference export
as the source of truth. Any older default values in `docs/architecture_spec.md`
that disagree with these checkpoint-derived values should be updated before
implementing checkpoint-compatible model loading.
