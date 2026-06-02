# Phase 4: OpenPOM-Compatible Featurizer

## Purpose

Phase 4 implemented the RDKit-to-graph feature path for this repository. The
goal was value parity with OpenPOM, not just matching tensor shapes.

This phase is critical because atom feature order, bond feature order, and edge
order all affect checkpoint-compatible inference.

## What Was Added

New package files:

```text
src/pom_repro/__init__.py
src/pom_repro/featurizer.py
src/pom_repro/graph_batch.py
```

New comparison and test files:

```text
scripts/compare_graph_features.py
tests/test_featurizer.py
```

## Featurizer Details

`src/pom_repro/featurizer.py` reproduces the OpenPOM graph featurizer without a
runtime dependency on OpenPOM or DeepChem.

Implemented public API:

```text
GraphConvConstants
atom_features(atom)
bond_features(bond)
GraphData
GraphFeaturizer
```

Feature dimensions:

```text
ATOM_FDIM = 134
BOND_FDIM = 6
```

Atom feature order:

```text
total valence
total degree
total number of hydrogens
formal charge
atomic number
hybridization
```

Important parity detail: OpenPOM encodes atomic number as
`atom.GetAtomicNum() - 1` against `range(100)`, so the reproduction does the
same.

Bond feature order:

```text
[no_bond_placeholder, is_single, is_double, is_triple, is_aromatic, is_in_ring]
```

For normal molecular bonds, `no_bond_placeholder = 0`. A missing/placeholder
bond uses:

```text
[1, 0, 0, 0, 0, 0]
```

## SMILES And Edge Order

The local `GraphFeaturizer.featurize(...)` matches the DeepChem/OpenPOM SMILES
path:

1. Parse SMILES with RDKit.
2. Apply `CanonicalRankAtoms`.
3. Apply `RenumberAtoms` with that rank list.
4. Featurize atoms and bonds in the resulting RDKit molecule order.

RDKit `Mol` inputs preserve the provided atom order, matching DeepChem behavior.

Each RDKit bond is converted into two directed edges in this order:

```text
begin_atom -> end_atom
end_atom -> begin_atom
```

The same bond feature row is duplicated for the two directed edges.

For `CCO`, the OpenPOM-compatible edge index is:

```text
[[0, 2, 2, 1],
 [2, 0, 1, 2]]
```

## Graph Batching

`src/pom_repro/graph_batch.py` adds small DGL helpers:

```text
to_dgl_graph(graph, self_loop=False)
batch_graphs(graphs, self_loop=False, device=None)
```

DGL field names match OpenPOM:

```text
g.ndata["x"]
g.edata["edge_attr"]
```

## Comparison Script

`scripts/compare_graph_features.py` compares local graph features against the
original OpenPOM implementation from:

```text
/home/xing/openpom
```

Default SMILES:

```text
C
CCO
CC(=O)O
c1ccccc1
CC(C)O
C1=CC=CC=C1O
```

The script reports per molecule:

```text
node shape
edge shape
edge_index shape
atom feature max_abs_diff
bond feature max_abs_diff
edge order identical
```

It exits nonzero if any atom feature diff, bond feature diff, or edge order
check fails.

## Tests

`tests/test_featurizer.py` covers:

- Basic shape checks for `C` and `CCO`.
- Exact `CCO` edge order.
- Single-bond feature rows.
- Aromatic ring bond rows for benzene.
- Hash parity against `reference_outputs/openpom_reference.json`.
- Optional live parity against original OpenPOM.
- DGL batched graph field names and total feature shapes.

## Validation

Commands run:

```bash
PYTHONPATH=src /home/xing/miniconda3/envs/pom_openpom_ref/bin/python -m py_compile src/pom_repro/featurizer.py src/pom_repro/graph_batch.py scripts/compare_graph_features.py tests/test_featurizer.py
```

Result:

```text
passed
```

```bash
PYTHONPATH=src /home/xing/miniconda3/envs/pom_openpom_ref/bin/python -m pytest tests/test_featurizer.py
```

Result:

```text
6 passed
```

```bash
PYTHONPATH=src /home/xing/miniconda3/envs/pom_openpom_ref/bin/python scripts/compare_graph_features.py
```

Result:

```text
Overall result: PASS
```

For all six default SMILES, the live OpenPOM comparison reported:

```text
atom feature max_abs_diff = 0
bond feature max_abs_diff = 0
edge order identical = True
```

## Notes For Later Phases

The featurizer and graph batching path should now be treated as the source of
truth for local model input. If later checkpoint or output parity fails, debug
model loading and model architecture before changing feature order.
