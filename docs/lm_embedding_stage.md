# Frozen Chemical LM Embedding Stage

## Purpose

This stage exports fixed molecular embeddings from a pretrained chemical
language model. It is intentionally limited to:

```text
SMILES -> frozen MoLFormer -> embeddings .npz + manifest
```

The default model is `ibm-research/MoLFormer-XL-both-10pct`, and the default
SMILES column is `nonStereoSMILES`.

## In Scope

- Read molecule rows from a CSV that contains `nonStereoSMILES`,
  `descriptors`, and the 138 OpenPOM label columns.
- Tokenize SMILES with the pretrained model tokenizer.
- Run the pretrained MoLFormer model in eval mode with gradients disabled.
- Pool hidden states with `pooler_output` by default.
- Write frozen embeddings to an `.npz` artifact.
- Write a manifest that records the input file, config, model id, pooling mode,
  row count, embedding dimension, and output artifact path.

## Out Of Scope

- No MLP head is trained in this stage.
- No OpenPOM checkpoint is loaded.
- No graph featurizer, MPNNPOM model, output parity script, or reference output
  is changed by this stage.
- No prediction probabilities or classification logits are produced.

## Configuration Contract

The default configuration lives at:

```text
configs/molformer_embedding.json
```

It pins the pretrained model, input column names, explicit 138-label order,
pooling strategy, and frozen/no-head behavior:

```text
model_name_or_path = ibm-research/MoLFormer-XL-both-10pct
smiles_column = nonStereoSMILES
descriptor_column = descriptors
pooling = pooler_output
freeze_model = true
train_mlp_head = false
```

The explicit `label_columns` list is metadata for reproducibility and later
alignment. The embedding export itself should not train on those labels.

## Artifact Contract

The `.npz` artifact should contain the exported embedding matrix and enough row
identity data to map embeddings back to the input CSV rows. The companion
manifest should describe how the artifact was produced, including the config,
model, pooling mode, and output paths.

These artifacts are generated data and should not be committed.
