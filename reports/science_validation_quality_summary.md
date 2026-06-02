# Science POM Validation Subset Quality Report

## Scope

This report uses only the supplement rows that currently have SMILES,
human panel descriptor profiles, and paper-provided GNN prediction
profiles. It is not a full 400-molecule Science benchmark and it is
not a 10-model ensemble evaluation.

Current model under test:

- checkpoint: `/home/xing/pom_inference_repro/checkpoints/openpom_experiments_1_checkpoint2.pt`
- inference: local OpenPOM-compatible single-checkpoint model
- label space compared: 55 Science validation descriptors mapped into the 138 OpenPOM labels

## Dataset Counts

- data_s1_samples: 400
- data_s4_samples: 397
- data_s5_samples: 397
- data_s7_smiles_samples: 209
- all_overlap_samples: 207
- clean_overlap_samples: 163
- science_labels: 55
- mapped_labels: 55

## Main Metrics

### all_overlap

Samples: 207; labels: 55

| Pair | Global Pearson | Global Spearman | Global Cosine | Top-5 overlap | Top-10 overlap | Median per-sample Pearson |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ours_vs_human | 0.5170 | 0.4298 | 0.6867 | 0.3971 | 0.4739 | 0.5453 |
| paper_gnn_vs_human | 0.5975 | 0.6231 | 0.6962 | 0.4483 | 0.5691 | 0.6066 |
| ours_vs_paper_gnn | 0.6209 | 0.5750 | 0.7189 | 0.4725 | 0.5440 | 0.6589 |

### clean_overlap

Samples: 163; labels: 55

| Pair | Global Pearson | Global Spearman | Global Cosine | Top-5 overlap | Top-10 overlap | Median per-sample Pearson |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ours_vs_human | 0.5231 | 0.4314 | 0.6897 | 0.4037 | 0.4785 | 0.5483 |
| paper_gnn_vs_human | 0.6009 | 0.6253 | 0.6996 | 0.4466 | 0.5730 | 0.6052 |
| ours_vs_paper_gnn | 0.6228 | 0.5909 | 0.7220 | 0.4675 | 0.5491 | 0.6575 |

## Interpretation Notes

- `ours_vs_human` is the current OpenPOM single-checkpoint inference quality on the available supplement subset.
- `paper_gnn_vs_human` is the paper-provided GNN prediction profile against the same human panel profiles, useful as a reference baseline.
- `ours_vs_paper_gnn` shows how similar our OpenPOM checkpoint behavior is to the paper-provided GNN prediction profiles on this subset.
- Differences can come from using OpenPOM rather than the original closed Science model, single checkpoint rather than ensemble, training set differences, label normalization, and the subset restriction to rows with SMILES.
