# Science POM Validation Subset Quality Report

## Scope

This report uses only the supplement rows that currently have SMILES,
human panel descriptor profiles, and paper-provided GNN prediction
profiles. It is not a full 400-molecule Science benchmark and it is
not a 10-model ensemble evaluation.

Current model under test:

- checkpoint: `/workspace/hedongchen/xing_work/data/pom/outputs/molformer_head/head.pt`
- inference: local frozen-LM-head inference
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
| ours_vs_human | 0.4845 | 0.6264 | 0.5397 | 0.5130 | 0.6130 | 0.5030 |
| paper_gnn_vs_human | 0.5975 | 0.6231 | 0.6962 | 0.4483 | 0.5691 | 0.6066 |
| ours_vs_paper_gnn | 0.5251 | 0.6454 | 0.5741 | 0.4473 | 0.5623 | 0.5106 |

### clean_overlap

Samples: 163; labels: 55

| Pair | Global Pearson | Global Spearman | Global Cosine | Top-5 overlap | Top-10 overlap | Median per-sample Pearson |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ours_vs_human | 0.4904 | 0.6239 | 0.5465 | 0.5166 | 0.6117 | 0.5165 |
| paper_gnn_vs_human | 0.6009 | 0.6253 | 0.6996 | 0.4466 | 0.5730 | 0.6052 |
| ours_vs_paper_gnn | 0.5278 | 0.6413 | 0.5781 | 0.4515 | 0.5589 | 0.5263 |

## Interpretation Notes

- `ours_vs_human` is the current local frozen-LM-head inference quality on the available supplement subset.
- `paper_gnn_vs_human` is the paper-provided GNN prediction profile against the same human panel profiles, useful as a reference baseline.
- `ours_vs_paper_gnn` shows how similar our predictions are to the paper-provided GNN prediction profiles on this subset.
- Differences can come from using OpenPOM rather than the original closed Science model, single checkpoint rather than ensemble, training set differences, label normalization, and the subset restriction to rows with SMILES.
