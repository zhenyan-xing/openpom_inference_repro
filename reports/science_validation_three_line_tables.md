# Science Validation Three-Line Tables

## Table 1. Overall Agreement On Available Science Validation Subsets

| Subset | Comparison | n molecules | n labels | Pearson r | Spearman rho | Cosine | MAE | RMSE | Top-5 overlap | Top-10 overlap |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| All overlap | Ours vs human | 207 | 55 | 0.517 | 0.430 | 0.687 | 0.108 | 0.156 | 0.397 | 0.474 |
| All overlap | Paper GNN vs human | 207 | 55 | 0.598 | 0.623 | 0.696 | 0.061 | 0.124 | 0.448 | 0.569 |
| All overlap | Ours vs paper GNN | 207 | 55 | 0.621 | 0.575 | 0.719 | 0.098 | 0.141 | 0.473 | 0.544 |
| Clean overlap | Ours vs human | 163 | 55 | 0.523 | 0.431 | 0.690 | 0.105 | 0.153 | 0.404 | 0.479 |
| Clean overlap | Paper GNN vs human | 163 | 55 | 0.601 | 0.625 | 0.700 | 0.059 | 0.122 | 0.447 | 0.573 |
| Clean overlap | Ours vs paper GNN | 163 | 55 | 0.623 | 0.591 | 0.722 | 0.097 | 0.140 | 0.468 | 0.549 |

Note: Top-k overlap is the mean fraction of shared labels between the two top-k descriptor lists.

## Table 2. Per-Molecule Odor-Profile Agreement On Clean Subset

| Comparison | Mean Pearson | Median Pearson | IQR Pearson | Mean Spearman | Median Spearman | Mean Cosine | Median Cosine |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Ours vs human | 0.504 | 0.548 | 0.378-0.680 | 0.461 | 0.472 | 0.685 | 0.723 |
| Paper GNN vs human | 0.597 | 0.605 | 0.477-0.736 | 0.645 | 0.659 | 0.705 | 0.720 |
| Ours vs paper GNN | 0.608 | 0.657 | 0.459-0.773 | 0.607 | 0.630 | 0.722 | 0.745 |

## Table 3. Per-Descriptor Agreement Across Molecules On Clean Subset

| Comparison | Mean Pearson | Median Pearson | IQR Pearson | Mean Spearman | Median Spearman | Mean Cosine | Median Cosine |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Ours vs human | 0.422 | 0.448 | 0.324-0.544 | 0.330 | 0.320 | 0.705 | 0.721 |
| Paper GNN vs human | 0.519 | 0.520 | 0.414-0.636 | 0.443 | 0.468 | 0.656 | 0.676 |
| Ours vs paper GNN | 0.556 | 0.578 | 0.477-0.652 | 0.501 | 0.516 | 0.679 | 0.687 |

## Table 4. Descriptor-Level Extremes On Clean Subset

| Comparison | Highest Pearson descriptors | Lowest Pearson descriptors |
| --- | --- | --- |
| Ours vs human | Alcoholic (0.680); Roasted (0.657); Apple (0.637); Earthy (0.620); Spicy (0.615) | Cucumber (-0.019); Cooling (0.052); Grassy (0.094); Musk (0.109); Citrus (0.192) |
| Paper GNN vs human | Caramellic (0.749); Roasted (0.744); Fruity (0.743); Tomato (0.736); Winey (0.718) | Cucumber (0.015); Lemon (0.263); Grassy (0.273); Musk (0.296); Citrus (0.307) |
| Ours vs paper GNN | Waxy (0.782); Roasted (0.766); Lemon (0.749); Sulfurous (0.729); Apple (0.714) | Cooling (0.270); Grassy (0.280); Musk (0.295); Cucumber (0.298); Orange (0.316) |

## Caption

Available-data Science validation subset analysis. `All overlap` includes molecules with SMILES, human panel descriptor profiles, and paper GNN prediction profiles. `Clean overlap` further excludes samples with non-empty disqualification notes in Data S1. The current model is the reproduced OpenPOM-compatible single-checkpoint inference model, not a 10-model ensemble.
