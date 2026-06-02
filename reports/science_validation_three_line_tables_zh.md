# Science Validation 子集三线表汇总

## 表 1. 可用 Science validation 子集上的总体一致性

| 子集 | 比较对象 | 分子数 | 标签数 | Pearson r | Spearman rho | Cosine | MAE | RMSE | Top-5 overlap | Top-10 overlap |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| All overlap | 我们 vs human panel | 207 | 55 | 0.517 | 0.430 | 0.687 | 0.108 | 0.156 | 0.397 | 0.474 |
| All overlap | Paper GNN vs human panel | 207 | 55 | 0.598 | 0.623 | 0.696 | 0.061 | 0.124 | 0.448 | 0.569 |
| All overlap | 我们 vs Paper GNN | 207 | 55 | 0.621 | 0.575 | 0.719 | 0.098 | 0.141 | 0.473 | 0.544 |
| Clean overlap | 我们 vs human panel | 163 | 55 | 0.523 | 0.431 | 0.690 | 0.105 | 0.153 | 0.404 | 0.479 |
| Clean overlap | Paper GNN vs human panel | 163 | 55 | 0.601 | 0.625 | 0.700 | 0.059 | 0.122 | 0.447 | 0.573 |
| Clean overlap | 我们 vs Paper GNN | 163 | 55 | 0.623 | 0.591 | 0.722 | 0.097 | 0.140 | 0.468 | 0.549 |

注：Top-k overlap 表示两个 top-k 气味标签列表之间的平均重合比例。

## 表 2. Clean subset 上逐分子 odor-profile 一致性

| 比较对象 | Mean Pearson | Median Pearson | IQR Pearson | Mean Spearman | Median Spearman | Mean Cosine | Median Cosine |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 我们 vs human panel | 0.504 | 0.548 | 0.378-0.680 | 0.461 | 0.472 | 0.685 | 0.723 |
| Paper GNN vs human panel | 0.597 | 0.605 | 0.477-0.736 | 0.645 | 0.659 | 0.705 | 0.720 |
| 我们 vs Paper GNN | 0.608 | 0.657 | 0.459-0.773 | 0.607 | 0.630 | 0.722 | 0.745 |

## 表 3. Clean subset 上逐气味描述符一致性

| 比较对象 | Mean Pearson | Median Pearson | IQR Pearson | Mean Spearman | Median Spearman | Mean Cosine | Median Cosine |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 我们 vs human panel | 0.422 | 0.448 | 0.324-0.544 | 0.330 | 0.320 | 0.705 | 0.721 |
| Paper GNN vs human panel | 0.519 | 0.520 | 0.414-0.636 | 0.443 | 0.468 | 0.656 | 0.676 |
| 我们 vs Paper GNN | 0.556 | 0.578 | 0.477-0.652 | 0.501 | 0.516 | 0.679 | 0.687 |

## 表 4. Clean subset 上各比较中表现最好和最差的气味描述符

| 比较对象 | Pearson 最高的描述符 | Pearson 最低的描述符 |
| --- | --- | --- |
| 我们 vs human panel | Alcoholic (0.680); Roasted (0.657); Apple (0.637); Earthy (0.620); Spicy (0.615) | Cucumber (-0.019); Cooling (0.052); Grassy (0.094); Musk (0.109); Citrus (0.192) |
| Paper GNN vs human panel | Caramellic (0.749); Roasted (0.744); Fruity (0.743); Tomato (0.736); Winey (0.718) | Cucumber (0.015); Lemon (0.263); Grassy (0.273); Musk (0.296); Citrus (0.307) |
| 我们 vs Paper GNN | Waxy (0.782); Roasted (0.766); Lemon (0.749); Sulfurous (0.729); Apple (0.714) | Cooling (0.270); Grassy (0.280); Musk (0.295); Cucumber (0.298); Orange (0.316) |

## 表注

这是一个基于当前可用数据的 Science validation 子集分析。`All overlap` 包含同时具有 SMILES、human panel descriptor profile 和 paper GNN prediction profile 的分子；`Clean overlap` 进一步排除了 Data S1 中带有 disqualification 备注的样本。当前模型是我们复现的 OpenPOM-compatible single-checkpoint inference model，不是 10-model ensemble。
