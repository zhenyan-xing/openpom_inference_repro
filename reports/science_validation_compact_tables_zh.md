# Science Validation 子集模型质量比较报告

## 结果概述

我们在 Science POM 论文 prospective validation 补充材料中，选取当前能够同时获得 SMILES、human panel descriptor profile 和 paper-provided GNN prediction profile 的分子子集，对复现的 OpenPOM-compatible inference model 进行了模型质量比较。该可比较子集包含 207 个分子；如果进一步排除 Data S1 中带有 disqualification reason 的样本，则 clean subset 包含 163 个分子。比较限制在 Science validation 使用的 55 个气味描述符上，这 55 个描述符全部可以映射到 OpenPOM 的 138-label 输出空间。

在 clean subset 上，我们的 single-checkpoint 复现模型与 human panel profile 的总体相关性为 Pearson r = 0.523、Spearman rho = 0.431，说明当前模型输出与人类评分之间存在中等程度的一致性。作为参照，论文补充材料中提供的 GNN prediction profile 与同一 human panel profile 的相关性更高，Pearson r = 0.601、Spearman rho = 0.625。我们的输出与 paper-provided GNN prediction profile 之间的相关性为 Pearson r = 0.623、Spearman rho = 0.591，说明复现模型捕捉到了与论文 GNN 相近的 odor-profile 结构，但在该子集上尚未达到 paper GNN baseline。

需要注意的是，这不是完整 400 分子的 Science benchmark，也不是 10-model ensemble 评估；当前结果基于可用 SMILES 子集和一个 OpenPOM checkpoint。

## 表 1. 所有 sample-label 条目上的总体一致性

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

## 表 4. Clean subset 上描述符层面的最好和最差结果

| 比较对象 | Pearson 最高的描述符 | Pearson 最低的描述符 |
| --- | --- | --- |
| 我们 vs human panel | Alcoholic (0.680); Roasted (0.657); Apple (0.637); Earthy (0.620); Spicy (0.615) | Cucumber (-0.019); Cooling (0.052); Grassy (0.094); Musk (0.109); Citrus (0.192) |
| Paper GNN vs human panel | Caramellic (0.749); Roasted (0.744); Fruity (0.743); Tomato (0.736); Winey (0.718) | Cucumber (0.015); Lemon (0.263); Grassy (0.273); Musk (0.296); Citrus (0.307) |
| 我们 vs Paper GNN | Waxy (0.782); Roasted (0.766); Lemon (0.749); Sulfurous (0.729); Apple (0.714) | Cooling (0.270); Grassy (0.280); Musk (0.295); Cucumber (0.298); Orange (0.316) |

## 可用于报告中的描述

基于当前可用的 Science POM validation supplement 子集，我们将复现的 OpenPOM single-checkpoint inference 输出与 human panel descriptor profiles 以及论文补充材料中提供的 GNN prediction profiles 进行了比较。在 207 个具有 SMILES 和两类参考 profile 的分子中，复现模型与 human panel 的总体 Pearson 相关系数为 0.517，Spearman 相关系数为 0.430。排除带有 disqualification 备注的样本后，clean subset 包含 163 个分子，结果基本一致：Pearson r = 0.523，Spearman rho = 0.431。

作为参照，paper-provided GNN prediction profile 在同一 clean subset 上与 human panel 的相关性为 Pearson r = 0.601、Spearman rho = 0.625。复现模型与 paper-provided GNN profile 之间的相关性为 Pearson r = 0.623、Spearman rho = 0.591。这说明当前复现的 inference pipeline 在可用 validation subset 上能够捕捉与论文 GNN 相近的气味 profile 结构，但由于当前只使用 single checkpoint，且模型来源为 OpenPOM checkpoint 而非 Science 原闭源模型，因此整体表现低于 paper GNN baseline。

## 限制说明

- 当前分析只使用 supplement 中有 SMILES 的子集，不是完整 400 分子的 Science prospective validation benchmark。
- 当前模型是 OpenPOM-compatible single-checkpoint inference model，不是 10-model ensemble。
- 当前比较对象是 Science validation 的 55 个气味描述符，而模型完整输出为 OpenPOM 的 138 个标签。
- `Paper GNN` 指补充材料 Data S5 中提供的 GNN prediction profile，不是我们重新训练得到的模型。
