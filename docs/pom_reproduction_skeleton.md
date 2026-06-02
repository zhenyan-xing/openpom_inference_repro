# POM / MPNNPOM 复现骨架备忘录

> Status: legacy broad-scope memo. This file preserves early architecture notes
> and training-oriented context. For the current checkpoint-compatible
> implementation target, use `docs/architecture_spec.md`,
> `docs/checkpoint_compatibility.md`, and `reports/checkpoint_keys.txt` as the
> authoritative sources. In particular, the current checkpoint-derived model
> uses `node_out_feats=100`, `edge_hidden_feats=75`,
> `edge_out_feats=100`, `num_step_message_passing=5`,
> Set2Set `n_iters=3`, `n_layers=2`, and
> `ffn_hidden_list=[392, 392]`.

本文档用于回答“复现 POM 模型核心时，骨架层面必须先把握什么”。结论先放前面：

复现的健康顺序是先建立最小可运行闭环，再逐层对齐细节：

```text
SMILES / CSV
  -> molecule standardization / RDKit Mol
  -> molecular graph
  -> atom features + bond features
  -> batched DGL graph
  -> MPNN message passing
  -> graph-level readout
  -> FFN / POM embedding
  -> multilabel odor prediction
  -> loss + metric + smoke tests
```

这 12 个问题是正确的复现骨架问题。建议额外补充：数据集与标签定义、数据划分/交叉验证、metric、class imbalance、checkpoint/ensemble、随机种子与版本依赖。

## 模型对象

- 论文模型：Principal Odor Map, POM，用分子结构预测人类嗅觉感知标签。
- 当前仓库实现：`MPNNPOM` / `MPNNPOMModel`，作为 OpenPOM 对 Lee et al. 2023 POM 论文的开源复现。
- 公开基准：README 和 GitHub 页面写的是 `curated_GS_LF_merged_4983.csv` 上 5-fold CV，每折 10-model ensemble，ROC-AUC 约 0.8872。
- 当前本地数据：`openpom/data/curated_datasets/curated_GS_LF_merged_4983.csv`，140 列，其中前两列为 `nonStereoSMILES`、`descriptors`，后面 138 列为 odor labels；本地行数为 4,983 个分子。

## 1. 输入是什么？

训练/推理层面的输入是分子 SMILES。

当前训练脚本使用：

- 数据文件：`openpom/data/curated_datasets/curated_GS_LF_merged_4983.csv`
- SMILES 字段：`nonStereoSMILES`
- 标签字段：138 个 odor descriptor 列

模型 `forward` 层面的直接输入不是字符串，而是 DGL batched graph。该图需要有：

- node feature: `g.ndata["x"]`
- edge feature: `g.edata["edge_attr"]`

源码依据：

- `openpom/hyper/scripts/mpnnpom_random_cv.py:19-43`
- `openpom/hyper/scripts/mpnnpom_random_cv.py:198-214`
- `openpom/models/mpnn_pom.py:79-80`
- `openpom/models/mpnn_pom.py:310-311`

## 2. 输出是什么？

分类模式下，输出是三元组：

```text
proba, logits, embeddings
```

其中：

- `proba`: 每个分子、每个 odor label 的预测概率。
- `logits`: FFN 最后一层输出 reshape 后的张量。
- `embeddings`: FFN 的倒数第二层输出，也就是 POM embedding。

在当前主任务中：

```text
batch_size x 138
```

因为 `n_tasks = 138`，`n_classes = 1`，所以 `proba` 最终 squeeze 为 `(batch_size, n_tasks)`。

源码依据：

- `openpom/models/mpnn_pom.py:174-177`
- `openpom/models/mpnn_pom.py:320-333`
- `openpom/models/test/test_mpnn_pom.py:141-187`

## 3. SMILES 如何变成 graph？

流程是：

```text
SMILES
  -> RDKit Mol
  -> atom feature matrix
  -> directed bond edge_index
  -> edge feature matrix
  -> DeepChem GraphData
  -> DGLGraph
  -> dgl.batch(...)
```

`GraphFeaturizer` 继承自 DeepChem 的 `MolecularFeaturizer`。它对每个 RDKit atom 计算 atom features，对每个 RDKit bond 计算 bond features。每条化学键会变成两条有向边：`start -> end` 和 `end -> start`。

单原子分子没有 bond 时，edge index 为空，edge feature shape 是 `(0, 6)`。

源码依据：

- `openpom/feat/graph_featurizer.py:113-236`
- `openpom/feat/graph_featurizer.py:167-188`
- `openpom/models/mpnn_pom.py:569-601`
- `openpom/models/test/test_mpnn_pom.py:16-20`

## 4. atom feature 有多少维？

atom feature 是 134 维。

组成包括：

- total valence one-hot: 0-6，外加 unknown bucket
- total degree one-hot: 0-5，外加 unknown bucket
- total number of hydrogens one-hot: 0-4，外加 unknown bucket
- formal charge one-hot: -1, -2, 1, 2, 0，外加 unknown bucket
- atomic number one-hot: 0-99，外加 unknown bucket
- hybridization one-hot: SP, SP2, SP3, SP3D, SP3D2，外加 unknown bucket

源码中通过：

```python
ATOM_FDIM = sum(len(choices) + 1 for choices in ATOM_FEATURES.values()) \
            + len(ATOM_FEATURES_HYBRIDIZATION) + 1
```

算出 134。

源码依据：

- `openpom/feat/graph_featurizer.py:21-43`
- `openpom/feat/graph_featurizer.py:46-79`
- `openpom/feat/graph_featurizer.py:122-131`
- `openpom/models/mpnn_pom.py:76`

## 5. bond feature 有多少维？

bond feature 是 6 维。

组成是：

```text
[no_bond_placeholder, is_single, is_double, is_triple, is_aromatic, is_in_ring]
```

对于真实 bond，第一维是 0。对于 `bond is None` 的占位情况，第一维是 1，其余为 0。

源码依据：

- `openpom/feat/graph_featurizer.py:43`
- `openpom/feat/graph_featurizer.py:82-110`
- `openpom/feat/graph_featurizer.py:133-139`
- `openpom/models/mpnn_pom.py:77`

## 6. GNN 用什么 message passing？

核心是 MPNN，具体实现基于 DGL-LifeSci 的 `MPNNGNN`，并把内部 GNN layer 替换为 DGL 的 `NNConv`。

默认配置：

- `node_out_feats = 64`
- `edge_hidden_feats = 128`
- `num_step_message_passing = 3`
- `mpnn_residual = True`
- `message_aggregator_type = "sum"`

可选 aggregator 包括 `sum`、`mean`、`max`，但超参数搜索里主要列了 `sum` 和 `mean`。

源码依据：

- `openpom/models/mpnn_pom.py:67-88`
- `openpom/models/mpnn_pom.py:179-186`
- `openpom/layers/pom_mpnn_gnn.py:6-61`
- `openpom/hyper/configs/model_configs.py:16-41`

## 7. edge feature 如何参与 message passing？

有两处参与方式，需要分清：

第一处，在 MPNN/NNConv message passing 中，edge feature 进入 edge network：

```text
edge feature, dim 6
  -> Linear(6, edge_hidden_feats)
  -> ReLU
  -> Linear(edge_hidden_feats, node_out_feats * node_out_feats)
  -> edge-conditioned message transform
```

也就是说，bond feature 决定消息传递时用于变换 node hidden state 的 edge-specific weight matrix。

第二处，在 readout 前，原始 edge feature 被投影到 `edge_out_feats`，然后与 source node embedding 拼接：

```text
project_edge_feats: Linear(6, edge_out_feats) + ReLU
src_msg = concat(source_node_embedding, edge_embedding)
```

然后按目标节点 sum 聚合，形成 node-level 的 atom+bond combined representation。

源码依据：

- `openpom/layers/pom_mpnn_gnn.py:54-61`
- `openpom/models/mpnn_pom.py:188-189`
- `openpom/models/mpnn_pom.py:247-271`

## 8. graph-level readout 用什么？

默认 readout 是 `set2set`。

默认参数：

- `readout_type = "set2set"`
- `num_step_set2set = 6`
- `num_layer_set2set = 3`

如果用 `set2set`，输入维度是：

```text
node_out_feats + edge_out_feats
```

默认是：

```text
64 + 64 = 128
```

Set2Set 输出维度是输入的 2 倍，因此默认进入 FFN 的维度是：

```text
2 * 128 = 256
```

另一个可选 readout 是 `global_sum_pooling`，但默认不是它。

源码依据：

- `openpom/models/mpnn_pom.py:81-83`
- `openpom/models/mpnn_pom.py:191-200`
- `openpom/models/mpnn_pom.py:273-280`

## 9. POM embedding 在哪里？

POM embedding 是 FFN 的倒数第二层输出。

默认设置：

```text
ffn_hidden_list = [300]
ffn_embeddings = 256
```

模型构造时会把它拼成：

```text
d_hidden_list = [300, 256]
```

FFN 最后一层前的 256 维向量就是 embedding。分类 forward 返回：

```text
return proba, logits, embeddings
```

源码依据：

- `openpom/models/mpnn_pom.py:84-85`
- `openpom/models/mpnn_pom.py:202-211`
- `openpom/layers/pom_ffn.py:6-14`
- `openpom/layers/pom_ffn.py:139-151`
- `openpom/models/mpnn_pom.py:320-333`

## 10. 最后一层如何做 multilabel classification？

这是多标签二分类，不是单标签多分类。

最后一层输出维度：

```text
n_tasks * n_classes
```

当前任务中：

```text
n_tasks = 138
n_classes = 1
ffn_output = 138
```

随后 reshape 为：

```text
logits: batch_size x n_tasks x 1
```

再对每个 label 独立做 sigmoid：

```text
proba = sigmoid(logits)
```

因为 `n_classes == 1`，最后 squeeze 成：

```text
batch_size x n_tasks
```

源码依据：

- `openpom/models/mpnn_pom.py:174-177`
- `openpom/models/mpnn_pom.py:205-211`
- `openpom/models/mpnn_pom.py:324-333`

## 11. loss 应该是什么？

当前分类实现使用自定义 `CustomMultiLabelLoss`。

逻辑上它是每个 task 的二分类交叉熵，然后按 task 汇总。class imbalance 时会乘以：

```text
log(1 + class_imbalance_ratio)
```

需要注意一个工程细节：代码里传入 loss 的 `output` 是 `output_types = ["prediction", "loss", "embedding"]` 中的 `loss`，对应模型返回的第二个值，即 `logits`。但 `CustomMultiLabelLoss` 内部变量名叫 `probabilities`，并把 `output[:, 0, :]` 与 `1 - output[:, 0, :]` 拼成二分类输入。复现时建议先忠实对齐这个实现，然后再单独评估是否应改成更标准的 `BCEWithLogitsLoss`。

class imbalance ratio 的本地实现是：

```text
class_counts / max_count
```

虽然 docstring 文字说的是 majority / label count，但代码实际是 label count / max count。这是一个需要记录的实现差异点。

源码依据：

- `openpom/models/mpnn_pom.py:521-529`
- `openpom/utils/loss.py:6-23`
- `openpom/utils/loss.py:61-130`
- `openpom/utils/data_utils.py:10-43`

## 12. smoke test 如何验证？

建议从低到高做四层 smoke test。

### A. Featurizer smoke test

输入几个简单 SMILES：

```text
C
CC
CC(=O)C
C1=CC=NC=C1
```

检查：

- node feature shape 是 `(num_atoms, 134)`
- edge feature shape 是 `(2 * num_bonds, 6)`
- 单原子分子 edge shape 是 `(0, 6)`
- edge index 是双向边

当前已有类似测试：

- `openpom/feat/test/test_graph_featurizer.py`

### B. Model forward smoke test

输入：

```text
["CC", "C"]
```

转成 batched DGL graph 后，测试分类输出：

```text
proba.shape == (2, n_tasks)
logits.shape == (2, n_tasks, 1)
embedding.shape == (2, ffn_embeddings)
```

当前已有类似测试：

- `openpom/models/test/test_mpnn_pom.py:95-187`

### C. Small overfit smoke test

用很小的样本训练，验证模型能过拟合。当前分类测试要求小样本上 ROC-AUC 大于 0.9。

源码依据：

- `openpom/models/test/test_mpnn_pom_model.py:11-43`

### D. Reproduction-level smoke test

真正复现时还应该加：

- 固定随机种子。
- 跑 1-fold 或 2-fold 小 epoch，确认训练 loop、metric、checkpoint 都能跑。
- 再跑完整 5-fold CV。
- 如果要对齐 README benchmark，需要做每折 10 个模型的 ensemble。

## 建议补充的复现问题

你的 12 个问题已经抓住了模型骨架。我建议再补 8 个问题：

13. 数据集是哪一个？有多少分子、多少标签、标签列顺序是否固定？
14. SMILES 是否 canonicalize？是否去盐、去重、处理中性化、保留 stereochemistry？
15. label 是 binary 还是强度值？缺失值如何处理？
16. batch collation 如何从 DeepChem GraphData 变成 DGL batched graph？
17. training/validation/test 如何划分？是否 iterative stratification 或 random stratified split？
18. metric 是什么？micro/macro ROC-AUC 如何聚合？
19. 训练是否用 ensemble？checkpoint 如何选择？
20. 复现允许误差是多少？shape 对齐、loss 下降、小样本 overfit、CV AUC 各自的验收标准是什么？

## 从 PyTorch 工程角度的复现顺序

建议顺序：

1. 写 `GraphFeaturizer` 等价实现，先过 shape test。
2. 写 `Dataset` / `collate_fn`，能产生 batched graph。
3. 写 `MPNNCore`，只保证 forward shape 正确。
4. 写 readout，先只做默认 `Set2Set`。
5. 写 FFN 和 embedding 返回。
6. 写 multilabel head，确认 `proba/logits/embedding` shape。
7. 写 loss，先忠实复现当前实现，再考虑标准 BCE 版本作 ablation。
8. 写 smoke tests：featurizer、forward、small overfit。
9. 写 train/eval loop：ROC-AUC、CV、checkpoint。
10. 最后才做超参搜索和 ensemble。

## 从 computational biology / cheminformatics 角度的注意点

不要把 featurization 当成“小函数”。对于分子 GNN，SMILES 标准化、显式/隐式氢、芳香性识别、formal charge、ring bond、stereochemistry 是否保留，都会影响 graph 和标签学习。

当前实现默认：

- 使用 `nonStereoSMILES` 字段。
- `GraphFeaturizer(is_adding_hs=False)`，默认不显式添加 H。
- atom feature 中使用 total hydrogens，来自 RDKit 的隐式氢信息。
- bond feature 包含 aromatic 与 ring flag。
- 每条 bond 变成双向 edge。

复现时应先完全跟随这些选择。等指标接近后，再讨论是否改为更规范的分子标准化流程。

## 最小骨架的伪代码

```python
class POMModel(nn.Module):
    def __init__(self, n_tasks=138):
        self.mpnn = MPNNCore(
            atom_dim=134,
            bond_dim=6,
            node_hidden=64,
            edge_hidden=128,
            message_steps=3,
        )
        self.edge_project = nn.Sequential(nn.Linear(6, 64), nn.ReLU())
        self.readout = Set2Set(input_dim=128, n_iters=6, n_layers=3)
        self.ffn = FeedForward(
            input_dim=256,
            hidden_dims=[300, 256],
            output_dim=n_tasks,
        )

    def forward(self, graph):
        node_x = graph.ndata["x"]
        edge_x = graph.edata["edge_attr"]
        node_h = self.mpnn(graph, node_x, edge_x)
        graph_h = self.radius0_readout(graph, node_h, edge_x)
        embedding, logits = self.ffn(graph_h)
        proba = torch.sigmoid(logits)
        return proba, logits.view(-1, n_tasks, 1), embedding
```

## 参考来源

本地源码：

- `README.md`
- `openpom/feat/graph_featurizer.py`
- `openpom/models/mpnn_pom.py`
- `openpom/layers/pom_mpnn_gnn.py`
- `openpom/layers/pom_ffn.py`
- `openpom/utils/loss.py`
- `openpom/utils/data_utils.py`
- `openpom/hyper/scripts/mpnnpom_random_cv.py`
- `openpom/models/test/test_mpnn_pom.py`
- `openpom/models/test/test_mpnn_pom_model.py`

外部来源：

- OpenPOM GitHub: https://github.com/BioMachineLearning/openpom
- PyPI OpenPOM: https://pypi.org/project/openpom/
- Zenodo reproduction package: https://zenodo.org/records/7992168
- Paper DOI: https://doi.org/10.1126/science.ade4401
- bioRxiv DOI: https://doi.org/10.1101/2022.09.01.504602
