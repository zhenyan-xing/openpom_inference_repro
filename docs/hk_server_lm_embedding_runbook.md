# Hong Kong Server LM Embedding Runbook

本 runbook 用于把本地代码推到 GitHub 后，在香港服务器上完成
pretrained chemical LM 缓存和 OpenPOM frozen embedding 导出。

本阶段只产出 frozen LM embeddings 和 manifest，不训练 MLP head，不加载
OpenPOM checkpoint，也不生成最终 odor 分类概率。

## 1. 本地推送代码到 GitHub

在本地代码目录执行：

```bash
cd /home/xing/pom_inference_repro
git status --short
```

大文件和生成产物不要进入 git：

```text
models/
artifacts/
outputs/
*.npz
*.pt
*.safetensors
```

这些模式已经应该由 `.gitignore` 覆盖。提交前可以检查是否被忽略：

```bash
git check-ignore -v models/ artifacts/ outputs/ '*.npz' '*.pt' '*.safetensors'
```

只提交代码、配置、文档和轻量依赖文件：

```bash
git add README.md docs/ scripts/ src/ configs/ requirements-lm.txt .gitignore
git status --short
git commit -m "Add LM embedding export runbook"
```

如果还没有设置 GitHub remote：

```bash
git remote add origin git@github.com:<USER>/<REPO>.git
git branch -M main
```

推送：

```bash
git push -u origin main
```

## 2. 香港服务器 clone repo 并创建环境

登录服务器后，建议把代码放在 `/data/pom/pom_inference_repro`：

```bash
ssh <HK_USER>@<HK_HOST>

mkdir -p /data/pom
cd /data/pom
git clone git@github.com:<USER>/<REPO>.git pom_inference_repro
cd /data/pom/pom_inference_repro
```

创建 conda 环境并安装 LM 阶段依赖：

```bash
conda create -n pom_lm python=3.10 -y
conda activate pom_lm

python -m pip install -U pip
python -m pip install -r requirements-lm.txt
```

## 3. 上传 OpenPOM curated CSV

在本地执行，把 curated CSV 上传到服务器数据目录：

```bash
ssh <HK_USER>@<HK_HOST> "mkdir -p /data/pom/openpom/curated_datasets"

rsync -avP \
  /home/xing/openpom/openpom/data/curated_datasets/curated_GS_LF_merged_4983.csv \
  <HK_USER>@<HK_HOST>:/data/pom/openpom/curated_datasets/
```

服务器上的输入 CSV 路径应为：

```text
/data/pom/openpom/curated_datasets/curated_GS_LF_merged_4983.csv
```

## 4. 服务器能联网：直接缓存 pretrained LM

在香港服务器的 repo 目录中执行：

```bash
cd /data/pom/pom_inference_repro
conda activate pom_lm

python scripts/cache_pretrained_lm.py \
  --model-id ibm-research/MoLFormer-XL-both-10pct \
  --output-dir models/MoLFormer-XL-both-10pct
```

缓存完成后，目录应包含 tokenizer、model 权重和 `cache_manifest.json`。

## 5. 服务器不能联网：本地下载后 rsync

在能访问 Hugging Face 的本地机器执行：

```bash
cd /home/xing/pom_inference_repro
python -m pip install -U huggingface_hub

huggingface-cli download \
  ibm-research/MoLFormer-XL-both-10pct \
  --local-dir models/MoLFormer-XL-both-10pct
```

如果本地安装的是新版 Hugging Face CLI，也可以用等价命令：

```bash
hf download \
  ibm-research/MoLFormer-XL-both-10pct \
  --local-dir models/MoLFormer-XL-both-10pct
```

再把本地缓存同步到服务器 repo 下：

```bash
ssh <HK_USER>@<HK_HOST> "mkdir -p /data/pom/pom_inference_repro/models"

rsync -avP \
  /home/xing/pom_inference_repro/models/MoLFormer-XL-both-10pct \
  <HK_USER>@<HK_HOST>:/data/pom/pom_inference_repro/models/
```

## 6. 导出 frozen LM embeddings

在香港服务器执行完整导出命令：

```bash
cd /data/pom/pom_inference_repro
conda activate pom_lm

mkdir -p /data/pom/artifacts/lm_embeddings

python scripts/export_lm_embeddings.py \
  --data /data/pom/openpom/curated_datasets/curated_GS_LF_merged_4983.csv \
  --model-path models/MoLFormer-XL-both-10pct \
  --model-id ibm-research/MoLFormer-XL-both-10pct \
  --output /data/pom/artifacts/lm_embeddings/openpom_4983_molformer.npz \
  --manifest /data/pom/artifacts/lm_embeddings/openpom_4983_molformer_manifest.json \
  --batch-size 64 \
  --device cuda
```

如果服务器没有 GPU，把最后一行改为：

```bash
  --device cpu
```

## 7. 检查产物

如果产物写在 `/data/pom/artifacts`：

```bash
ls -lh /data/pom/artifacts
ls -lh /data/pom/artifacts/lm_embeddings
```

如果产物写在 repo 内默认 `artifacts`：

```bash
ls -lh artifacts
ls -lh artifacts/lm_embeddings
```

读取 `.npz` 并打印 embedding 与 label 形状：

```bash
python -c "import numpy as np; p='/data/pom/artifacts/lm_embeddings/openpom_4983_molformer.npz'; d=np.load(p, allow_pickle=False); print('embeddings shape:', d['embeddings'].shape); print('labels shape:', d['labels'].shape)"
```

对 4983 行 OpenPOM curated CSV，`labels shape` 应为 `(4983, 138)`；
`embeddings shape` 的第一维也应为 `4983`，第二维为 MoLFormer embedding 维度。

同时检查 manifest：

```bash
python -m json.tool /data/pom/artifacts/lm_embeddings/openpom_4983_molformer_manifest.json | head -80
```

确认 manifest 中 `train_mlp_head` 不会出现为 true；本阶段只做 frozen
pretrained LM inference，不训练任何 MLP head。
