# 开发文档

## 当前分支实现范围

当前分支的实现边界很明确：

1. 共享底座位于 `src/taac2026`。
2. 独立实验包位于 `config/gen`。
3. 当前真实可用的 CLI 只有 `taac-train` 与 `taac-evaluate`。
4. 当前回归入口是 `pytest tests -q`。

旧版文档里出现过的 `taac2026/experiments` 注册目录、`taac-visualize`、`taac-feature-*`、`taac-truncation-sweep` 等命令，在本分支里都没有对应实现。

## 环境准备

基于 uv 管理环境。依赖的事实来源是 `pyproject.toml` 与 `uv.lock`。

```bash
uv python install 3.14
uv sync --locked
```

Linux 环境下，`torch` 已经通过 `pyproject.toml` 的 uv source 显式固定到 PyTorch 官方 `cu128` 索引。也就是说，这里的环境同步会安装 CUDA 12.8 轮子，而不会继续跟着最新版默认 CUDA 13.0。

当前实验包已经在各自的 `__init__.py` 中写死了 sample parquet 默认路径：

```text
data/datasets--TAAC2026--data_sample_1000/snapshots/2f0ddba721a8323495e73d5229c836df5d603b39/sample_data.parquet
```

如果你要切换到别的数据集，当前 CLI 没有 `--dataset-path` 覆写。推荐做法是写一个本地 wrapper 实验包：

```python
from config.gen.oo import EXPERIMENT

EXPERIMENT = EXPERIMENT.clone()
EXPERIMENT.data.dataset_path = "/path/to/your.parquet"
EXPERIMENT.train.output_dir = "outputs/custom/oo"
```

然后把这个目录路径传给 `--experiment`。

## 训练命令

命令行可以接受模块路径，也可以直接接受实验目录路径。当前仓库内实际存在的独立实验包如下：

```bash
uv run taac-train --experiment config/gen/baseline
uv run taac-train --experiment config/gen/ctr_baseline
uv run taac-train --experiment config/gen/deepcontextnet
uv run taac-train --experiment config/gen/interformer
uv run taac-train --experiment config/gen/onetrans
uv run taac-train --experiment config/gen/hyformer
uv run taac-train --experiment config/gen/unirec
uv run taac-train --experiment config/gen/uniscaleformer
uv run taac-train --experiment config/gen/oo
```

如果只想把输出落到单独目录，使用 `--run-dir`：

```bash
uv run taac-train --experiment config/gen/oo --run-dir outputs/smoke/oo_manual
```

## 评估命令

单实验评估会默认读取实验包 `train.output_dir` 下的 `best.pt`：

```bash
uv run taac-evaluate single --experiment config/gen/baseline
uv run taac-evaluate single --experiment config/gen/oo --run-dir outputs/smoke/oo
```

也可以显式指定 checkpoint 与输出文件：

```bash
uv run taac-evaluate single \
	--experiment config/gen/interformer \
	--checkpoint outputs/smoke/interformer/best.pt \
	--output-path outputs/smoke/interformer/evaluation.json
```

批量评估当前支持：

```bash
uv run taac-evaluate batch --experiment-paths \
	config/gen/baseline \
	config/gen/ctr_baseline \
	config/gen/deepcontextnet \
	config/gen/interformer \
	config/gen/onetrans \
	config/gen/hyformer \
	config/gen/unirec \
	config/gen/uniscaleformer \
	config/gen/oo
```

注意：`batch` 模式当前不会自动忽略错误；如果某个实验缺少 `best.pt`，或者 checkpoint 与当前模型定义不兼容，命令会直接失败。

## 回归测试

最重要的回归入口是：

```bash
uv run pytest tests -q
```

这个测试文件当前覆盖：

1. 目录式实验包加载。
2. 流式 parquet 数据管线。
3. baseline / ctr_baseline / deepcontextnet / interformer / onetrans / hyformer / unirec / uniscaleformer / oo 的前向构建。
4. train / evaluate 基本闭环。
5. checkpoint 兼容性校验。

## 当前默认训练轮数

当前 `config/gen` 下九个独立实验包的默认 `epochs` 都已统一调整为 10：

1. `baseline`
2. `ctr_baseline`
3. `deepcontextnet`
4. `interformer`
5. `onetrans`
6. `hyformer`
7. `unirec`
8. `uniscaleformer`
9. `oo`

如果只是做更快的链路检查，可以像下面这样临时缩短轮数：

```python
from config.gen.oo import EXPERIMENT
from taac2026.train import run_training

experiment = EXPERIMENT.clone()
experiment.train.epochs = 1
experiment.train.output_dir = "outputs/smoke/oo_quickcheck"
run_training(experiment)
```

## 输出文件

每次训练会在输出目录下写四类主要产物：

```text
best.pt
summary.json
training_curves.json
training_curves.png
```

其中：

1. `best.pt` 保存当前最佳 epoch 的模型参数和指标。
2. `summary.json` 保存最佳 AUC、PR-AUC、Brier、logloss、latency、`model_profile`，以及 `compute_profile`。其中 `model_profile` 是单次验证前向的 profile batch size、TFLOPs/批次、每样本 FLOPs 与参数量 MB；`compute_profile` 是按单次训练步 / 验证前向 profile 放大的完整训练总算力估算。
3. `training_curves.json` 保存逐 epoch 的 train loss、val loss 与 val AUC。
4. `training_curves.png` 会在每个 epoch 结束后由 matplotlib 覆盖刷新，直观显示 train loss、val loss 和 val AUC 的变化折线。

## 当前未覆盖内容

下面这些内容不在当前分支实现范围内：

1. 正式比赛线上提交流程。
2. 官方评测环境封装。
3. 可视化、EDA、聚类分析 CLI。
4. truncation sweep / feature engineering 专用脚本入口。

如果后续这些能力重新回到主分支，文档需要基于实际代码重新补齐，而不是继续沿用旧命令说明。