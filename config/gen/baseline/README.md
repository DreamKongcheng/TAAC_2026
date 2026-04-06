# Baseline

这是当前仓库内置的 unified baseline，也是 `config/gen` 下最基础的目录式实验包。

## 当前定位

这个 baseline 不追求逐字节复刻外部仓库，而是作为当前工作区里的默认对照：

1. 统一读取 sample parquet。
2. 统一输出当前 batch contract。
3. 用单个可训练 backbone 验证训练、评估、checkpoint 与 latency 统计链路。

## 结构概览

当前 baseline 主要做了下面几件事：

1. 把 user、context、candidate、history 都编码到同一隐藏维度。
2. 将 history_post、history_author、history_action、history_time_gap、history_group_ids 一起用于历史事件表示。
3. 把 user / candidate / history 压成统一的融合表示，再通过一个轻量 MLP 输出 CTR logit。
4. 训练目标使用 BCE + AdamW，优先保证链路简单和可复核。

## 目录内容

1. `__init__.py`：默认配置与 `EXPERIMENT` 装配。
2. `data.py`：私有 parquet 数据管线。
3. `model.py`：baseline 主体模型。
4. `utils.py`：loss / optimizer 装配。

## 默认配置

1. 默认数据集：`data/datasets--TAAC2026--data_sample_1000/.../sample_data.parquet`
2. 默认输出目录：`outputs/gen/baseline`
3. 默认训练轮数：10
4. 默认 batch size：64

## 运行方式

```bash
uv run taac-train --experiment config/gen/baseline
uv run taac-evaluate single --experiment config/gen/baseline
```

如果只想做一轮 smoke，可以覆写输出目录：

```bash
uv run taac-train --experiment config/gen/baseline --run-dir outputs/smoke/baseline_manual
```

## 当前验证状态

当前可直接复核的证据：

1. sample smoke summary：`outputs/smoke/baseline/summary.json`
2. 训练栈总回归：`tests/`

当前工作区里的 10-epoch sample smoke 指标为：

1. 最佳 epoch：10
2. AUC：0.6979
3. PR-AUC：0.2908
4. 平均时延：0.7616 ms / sample

这些数字只说明 baseline 当前链路可跑，不代表正式赛题上的最终水平。
