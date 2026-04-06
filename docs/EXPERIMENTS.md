# 实验包与验证记录

## 用途

本文件只记录当前分支里真实存在的独立实验包，以及当前可以直接复核的验证证据。旧版文档里出现过但当前仓库里已经不存在的 `taac2026/experiments/*` 目录、旧模型家族和历史排行榜结果，都不再保留。

## 当前独立实验包

| 实验包         | 目录                                                   | 模型名             | 默认输出目录                 | 主要来源                                                                                                                                      |
| -------------- | ------------------------------------------------------ | ------------------ | ---------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| Baseline       | [config/gen/baseline](config/gen/baseline)             | `grok_baseline`    | `outputs/gen/baseline`       | 本仓库本地 unified baseline                                                                                                                   |
| CTR Baseline   | [config/gen/ctr_baseline](config/gen/ctr_baseline)     | `ctr_baseline_din` | `outputs/gen/ctr_baseline`   | [creatorwyx/TAAC2026-CTR-Baseline](https://github.com/creatorwyx/TAAC2026-CTR-Baseline)                                                       |
| DeepContextNet | [config/gen/deepcontextnet](config/gen/deepcontextnet) | `deepcontextnet`   | `outputs/gen/deepcontextnet` | [suyanli220/TAAC-2026-Baseline-Tencent-Advertisement-Contest](https://github.com/suyanli220/TAAC-2026-Baseline-Tencent-Advertisement-Contest) |
| InterFormer    | [config/gen/interformer](config/gen/interformer)       | `interformer`      | `outputs/gen/interformer`    | [InterFormer paper](https://arxiv.org/abs/2411.09852)                                                                                         |
| OneTrans       | [config/gen/onetrans](config/gen/onetrans)             | `onetrans`         | `outputs/gen/onetrans`       | [OneTrans paper](https://arxiv.org/abs/2510.26104)                                                                                            |
| HyFormer       | [config/gen/hyformer](config/gen/hyformer)             | `hyformer`         | `outputs/gen/hyformer`       | [HyFormer paper](https://arxiv.org/abs/2601.12681)                                                                                            |
| UniRec         | [config/gen/unirec](config/gen/unirec)                 | `unirec`           | `outputs/gen/unirec`         | [hojiahao/TAAC2026](https://github.com/hojiahao/TAAC2026)                                                                                     |
| UniScaleFormer | [config/gen/uniscaleformer](config/gen/uniscaleformer) | `uniscaleformer`   | `outputs/gen/uniscaleformer` | [twx145/Unirec](https://github.com/twx145/Unirec)                                                                                             |
| O_o            | [config/gen/oo](config/gen/oo)                         | `oo`               | `outputs/gen/oo`             | [salmon1802/O_o](https://github.com/salmon1802/O_o)                                                                                           |

## 当前回归基线

当前最重要的统一回归命令是：

```bash
uv run pytest tests -q
```

它覆盖目录式实验包加载、数据管线、前向构建、train / evaluate 闭环和 checkpoint 兼容性。其中 `ctr_baseline`、`deepcontextnet`、`interformer`、`onetrans`、`hyformer`、`unirec`、`uniscaleformer`、`oo` 都有单独的 package build + forward 用例。

## 当前可复核 smoke 结果

下面的指标全部来自 sample parquet 上按统一默认配置跑出的 10-epoch sample 训练，只用于说明“链路可跑”和“当前工作区里已有可复核产物”，不能拿来当作正式赛题结论。

| 实验包         | 证据                                                                                   | 最佳 epoch |    AUC | PR-AUC |  Brier | 平均时延（毫秒/样本） | P95 时延（毫秒/样本） | TFLOPs/批次 | 完整训练总 TFLOPs | 参数量（MB） | 说明                                     |
| -------------- | -------------------------------------------------------------------------------------- | ---------: | -----: | -----: | -----: | --------------------: | --------------------: | ----------: | ----------------: | -----------: | ---------------------------------------- |
| Baseline       | [outputs/smoke/baseline/summary.json](outputs/smoke/baseline/summary.json)             |          6 | 0.6763 | 0.2539 | 0.1289 |                0.7610 |                1.2856 |    0.009219 |          3.775130 |      67.5054 | 基线总算力中等，profile batch size 为 64 |
| CTR Baseline   | [outputs/smoke/ctr_baseline/summary.json](outputs/smoke/ctr_baseline/summary.json)     |          8 | 0.6478 | 0.3168 | 0.1227 |                0.3493 |                0.5939 |    0.000224 |          0.091811 |      49.0020 | 当前最轻，前向与完整训练总算力都最低     |
| DeepContextNet | [outputs/smoke/deepcontextnet/summary.json](outputs/smoke/deepcontextnet/summary.json) |          5 | 0.6206 | 0.1747 | 0.2126 |                0.1470 |                0.3082 |    0.001852 |         10.573857 |      67.5103 | 推理最轻，但因 batch size 32 总算力不低  |
| InterFormer    | [outputs/smoke/interformer/summary.json](outputs/smoke/interformer/summary.json)       |          5 | 0.6243 | 0.2309 | 0.1580 |               19.2306 |               19.5413 |    0.033679 |         13.356647 |     128.3087 | 单批次前向最重，参数量也最大             |
| OneTrans       | [outputs/smoke/onetrans/summary.json](outputs/smoke/onetrans/summary.json)             |          1 | 0.7088 | 0.2581 | 0.2054 |               10.1906 |               10.6896 |    0.016967 |          6.944763 |      96.7710 | 高 AUC 组里算力中等偏上                  |
| HyFormer       | [outputs/smoke/hyformer/summary.json](outputs/smoke/hyformer/summary.json)             |          6 | 0.6487 | 0.3394 | 0.1410 |               19.6514 |               20.2111 |    0.017638 |          7.067170 |      82.3516 | 与 OneTrans 总算力接近                   |
| UniRec         | [outputs/smoke/unirec/summary.json](outputs/smoke/unirec/summary.json)                 |          6 | 0.7292 | 0.4199 | 0.2045 |               10.1691 |               10.4839 |    0.022693 |         18.975869 |      75.5584 | 当前 AUC 最高，但完整训练总算力也最高    |
| UniScaleFormer | [outputs/smoke/uniscaleformer/summary.json](outputs/smoke/uniscaleformer/summary.json) |          3 | 0.7125 | 0.3325 | 0.2160 |                0.8987 |                1.2936 |    0.008117 |          3.323026 |      72.9640 | 高 AUC 组里总算力最省                    |
| O_o            | [outputs/smoke/oo/summary.json](outputs/smoke/oo/summary.json)                         |          2 | 0.6483 | 0.3735 | 0.1869 |                0.4345 |                0.7286 |    0.012981 |          5.308285 |      67.9348 | 仍属低延迟组，总算力也相对温和           |

其中 `TFLOPs/批次` 与参数量来自各自 `summary.json` 里的 `model_profile`，口径仍是单次验证前向 profile。新增的 `完整训练总 TFLOPs` 来自 `compute_profile.estimated_end_to_end_tflops_total`，按“单次训练步 profile × 实际 train sample 数 + 单次验证前向 profile × 实际 val sample 数 + 训练结束后的 latency probe sample 数”估算。`deepcontextnet` 的 profile batch size 为 32，其余实验默认是 64。

## 当前结论

1. 当前仓库里真正存在、可直接运行的独立实验包仍是 baseline、ctr_baseline、deepcontextnet、interformer、onetrans、hyformer、unirec、uniscaleformer、oo 九个，并且现在都已经有可复核的 10-epoch sample smoke summary。
2. 这次统一重跑之后，sample 上 AUC 最高的是 `unirec`，其次是 `uniscaleformer`，再之后是 `onetrans`；先前文档里基于旧 smoke 产物得出的排序已经不再成立。
3. 如果主要看推理效率，当前最低平均时延是 `deepcontextnet`，其次是 `ctr_baseline` 和 `oo`；但如果看完整训练总算力，最低的是 `ctr_baseline`，其次是 `uniscaleformer` 和 `baseline`，而不是 `deepcontextnet`。
4. 如果看“效果 / 总算力”折中，`uniscaleformer` 目前比 `unirec` 更省算力：AUC 0.7125 对应约 3.3230 TFLOPs 完整训练估算，而 `unirec` 虽然 AUC 更高，但完整训练总算力约 18.9759 TFLOPs。
5. 当前 smoke 结果依然只适合做链路验证和粗方向判断，不适合做精细排序，因为数据仍然是 sample parquet，GAUC 覆盖率仍为 0，且完整训练总算力是基于单步 profile 放大的估算值，而不是逐 step 实时累积的硬件计数器结果。

## 后续更新规则

如果后续要继续维护本文件，请遵守下面几点：

1. 只记录当前分支真实存在的实验包。
2. 指标只写能直接打开文件复核的产物。
3. 产物优先链接到 `summary.json` 或 `evaluation.json`。
4. 如果某个实验还没有 smoke 结果，就写“仅 forward regression”，不要补猜测值。
