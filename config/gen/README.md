# GEN 阶段说明

`config/gen` 用来放目录式实验包。这里的实验包主要承担两类任务：

1. 吸收公开代码或公开论文里的核心建模思路。
2. 在不破坏当前 parquet batch 契约的前提下，把这些结构落成可训练、可评估、可回归测试的独立入口。

## 当前约定

每个可执行实验包都应该满足下面的目录契约：

1. `__init__.py` 导出 `EXPERIMENT`。
2. `data.py` 私有实现数据管线。
3. `model.py` 私有实现模型主体。
4. `utils.py` 私有实现损失与优化器装配。
5. `README.md` 说明来源、映射方式、运行命令与当前验证状态。

## 当前已有包

1. `baseline`
2. `ctr_baseline`
3. `deepcontextnet`
4. `interformer`
5. `onetrans`
6. `hyformer`
7. `unirec`
8. `uniscaleformer`
9. `oo`

`symbiosis` 当前只是概念说明，不是可执行实验包。

## 默认运行方式

```bash
uv run taac-train --experiment config/gen/baseline
uv run taac-evaluate single --experiment config/gen/baseline
```

其余实验包把 `baseline` 替换成对应目录名即可。

## 设计取向

GEN 阶段仍然坚持三条原则：

1. 最大化保留外部方案真正有辨识度的结构。
2. 明确写出哪些地方因为当前 batch 契约做了适配。
3. 优先让实验包自包含、可回归，而不是把跨实验依赖越堆越多。