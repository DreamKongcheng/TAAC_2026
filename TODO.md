# TODO

## 运行时与 CLI

1. 给 `taac-train` 增加可选的 `--dataset-path` 覆写，避免每次为新数据集复制 wrapper 包。
2. 给 `taac-evaluate batch` 增加 continue-on-error 模式，把缺 checkpoint 和 checkpoint 不兼容从“整体失败”改成“逐项汇报”。
3. 明确 `summary.json` 与未来 `evaluation.json` 的字段约定，减少后续文档漂移。

## 实验与验证

1. 用同一时间点、同一 smoke 配置重新补齐 baseline / interformer / onetrans / hyformer / oo 的一轮统一 smoke，避免引用不同时间留下的历史输出。
2. 继续为 baseline 补独立的 package build + forward 回归用例，使其覆盖粒度和其它实验包一致。
3. 评估是否要把 O_o 的 InfoNCE / -logQ / feature crossing 继续往当前共享训练栈里迁移。

## 文档维护

1. 文档以后只写当前分支真实存在的入口和脚本，不再保留“未来可能有”的命令示例。
2. `docs/EXPERIMENTS.md` 只接受可直接链接复核的产物文件。
3. 如果未来重新引入可视化或 EDA CLI，需要同步恢复 README 与 dev 文档，不要让命令再次先于代码出现。
