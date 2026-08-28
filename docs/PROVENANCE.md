# 代码来源与公开边界

## 来源

本仓库基于 NVlabs LongLive，并保留上游 Apache License 2.0。LayerRecall、CHPM 和 training-compatible Streaming Ulysses 为本项目新增实现。

## 公开内容

- LayerRecall hard/soft inference；
- CHPM student full rollout；
- SP=1 与 SP=2 训练；
- CHPM v3 exact resume；
- LongLive2 原有 DMD trainer。

## 排除内容

- 模型权重、训练数据和实验产物；
- 研发期可视化、抽帧、热力图与评测模板；
- teacher-context student rollout；
- 独立 inference-only sequence-parallel 路径；
- 量化训练与推理路径；
- LayerRecall RoPE 实验分支。

模型与数据需由使用者根据各自许可证单独获取。
