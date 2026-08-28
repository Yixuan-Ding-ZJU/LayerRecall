# LayerRecall

本仓库包含 LayerRecall 在 LongLive2 上的公开实现。

## 保留内容

- current-conditioned LayerRecall 与 per-layer history bank；
- memory-sensitive layers：`4, 9, 10, 12, 13, 15, 16, 17, 18, 26`；
- 固定 `sink + selected history + current` attention budget；
- hard inference 与 soft inference；
- CHPM student full rollout 训练与 6 个 prediction anchors；
- 传统多卡训练（`SP=1`）与 Streaming Ulysses（`SP=2`）；
- 两种拓扑下的 CHPM v3 exact resume；
- LongLive2 原有 DMD trainer。

## 主要入口

- 推理：`inference.py`
- 训练：`train.py`
- CHPM launcher：`scripts/train_chpm.sh`
- LayerRecall runtime：`utils/layer_recall.py`
- CHPM model/trainer：`model/chpm.py`、`trainer/chpm.py`
- Streaming Ulysses：`wan_5b/distributed/streaming_ulysses.py`

## 配置文件

- `configs/inference_layer_recall.yaml`
- `configs/train_chpm_384_dp.yaml`
- `configs/train_chpm_384_sp2.yaml`
- `configs/train_chpm_384_dp_resume_smoke.yaml`
- `configs/train_chpm_384_sp2_resume_smoke.yaml`

机器相关路径均通过环境变量传入。CHPM launcher 的拓扑、rendezvous、输出目录、resume 和 W&B 参数也均可由外部环境变量控制；W&B 默认关闭。

环境安装、模型下载、数据准备和正式启动说明将在后续 README 专项中补充。
