# SeaAgent 部署说明

## 一、运行环境

- 建议使用 Python 3.10 或 3.11。
- 视频转码和目标船片段生成需要 `ffmpeg`。
- `Qwen3-VL-4B` 建议部署在独立显卡上。
- `Qwen3-VL-Embedding-2B` 使用独立接口服务，建议部署在另一张显卡上。

## 二、安装项目依赖

安装项目依赖：

```bash
pip install -e .
```

确认系统命令可用：

```bash
ffmpeg -version
python -c "import cv2, fastapi, faiss; print('基础依赖正常')"
```

## 三、部署 Qwen3-VL-4B-AWQ

单帧识别、规划智能体、观察智能体、反思智能体和灰区核验共享同一个兼容接口服务。无需为三个子智能体分别启动模型。

示例：

```bash
vllm serve Qwen/Qwen3-VL-4B-AWQ \
  --served-model-name Qwen/Qwen3-VL-4B-AWQ \
  --api-key abc123 \
  --port 7890 \
  --max-model-len 32768
```

对应配置位于 `config/app.yaml`：

```yaml
llm:
  model: "Qwen/Qwen3-VL-4B-AWQ"
  api_key: "abc123"
  base_url: "http://localhost:7890/v1"
```

## 四、部署 Qwen3-VL-Embedding-2B

使用兼容接口单独启动向量服务：

```bash
vllm serve Qwen/Qwen3-VL-Embedding-2B \
  --served-model-name Qwen/Qwen3-VL-Embedding-2B \
  --runner pooling \
  --api-key abc123 \
  --port 7891
```

对应配置：

```yaml
embedding:
  model: "Qwen/Qwen3-VL-Embedding-2B"
  api_key: "abc123"
  base_url: "http://localhost:7891/v1"
  timeout_seconds: 60
  dimension: 2048
  normalize: true
```

应用通过 `/v1/embeddings` 调用向量服务。正式关键帧和先验库参考图只保存图像特征；用户描述在查询时即时编码。

## 五、配置检测器

`config/yolo.yaml` 默认使用：

```yaml
yolo:
  model: yolov8n.pt
  device: ""
  confidence: 0.25
  iou: 0.45
  classes: [8]
  tracker: bytetrack
```

如使用自训练船舶检测模型，修改 `model` 和 `classes`。检测器只负责船舶区域，不评估舷号区域质量。

## 六、初始化存储

无需手工建表。首次启动时自动创建：

```text
data/memory/*.csv
data/registry/*.csv
data/memory/keyframes
data/memory/trajectories
data/memory/clips
data/registry/reference_images
data/indexes
```

先验库发生增删改时自动完整重建先验库索引。正式关键帧索引随正式池替换增删活动向量。

## 七、启动网页

```bash
python -m web.app
```

或：

```bash
seaagent
```

默认访问：

```text
http://127.0.0.1:8000
```

## 八、处理视频

```bash
python -m pipeline.cli data/videos/example.mp4 \
  --demo \
  --output output/result.mp4
```

常用参数：

```text
--conf             检测置信度
--iou              交并比阈值
--detect-every     检测帧间隔
--target-fps       目标处理帧率
--max-frames       最大处理帧数
--device           检测设备
--yolo-model       检测模型路径
--display          本地实时窗口
--no-output        不保存结果视频
```

## 九、显存与进程安排

推荐部署：

```text
进程一：Qwen3-VL-4B 接口服务，供识别和三个子智能体共享
进程二：Qwen3-VL-Embedding-2B 向量接口服务
进程三：SeaAgent 网页服务和当前单视频流水线
```

系统面向单视频监控任务，`config/pipeline.yaml` 默认固定 `max_parallel_pipelines: 1`，因为同一任务只维护一段监控视频、一套轨迹记忆和两类共享特征索引。多个视频流水线不得同时重置或写入这些共享状态。

## 十、验证

```bash
python -m compileall -q agent config memory pipeline services tools vector_store web
python -c "from config import load_config; print(load_config()['app']['name'])"
python -c "from web.app import app; print(len(app.routes))"
```

统一多模态特征模型就绪后再验证：

```bash
python -c "from services import QwenMultimodalEmbedder; print(QwenMultimodalEmbedder().dimension)"
```