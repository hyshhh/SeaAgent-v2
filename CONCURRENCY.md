# 当前并发模型

本项目面向单段海域监控视频和一套共享轨迹记忆，默认只允许一个视频流水线运行。并发主要用于网页异步通信、实时推流和单帧识别，不用于同时构建多段视频记忆。

## 1. 网页任务并发

```text
HTTP 或 WebSocket 请求
  → asyncio 事件循环
  → asyncio.Semaphore(1)
  → 单个视频流水线任务
```

- `web/routes/pipeline_api.py` 使用 `asyncio.Semaphore(1)` 限制流水线数量。
- `config/pipeline.yaml` 固定 `max_parallel_pipelines: 1`。
- 视频上传和服务器摄像头由网页启动独立流水线子进程。
- 浏览器摄像头由后台线程运行流水线，但仍占用同一个流水线名额。
- 单任务限制避免多个流水线同时重置或写入共享轨迹记忆、关键帧表和特征索引。

## 2. 浏览器摄像头队列

```text
浏览器摄像头
  → WebSocket 或 WebRTC
  → queue.Queue
  → VirtualCamera
  → 视频流水线
```

- 浏览器可发送逐帧图像、浏览器编码视频流或 WebRTC 视频轨。
- 后端将解码后的图像写入线程安全的 `queue.Queue`。
- `VirtualCamera` 从队列读取图像，并作为统一视频输入交给流水线。
- 队列满时删除旧数据并保留最新数据，避免输入延迟持续累积。

## 3. H.264 结果推流

```text
流水线原始 BGR 图像
  → FFmpeg 编码
  → 分片 MP4
  → 每观众 asyncio.Queue
  → /ws/h264/{task_id}
  → 浏览器 MediaSource
```

- `_start_h264_reader` 读取流水线标准输出中的原始 BGR 图像。
- FFmpeg 将图像编码为 H.264 分片 MP4。
- 每个观看者维护独立的 `asyncio.Queue`，互不阻塞。
- 观看者消费速度过慢时丢弃旧分片，只保留较新的监控结果。
- 浏览器通过 `MediaSource` 追加初始化分片和媒体分片，实现连续播放。

## 4. 单帧识别并发

```text
候选船舶裁剪图
  → ThreadPoolExecutor
  → Qwen3-VL-4B 单帧识别
  → 正式关键帧池
```

- `pipeline/track_memory_builder.py` 使用 `ThreadPoolExecutor` 异步处理候选帧。
- 主检测与跟踪链路不等待每张候选帧识别完成。
- 同一轨迹结束前会等待其未完成识别任务，并将可用结果写入正式关键帧池。
- 正式关键帧写入、特征索引更新和数据库记录更新按一致性事务提交。

## 5. 三智能体模型服务

```text
Planner ─┐
Observer ├→ 共享 Qwen3-VL-4B 服务
Reflector┘
```

- 三个子智能体是不同角色和提示词，不需要启动三个独立模型。
- 单帧识别、规划、观察、反思和灰区核验共享一个 `Qwen3-VL-4B` 服务。
- `Qwen3-VL-Embedding-2B` 独立负责文本与图像的统一多模态特征生成。

## 6. 当前约束

- 系统默认只维护一段监控视频和一套轨迹记忆。
- 不支持多个流水线同时写入同一数据库和同一特征索引。
- 多个网页观看者可以同时观看同一个任务的 H.264 输出。
- 问答请求可复用已经构建完成的轨迹记忆，但不得与记忆重置操作并发执行。