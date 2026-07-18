# H.264 摄像头与视频处理链路

本文档说明当前网页输入、后端流水线和 H.264 结果推流之间的真实链路。所有输入最终转换为逐帧 BGR 图像交给同一视频流水线，处理结果统一通过 H.264 WebSocket 返回网页。

## 1. 视频上传与服务器摄像头

```text
网页调用 /api/pipeline/start
  → 后端启动 pipeline 子进程
  → 视频文件、USB 摄像头或网络摄像头逐帧输入
  → 检测、跟踪、关键帧与轨迹记忆构建
  → 原始 BGR 结果写入子进程标准输出
  → _start_h264_reader
  → FFmpeg 编码为 H.264 分片 MP4
  → /ws/h264/{task_id}
  → 浏览器 MediaSource 播放
```

- 视频上传后由后端保存，再通过 `/api/pipeline/start` 启动任务。
- 服务器摄像头可使用设备编号、RTSP 地址或 HTTP 视频流地址。
- 流水线只向标准输出写入固定尺寸的原始 BGR 图像，日志写入标准错误输出。
- `_start_h264_reader` 按输出宽高读取完整帧，并将帧送入 FFmpeg。
- 浏览器通过 `/ws/h264/{task_id}` 接收初始化分片和媒体分片，再由 `MediaSource` 连续播放。

## 2. 浏览器摄像头输入

浏览器摄像头先调用 `/api/pipeline/start-browser-camera` 创建任务，再选择以下任一种输入方式。

### 2.1 逐帧 JPEG 图像流

```text
浏览器摄像头
  → Canvas 抓帧并编码 JPEG
  → /ws/camera/{task_id}
  → 后端解码为 BGR 图像
  → queue.Queue
  → VirtualCamera
  → 视频流水线
```

该方式兼容性最好，但浏览器需要逐帧编码 JPEG。

### 2.2 浏览器编码视频流

```text
浏览器摄像头
  → MediaRecorder 编码视频块
  → /ws/camera/{task_id}
  → 后端 FFmpeg 解码
  → queue.Queue
  → VirtualCamera
  → 视频流水线
```

浏览器优先选择可用的 H.264 编码格式；不支持时可回退为其他浏览器编码格式或逐帧 JPEG 图像流。

### 2.3 WebRTC

```text
浏览器摄像头
  → WebRTC 视频轨
  → /api/pipeline/webrtc/offer/{task_id}
  → 后端接收并转换为 BGR 图像
  → queue.Queue
  → VirtualCamera
  → 视频流水线
```

该方式适合低延迟摄像头输入。候选网络地址通过对应接口补充，断开时关闭连接并结束输入。

## 3. 统一流水线输入

```text
视频文件或服务器摄像头 → VideoInput ─┐
浏览器摄像头帧队列      → VirtualCamera ├→ 同一视频流水线
帧目录                  → VirtualCamera ┘
```

- `VideoInput` 负责视频文件、USB 摄像头和网络流。
- `VirtualCamera` 负责浏览器摄像头帧队列或帧目录。
- 后续检测、跟踪、轨迹记忆、正式关键帧和多模态特征生成不区分输入来源。

## 4. 统一结果输出

```text
流水线标注结果
  → 原始 BGR 图像
  → FFmpeg H.264 编码
  → 每观众独立 asyncio.Queue
  → H.264 WebSocket
  → 浏览器 MediaSource
```

无论输入来自上传视频、服务器摄像头还是浏览器摄像头，网页监控结果都通过同一 H.264 推流链路返回。原有逐帧图像流接口仍保留，用于兼容和调试。

## 5. 背压策略

- 浏览器输入队列满时删除较旧帧，只保留最新输入，避免流水线处理过期画面。
- H.264 观看队列满时删除旧分片，只保留较新的监控结果。
- 前端媒体分片积压过多时主动裁剪队列，防止播放延迟不断增加。
- 慢速观看者只影响自己的队列，不阻塞流水线和其他观看者。

## 6. 功能保留范围

当前实现必须同时保留：

- 视频上传处理。
- 服务器 USB、RTSP 和 HTTP 摄像头输入。
- 浏览器逐帧 JPEG 图像流。
- 浏览器编码视频流。
- WebRTC 摄像头输入。
- H.264 分片 MP4 实时输出。
- 原有逐帧图像流兼容接口。