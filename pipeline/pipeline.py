"""SeaAgent 视频检测、轨迹记忆构建与实时推流主流水线。"""
from __future__ import annotations
import logging
import sys
import time
from pathlib import Path
from typing import Any, Callable
import cv2
import numpy as np
from config import load_config
from pipeline.demo import DemoRenderer
from pipeline.detector import Detection, ShipDetector
from pipeline.fps import FPSMeter, LatencyMeter
from pipeline.track_memory_builder import TrackMemoryBuilder
from pipeline.video_input import InputSource

logger = logging.getLogger(__name__)

class ShipPipeline:
    def __init__(self, config: dict[str, Any] | None = None, pool_event_callback: Callable[[dict[str, Any]], None] | None = None):
        self._config = config or load_config()
        settings = self._config["pipeline"]
        self._target_fps = float(settings.get("target_fps", 0))
        self._monitor_start_time = float(settings.get("monitor_start_time", 0))
        self._detect_every_n = max(1, int(settings.get("detect_every_n_frames", 1)))
        self._demo_enabled = bool(settings.get("demo", True))
        self._save_output_video = bool(settings.get("save_output_video", True))
        pipe_output_size = settings.get("pipe_output_size") or settings.get("output_size")
        self._pipe_output_size = tuple(pipe_output_size) if pipe_output_size else None
        self._stop_file = Path(settings["stop_file"]) if settings.get("stop_file") else None
        self._raw_stdout = bool(settings.get("raw_stdout", False))
        self._no_output = bool(settings.get("no_output", False))
        self._detector = ShipDetector(model_path=settings.get("yolo_model", "yolov8n.pt"), device=settings.get("device", ""), conf_threshold=float(settings.get("conf_threshold", 0.25)), iou_threshold=float(settings.get("iou_threshold", 0.45)), tracker_type=settings.get("tracker", "bytetrack"), tracker_params=settings.get("tracker_params"), classes=settings.get("detect_classes", [8]))
        self._memory = TrackMemoryBuilder(self._config, pool_event_callback=pool_event_callback)
        self._renderer = DemoRenderer(show_fps=True, show_track_id=True)
        self._fps = FPSMeter(window_seconds=10.0)
        self._latency = LatencyMeter(window_seconds=10.0)

    def process(self, source: str | int | object, output_path: str | None = None, display: bool = False, max_frames: int = 0, frame_callback: Callable[[np.ndarray, int], None] | None = None, stream_dir: str | Path | None = None) -> dict[str, Any]:
        input_source = InputSource(source)
        source_path = str(source) if isinstance(source, (str, Path)) else ""
        source_fps = float(input_source.source_fps or 25)
        target_fps = self._target_fps or source_fps
        skip_interval = max(1, round(source_fps / target_fps)) if target_fps and source_fps > target_fps else 1
        stop_file = Path(stream_dir) / "__STOP__" if stream_dir else self._stop_file
        stream_path = Path(stream_dir) if stream_dir else None
        if stream_path:
            stream_path.mkdir(parents=True, exist_ok=True)
        if self._save_output_video and not output_path and not self._no_output:
            output_dir = Path(self._config["demo_video"]["output_dir"])
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = str(output_dir / f"seaagent_{time.strftime('%Y%m%d_%H%M%S')}.mp4")
        writer = None
        if output_path and not self._no_output:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), source_fps, (input_source.width, input_source.height))
            if not writer.isOpened():
                writer = None
                logger.warning("无法创建输出视频：%s", output_path)
        frame_index = processed = detections_total = 0
        seen_track_ids: set[str] = set()
        last_detections: list[Detection] = []
        start = time.time()
        stream_every = max(1, int(self._config["pipeline"].get("stream_write_every_n_frames", 2)))
        stream_jpeg_quality = int(self._config["pipeline"].get("stream_jpeg_quality", 65))
        try:
            while True:
                if stop_file and stop_file.exists():
                    break
                ok, frame = input_source.read()
                if not ok:
                    break
                frame_index += 1
                if max_frames and frame_index > max_frames:
                    break
                if skip_interval > 1 and frame_index % skip_interval != 1:
                    continue
                processed += 1
                self._fps.tick("stream")
                if processed % self._detect_every_n == 0:
                    with self._latency.measure("yolo"):
                        last_detections = self._detector.detect(frame, frame_index)
                    self._memory.observe(frame, last_detections, frame_index, frame_index / source_fps, source_path, source_fps)
                    seen_track_ids.update(str(item.track_id) for item in last_detections)
                    detections_total += len(last_detections)
                display_frame = self._renderer.render(frame, last_detections, self._memory.display_tracks(), self._fps.get_all_fps(), frame_index) if self._demo_enabled or writer or display or stream_path or self._raw_stdout else frame
                if writer:
                    writer.write(display_frame)
                if stream_path and (processed % stream_every == 0):
                    temp = stream_path / "latest.tmp.jpg"
                    cv2.imwrite(str(temp), display_frame, [cv2.IMWRITE_JPEG_QUALITY, stream_jpeg_quality])
                    temp.replace(stream_path / "latest.jpg")
                if self._raw_stdout:
                    output_frame = display_frame
                    if self._pipe_output_size and (output_frame.shape[1], output_frame.shape[0]) != self._pipe_output_size:
                        output_frame = cv2.resize(output_frame, self._pipe_output_size, interpolation=cv2.INTER_AREA)
                    sys.stdout.buffer.write(np.ascontiguousarray(output_frame).tobytes())
                    sys.stdout.buffer.flush()
                if frame_callback:
                    frame_callback(display_frame, frame_index)
                if display:
                    cv2.imshow("SeaAgent", display_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
            self._memory.finalize_all()
            elapsed = max(1e-6, time.time() - start)
            video_duration = frame_index / source_fps
            monitor_start_time = self._monitor_start_time or start
            monitor_end_time = monitor_start_time + video_duration if self._monitor_start_time else time.time()
            return {"total_frames": frame_index, "processed_frames": processed, "total_detections": detections_total, "total_tracks": len(seen_track_ids), "video_duration_seconds": round(video_duration, 6), "monitor_start_time": monitor_start_time, "monitor_end_time": monitor_end_time, "elapsed_seconds": round(elapsed, 2), "avg_fps": round(processed / elapsed, 2), "output_path": output_path or ""}
        finally:
            try:
                if self._memory.active:
                    self._memory.finalize_all()
            except Exception as error:
                logger.warning("轨迹记忆收尾失败：%s", error)
            input_source.release()
            if writer:
                writer.release()
            if display:
                cv2.destroyAllWindows()
            self._detector.cleanup()

    @property
    def agent_trace(self) -> list[dict[str, Any]]:
        return list(self._memory.trace)

    def set_demo(self, enabled: bool) -> None:
        self._demo_enabled = enabled
