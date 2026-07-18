"""轨迹记忆展示、清理与自动过期。"""
from __future__ import annotations

import json
import os
import shutil
import threading
from pathlib import Path
from typing import Any, Iterable

from memory.repository import MemoryRepository
from vector_store import VectorCatalog


class MemorySettingsStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._lock = threading.RLock()
        self._cached_signature = None
        self._cached_value = {"retentionSeconds": 0.0}

    def read(self) -> dict[str, float]:
        with self._lock:
            signature = self._signature()
            if signature == self._cached_signature:
                return dict(self._cached_value)
            value = {"retentionSeconds": 0.0}
            if self.path.is_file():
                try:
                    payload = json.loads(self.path.read_text(encoding="utf-8"))
                    value["retentionSeconds"] = max(0.0, float(payload.get("retentionSeconds", 0)))
                except (OSError, ValueError, TypeError, json.JSONDecodeError):
                    value = {"retentionSeconds": 0.0}
            self._cached_signature = signature
            self._cached_value = value
            return dict(value)

    def write(self, retention_seconds: float) -> dict[str, float]:
        value = {"retentionSeconds": max(0.0, float(retention_seconds))}
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_suffix(self.path.suffix + ".tmp")
            temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(temporary, self.path)
            self._cached_signature = self._signature()
            self._cached_value = value
        return dict(value)

    def _signature(self):
        try:
            stat = self.path.stat()
            return stat.st_mtime_ns, stat.st_size
        except FileNotFoundError:
            return None


class TrackMemoryManager:
    def __init__(self, config: dict[str, Any], repository: MemoryRepository, vectors: VectorCatalog):
        self.config = config
        self.repository = repository
        self.vectors = vectors
        self.settings = MemorySettingsStore(config["paths"]["memory_settings_json"])
        self._lock = threading.RLock()

    def snapshot(self) -> dict[str, Any]:
        tracks = self.repository.find_tracks()
        grouped = self.repository.get_keyframes([item["trackId"] for item in tracks], embedded_only=False)
        embedded_count = 0
        for track in tracks:
            frames = grouped.get(str(track["trackId"]), [])
            track["keyframeCount"] = len(frames)
            track["embeddedKeyframeCount"] = sum(1 for frame in frames if frame.get("isEmbedded"))
            track["memoryState"] = "已完成" if track.get("trajectoryPath") else "构建中"
            embedded_count += track["embeddedKeyframeCount"]
        return {
            "tracks": tracks,
            "trackCount": len(tracks),
            "keyframeCount": sum(item["keyframeCount"] for item in tracks),
            "embeddedKeyframeCount": embedded_count,
            "settings": self.settings.read(),
        }

    def clear_all(self) -> dict[str, int]:
        with self._lock:
            snapshot = self.snapshot()
            self.vectors.keyframes.rebuild([])
            self.repository.clear_track_memory()
            self._clear_directories(("keyframe_dir", "trajectory_dir", "clip_dir"))
        return {"deletedTracks": snapshot["trackCount"], "deletedKeyframes": snapshot["keyframeCount"]}

    def prune_expired(self, reference_time: float, protected_track_ids: Iterable[str | int] = ()) -> list[str]:
        retention = self.settings.read()["retentionSeconds"]
        if retention <= 0:
            return []
        protected = {str(value) for value in protected_track_ids}
        expired = [
            item["trackId"] for item in self.repository.find_tracks()
            if str(item["trackId"]) not in protected and reference_time - float(item["endTime"]) > retention
        ]
        if not expired:
            return []
        with self._lock:
            expired_ids = {str(value) for value in expired}
            grouped = self.repository.get_keyframes(expired, embedded_only=False)
            vector_ids = [
                int(frame["keyframeVectorId"])
                for frames in grouped.values() for frame in frames
                if frame.get("keyframeVectorId") is not None
            ]
            if vector_ids:
                self.vectors.keyframes.remove(vector_ids)
            tracks = {str(item["trackId"]): item for item in self.repository.find_tracks() if str(item["trackId"]) in expired_ids}
            for track_id in expired:
                for frame in grouped.get(str(track_id), []):
                    self._unlink(frame.get("keyframePath"))
                self._unlink(tracks.get(str(track_id), {}).get("trajectoryPath"))
                self.repository.delete_track(track_id)
            self._clear_directories(("clip_dir",))
        return [str(value) for value in expired]

    def _clear_directories(self, keys: Iterable[str]) -> None:
        for key in keys:
            directory = Path(self.config["paths"][key]).resolve()
            directory.mkdir(parents=True, exist_ok=True)
            for item in directory.iterdir():
                shutil.rmtree(item) if item.is_dir() else item.unlink(missing_ok=True)

    @staticmethod
    def _unlink(value: str | None) -> None:
        if value:
            Path(value).unlink(missing_ok=True)
