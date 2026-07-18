"""Qwen3-VL-Embedding-2B 的延迟加载封装。"""

from __future__ import annotations

import sys
import threading
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from config import load_config


class EmbeddingUnavailableError(RuntimeError):
    pass


class QwenMultimodalEmbedder:
    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or load_config()
        self.settings = self.config["embedding"]
        self.dimension = int(self.settings.get("dimension", 2048))
        self._model = None
        self._lock = threading.RLock()

    def _load(self):
        with self._lock:
            if self._model is not None:
                return self._model
            source_path = Path(self.settings.get("source_path", ""))
            if source_path.exists() and str(source_path.resolve()) not in sys.path:
                sys.path.insert(0, str(source_path.resolve()))
            try:
                from src.models.qwen3_vl_embedding import Qwen3VLEmbedder
            except ImportError:
                try:
                    from scripts.qwen3_vl_embedding import Qwen3VLEmbedder
                except ImportError as error:
                    raise EmbeddingUnavailableError(
                        "未找到 Qwen3-VL-Embedding 官方源码，请按 SETUPREADME.md 安装"
                    ) from error
            kwargs: dict[str, Any] = {"model_name_or_path": self.settings["model_path"]}
            try:
                import torch
                dtype_name = str(self.settings.get("dtype", "bfloat16"))
                kwargs["torch_dtype"] = getattr(torch, dtype_name, torch.bfloat16)
            except ImportError:
                pass
            attention = self.settings.get("attention")
            if attention:
                kwargs["attn_implementation"] = attention
            self._model = Qwen3VLEmbedder(**kwargs)
            return self._model

    def process(self, inputs: list[dict[str, Any]]) -> np.ndarray:
        if not inputs:
            return np.empty((0, self.dimension), dtype=np.float32)
        model = self._load()
        with self._lock:
            try:
                values = model.process(inputs, normalize=bool(self.settings.get("normalize", True)))
            except TypeError:
                values = model.process(inputs)
        if hasattr(values, "detach"):
            values = values.detach().float().cpu().numpy()
        vectors = np.asarray(values, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        if vectors.shape[1] != self.dimension:
            raise ValueError(f"向量维度错误：期望 {self.dimension}，实际 {vectors.shape[1]}")
        return self._normalize(vectors)

    def encode_images(self, image_paths: Iterable[str | Path]) -> np.ndarray:
        return self.process([{"image": str(Path(path).resolve())} for path in image_paths])

    def encode_text(self, text: str, instruction: str) -> np.ndarray:
        return self.process([{"text": text, "instruction": instruction}])[0]

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if np.any(norms <= 1e-12):
            raise ValueError("模型返回零向量")
        return vectors / norms
