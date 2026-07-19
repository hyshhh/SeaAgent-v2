"""运行时可调参数的读取、校验和持久化。"""
from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path
from typing import Any

import yaml

from .loader import load_config, project_root

SETTING_SPECS: dict[str, dict[str, Any]] = {
    "yolo.confidence": {"type": "float", "min": 0.01, "max": 0.99, "step": 0.01},
    "yolo.iou": {"type": "float", "min": 0.01, "max": 0.99, "step": 0.01},
    "yolo.detect_every_n_frames": {"type": "int", "min": 1, "max": 100, "step": 1},
    "yolo.tracker_params.track_buffer": {"type": "int", "min": 1, "max": 1000, "step": 1},
    "yolo.tracker_params.match_thresh": {"type": "float", "min": 0.01, "max": 1.0, "step": 0.01},
    "pipeline.max_stale_frames": {"type": "int", "min": 1, "max": 1000, "step": 1},
    "pipeline.candidate_every_n_frames": {"type": "int", "min": 1, "max": 1000, "step": 1},
    "pipeline.candidate_pool_size": {"type": "int", "min": 1, "max": 100, "step": 1},
    "pipeline.keyframe_pool_size": {"type": "int", "min": 1, "max": 6, "step": 1},
    "pipeline.quality.min_score": {"type": "float", "min": 0.0, "max": 1.0, "step": 0.01},
    "pipeline.recognition_timeout_seconds": {"type": "int", "min": 1, "max": 600, "step": 1},
    "pipeline.retention.replace_margin": {"type": "float", "min": 0.0, "max": 1.0, "step": 0.01},
    "pipeline.aggregation.null_weight": {"type": "float", "min": 0.0, "max": 1.0, "step": 0.01},
    "pipeline.aggregation.conflict_margin": {"type": "float", "min": 0.0, "max": 20.0, "step": 0.01},
    "pipeline.aggregation.confirmed_support": {"type": "float", "min": 0.0, "max": 20.0, "step": 0.01},
    "pipeline.aggregation.confirmed_count": {"type": "int", "min": 1, "max": 6, "step": 1},
    "pipeline.aggregation.null_margin": {"type": "float", "min": 0.0, "max": 20.0, "step": 0.01},
    "pipeline.retrieval.top_k": {"type": "int", "min": 1, "max": 20, "step": 1},
    "pipeline.retrieval.text_match": {"type": "float", "min": 0.0, "max": 1.0, "step": 0.01},
    "pipeline.retrieval.text_exclude": {"type": "float", "min": 0.0, "max": 1.0, "step": 0.01},
    "pipeline.retrieval.image_match": {"type": "float", "min": 0.0, "max": 1.0, "step": 0.01},
    "pipeline.retrieval.image_exclude": {"type": "float", "min": 0.0, "max": 1.0, "step": 0.01},
    "pipeline.retrieval.dedup_high": {"type": "float", "min": 0.0, "max": 1.0, "step": 0.01},
    "pipeline.retrieval.dedup_low": {"type": "float", "min": 0.0, "max": 1.0, "step": 0.01},
    "pipeline.agent.max_rounds": {"type": "int", "min": 1, "max": 10, "step": 1},
}

RUNTIME_FILE = Path(os.getenv("SEAAGENT_CONFIG_DIR", project_root() / "config")) / "runtime.yaml"


def _get(data: dict[str, Any], path: str) -> Any:
    current: Any = data
    for key in path.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _set(data: dict[str, Any], path: str, value: Any) -> None:
    keys = path.split(".")
    current = data
    for key in keys[:-1]:
        current = current.setdefault(key, {})
    current[keys[-1]] = value


def _flatten(data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    values: dict[str, Any] = {}
    for key, value in data.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            values.update(_flatten(value, path))
        else:
            values[path] = value
    return values


def _read_runtime() -> dict[str, Any]:
    if not RUNTIME_FILE.exists():
        return {}
    content = yaml.safe_load(RUNTIME_FILE.read_text(encoding="utf-8")) or {}
    return content if isinstance(content, dict) else {}


def _write_runtime(data: dict[str, Any]) -> None:
    RUNTIME_FILE.parent.mkdir(parents=True, exist_ok=True)
    temporary = RUNTIME_FILE.with_suffix(".tmp")
    temporary.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    temporary.replace(RUNTIME_FILE)


def _coerce(path: str, value: Any) -> int | float:
    spec = SETTING_SPECS[path]
    try:
        number = int(value) if spec["type"] == "int" else float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"参数 {path} 必须是数字") from error
    if not spec["min"] <= number <= spec["max"]:
        raise ValueError(f"参数 {path} 必须在 {spec['min']} 到 {spec['max']} 之间")
    return number


def _validate_relations(config: dict[str, Any]) -> None:
    retrieval = config["pipeline"]["retrieval"]
    if retrieval["text_exclude"] >= retrieval["text_match"]:
        raise ValueError("文本排除阈值必须小于文本匹配阈值")
    if retrieval["image_exclude"] >= retrieval["image_match"]:
        raise ValueError("图像排除阈值必须小于图像匹配阈值")
    if retrieval["dedup_low"] >= retrieval["dedup_high"]:
        raise ValueError("去重低阈值必须小于去重高阈值")


def public_settings(config: dict[str, Any]) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for path in SETTING_SPECS:
        _set(values, path, _get(config, path))
    return {"settings": values, "specs": deepcopy(SETTING_SPECS)}


def update_settings(config: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    submitted = payload.get("settings", payload)
    if not isinstance(submitted, dict):
        raise ValueError("设置内容格式不正确")
    changes = {path: _coerce(path, value) for path, value in _flatten(submitted).items() if path in SETTING_SPECS}
    if not changes:
        raise ValueError("没有收到可修改的参数")
    updated = deepcopy(config)
    for path, value in changes.items():
        _set(updated, path, value)
    _validate_relations(updated)
    runtime = _read_runtime()
    for path, value in changes.items():
        _set(runtime, path, value)
    _write_runtime(runtime)
    for path, value in changes.items():
        _set(config, path, value)
    return public_settings(config)


def reset_settings(config: dict[str, Any]) -> dict[str, Any]:
    if RUNTIME_FILE.exists():
        RUNTIME_FILE.unlink()
    defaults = load_config()
    for path in SETTING_SPECS:
        _set(config, path, _get(defaults, path))
    return public_settings(config)
