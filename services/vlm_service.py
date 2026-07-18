"""Qwen3-VL-4B 的统一识别、核验和智能体请求服务。"""
from __future__ import annotations
import base64
import json
import re
from pathlib import Path
from typing import Any, Iterable
import cv2
import httpx
import numpy as np
from config import load_config

class LLMServiceError(RuntimeError):
    pass

def _extract_json(text: str) -> dict[str, Any]:
    content = text.strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        value = json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if not match:
            raise LLMServiceError("模型未返回 JSON")
        value = json.loads(match.group())
    if not isinstance(value, dict):
        raise LLMServiceError("模型 JSON 顶层必须是对象")
    return value

def _data_url(image: str | Path | np.ndarray) -> str:
    if isinstance(image, np.ndarray):
        ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            raise ValueError("图像编码失败")
        raw, mime = encoded.tobytes(), "image/jpeg"
    else:
        path = Path(image)
        raw = path.read_bytes()
        mime = {".png": "image/png", ".webp": "image/webp"}.get(path.suffix.lower(), "image/jpeg")
    return f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"

class AgentLLMService:
    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or load_config()
        self.settings = self.config["llm"]
        self.prompts = self.config.get("prompts", {})

    def complete_json(self, prompt: str, images: Iterable[str | Path | np.ndarray] = (), retries: int = 1) -> dict[str, Any]:
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        content.extend({"type": "image_url", "image_url": {"url": _data_url(image)}} for image in images)
        payload = {"model": self.settings["model"], "temperature": self.settings.get("temperature", 0.0), "messages": [{"role": "user", "content": content}]}
        url = f"{self.settings['base_url'].rstrip('/')}/chat/completions"
        headers = {"Authorization": f"Bearer {self.settings.get('api_key', '')}", "Content-Type": "application/json"}
        last_error: Exception | None = None
        for _ in range(retries + 1):
            try:
                response = httpx.post(url, headers=headers, json=payload, timeout=float(self.settings.get("timeout_seconds", 60)))
                response.raise_for_status()
                return _extract_json(response.json()["choices"][0]["message"]["content"])
            except Exception as error:
                last_error = error
        raise LLMServiceError(f"生成模型调用失败：{last_error}")

    def recognize(self, image: str | Path | np.ndarray) -> dict[str, Any]:
        result = self.complete_json(self.prompts["single_frame_recognition"], [image])
        readable = "yes" if str(result.get("has_readable_hull_number", "no")).lower() == "yes" else "no"
        hull = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]", "", str(result.get("vlm_hull_number") or "")).upper() or None
        confidence = min(1.0, max(0.0, float(result.get("readability_confidence") or 0)))
        return {"has_readable_hull_number": readable, "vlm_hull_number": hull if readable == "yes" else None, "readability_confidence": confidence, "description": str(result.get("description") or "").strip()}

    def verify(self, description: str | None, reference_images: list[str], evidence_images: list[str]) -> dict[str, Any]:
        if description:
            prompt = self.prompts["verify_description"].replace("{description}", description)
            images = evidence_images
        else:
            prompt = self.prompts["verify_registry"]
            images = reference_images[:3] + evidence_images[:3]
        result = self.complete_json(prompt, images)
        decision = str(result.get("decision", "uncertain")).lower()
        if decision not in {"match", "mismatch", "uncertain"}:
            decision = "uncertain"
        facts = result.get("facts", [])
        return {"decision": decision, "facts": facts if isinstance(facts, list) else [str(facts)]}

    def role(self, role: str, payload: dict[str, Any]) -> dict[str, Any]:
        prompt = self.prompts[role] + "\n输入：" + json.dumps(payload, ensure_ascii=False)
        return self.complete_json(prompt)
