"""Qwen3-VL-4B 的统一识别、核验和智能体请求服务。"""
from __future__ import annotations
import base64
import json
import re
from pathlib import Path
from typing import Any, Callable, Iterable
import cv2
import httpx
import numpy as np
from config import load_config

class LLMServiceError(RuntimeError):
    pass

def _extract_json(text: str) -> dict[str, Any]:
    """从模型输出中提取完整对象，优先保留最外层计划对象。"""
    content = str(text or "").strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    if content.lower().startswith("json\n"):
        content = content[5:].lstrip()

    decoder = json.JSONDecoder()
    candidates: list[dict[str, Any]] = []
    last_error: Exception | None = None
    for match in re.finditer(r"\{", content):
        candidate = content[match.start():]
        for value_text in (candidate, re.sub(r",(\s*[}\]])", r"\1", candidate)):
            try:
                value, _ = decoder.raw_decode(value_text)
            except json.JSONDecodeError as error:
                last_error = error
                continue
            if isinstance(value, dict):
                candidates.append(value)
                break

    if candidates:
        priority = ("calls", "goal", "proposedState", "state", "decision")
        candidates.sort(key=lambda item: sum(key in item for key in priority), reverse=True)
        return candidates[0]
    if not content:
        raise LLMServiceError("模型未返回 JSON")
    raise LLMServiceError(f"模型 JSON 无法解析：{last_error}")


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

    def _thinking_enabled(self) -> bool:
        """默认关闭思考模式；配置可为 bool 或 true/false 字符串。"""
        settings = self.config.get("llm", self.settings) if isinstance(self.config, dict) else self.settings
        value = (settings or {}).get("enable_thinking", False)
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)

    def _base_payload(self, messages: list[dict[str, Any]], stream: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.settings["model"],
            "temperature": self.settings.get("temperature", 0.0),
            "messages": messages,
        }
        if stream:
            payload["stream"] = True
        # Qwen3 / vLLM / SGLang：默认关闭思考链，避免前端流式刷长推理
        enable_thinking = self._thinking_enabled()
        payload["chat_template_kwargs"] = {"enable_thinking": enable_thinking}
        payload["enable_thinking"] = enable_thinking
        return payload

    @staticmethod
    def _strip_thinking(text: str) -> str:
        """去掉模型误输出的思考标签，只保留最终正文。"""
        content = str(text or "")
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.IGNORECASE | re.DOTALL)
        content = re.sub(r"<thinking>.*?</thinking>", "", content, flags=re.IGNORECASE | re.DOTALL)
        # 残留未闭合标签时截断思考段
        content = re.sub(r"<think>.*$", "", content, flags=re.IGNORECASE | re.DOTALL)
        content = re.sub(r"<thinking>.*$", "", content, flags=re.IGNORECASE | re.DOTALL)
        return content.strip()

    def _prompt(self, key: str) -> str:
        prompts = self.config.get("prompts", self.prompts) if isinstance(self.config, dict) else self.prompts
        value = (prompts or {}).get(key) or (self.prompts or {}).get(key) or ""
        if not value:
            raise LLMServiceError(f"缺少提示词：{key}")
        return value

    def complete_json(self, prompt: str, images: Iterable[str | Path | np.ndarray] = (), retries: int = 1) -> dict[str, Any]:
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        content.extend({"type": "image_url", "image_url": {"url": _data_url(image)}} for image in images)
        payload = self._base_payload([{"role": "user", "content": content}])
        url = f"{self.settings['base_url'].rstrip('/')}/chat/completions"
        headers = {"Authorization": f"Bearer {self.settings.get('api_key', '')}", "Content-Type": "application/json"}
        last_error: Exception | None = None
        for _ in range(retries + 1):
            try:
                response = httpx.post(url, headers=headers, json=payload, timeout=float(self.settings.get("timeout_seconds", 60)))
                response.raise_for_status()
                message = response.json()["choices"][0]["message"]
                raw = message.get("content") or message.get("reasoning_content") or ""
                return _extract_json(self._strip_thinking(raw))
            except Exception as error:
                last_error = error
        raise LLMServiceError(f"生成模型调用失败：{last_error}")

    def complete_text(self, prompt: str) -> str:
        payload = self._base_payload([{"role": "user", "content": prompt}])
        url = f"{self.settings['base_url'].rstrip('/')}/chat/completions"
        headers = {"Authorization": f"Bearer {self.settings.get('api_key', '')}", "Content-Type": "application/json"}
        try:
            response = httpx.post(url, headers=headers, json=payload, timeout=float(self.settings.get("timeout_seconds", 60)))
            response.raise_for_status()
            message = response.json()["choices"][0]["message"]
            raw = message.get("content") or ""
            return self._strip_thinking(str(raw))
        except Exception as error:
            raise LLMServiceError(f"生成模型调用失败：{error}") from error

    def complete_text_stream(self, prompt: str, on_delta: Callable[[str], None]) -> str:
        payload = self._base_payload([{"role": "user", "content": prompt}], stream=True)
        url = f"{self.settings['base_url'].rstrip('/')}/chat/completions"
        headers = {"Authorization": f"Bearer {self.settings.get('api_key', '')}", "Content-Type": "application/json"}
        chunks: list[str] = []
        # 流式时屏蔽 thinking/reasoning 增量，只展示最终 content
        try:
            with httpx.stream("POST", url, headers=headers, json=payload, timeout=float(self.settings.get("timeout_seconds", 60))) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    event = json.loads(data)
                    choices = event.get("choices") or []
                    if not choices:
                        continue
                    delta_obj = choices[0].get("delta", {}) or {}
                    # 忽略 reasoning/thinking 字段，避免前端刷长链思考
                    if delta_obj.get("reasoning_content") or delta_obj.get("reasoning") or delta_obj.get("thinking"):
                        continue
                    delta = delta_obj.get("content") or ""
                    if not delta:
                        continue
                    # 若 content 内仍夹带 think 标签，整段丢弃到闭合前
                    if "<think>" in delta.lower() or "<thinking>" in delta.lower():
                        continue
                    if "</think>" in delta.lower() or "</thinking>" in delta.lower():
                        continue
                    chunks.append(delta)
                    on_delta(delta)
        except Exception as error:
            raise LLMServiceError(f"生成模型流式调用失败：{error}") from error
        return self._strip_thinking("".join(chunks))

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

    def role(self, role: str, payload: dict[str, Any], on_delta: Callable[[str], None] | None = None) -> dict[str, Any]:
        prompt = self._prompt(role) + "\n请使用简洁自然语言写可朗读的执行摘要。不要输出 JSON 或代码块，不要输出逐步推理或思考过程，总长度不超过 80 字。\n输入：" + json.dumps(payload, ensure_ascii=False)
        content = self.complete_text_stream(prompt, on_delta) if on_delta else self.complete_text(prompt)
        return {"summary": self._strip_thinking(content)}
