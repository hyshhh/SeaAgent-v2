"""LangChain ChatModel：对接现有 OpenAI 兼容 LLM 配置。"""
from __future__ import annotations

from typing import Any

from langchain_openai import ChatOpenAI

from services import AgentLLMService


def build_chat_model(llm: AgentLLMService | None = None, config: dict[str, Any] | None = None) -> ChatOpenAI:
    """用 app.yaml 的 llm 配置构造 ChatOpenAI。"""
    if llm is not None:
        settings = llm.settings
    else:
        from config import load_config
        settings = (config or load_config())["llm"]
    return ChatOpenAI(
        model=str(settings.get("model") or "gpt-4o-mini"),
        api_key=str(settings.get("api_key") or "EMPTY"),
        base_url=str(settings.get("base_url") or "").rstrip("/"),
        temperature=float(settings.get("temperature") or 0.0),
        timeout=float(settings.get("timeout_seconds") or 60),
        max_retries=1,
        model_kwargs={
            "extra_body": {
                "chat_template_kwargs": {"enable_thinking": bool(settings.get("enable_thinking", False))},
                "enable_thinking": bool(settings.get("enable_thinking", False)),
            }
        },
    )
