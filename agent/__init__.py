"""SeaAgent 四子智能体：LangChain 工具 + LangGraph 编排。

- IntentAgent / PlanAgent / ObserveAgent / ReflectAgent
- handoff 工具在 Agent 间移交
- AgentController.answer() 保持前端契约
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

__all__ = ["AgentController"]

if TYPE_CHECKING:
    from .controller import AgentController


def __getattr__(name: str) -> Any:
    if name == "AgentController":
        from .controller import AgentController
        return AgentController
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
