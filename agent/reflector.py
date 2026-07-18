"""检查证据充分性并控制循环退出。"""
from __future__ import annotations
from typing import Any
from services import AgentLLMService

class Reflector:
    ALLOWED = {"sufficient", "replan", "conflict", "uncertain"}
    def __init__(self, llm: AgentLLMService):
        self.llm = llm

    def review(self, default_state: str, reason: str, observation_summary: dict[str, Any], evidence_gap: str | None = None) -> dict[str, Any]:
        state = default_state if default_state in self.ALLOWED else "uncertain"
        reflection = {"state": state, "reason": reason, "evidenceGap": evidence_gap}
        try:
            reflection["modelReflection"] = self.llm.role("reflector", {"proposedState": state, "reason": reason, "observation": observation_summary, "evidenceGap": evidence_gap})
        except Exception as error:
            reflection["modelFallback"] = str(error)
        return reflection
