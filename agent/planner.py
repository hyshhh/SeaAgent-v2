"""生成受控的最短工具计划。"""
from __future__ import annotations
import re
from typing import Any, Callable
from services import AgentLLMService

class Planner:
    def __init__(self, llm: AgentLLMService, allowed_tools: set[str]):
        self.llm = llm
        self.allowed_tools = allowed_tools

    def classify(self, question: str) -> dict[str, Any]:
        time_range = self._time_range(question)
        if "多少" in question or "数量" in question:
            question_type = "count"
        elif "未在库" in question or "不在库" in question:
            question_type = "out_of_registry"
        elif "在库" in question:
            question_type = "in_registry"
        elif "舷号" in question or "弦号" in question:
            question_type = "hull"
        else:
            question_type = "description"
        hull_match = re.search(r"[舷弦]号\s*[:：]?\s*([0-9A-Za-z-]+)", question, re.I)
        return {"questionType": question_type, "timeRange": time_range, "hullNumber": hull_match.group(1).upper() if hull_match else None, "description": question if question_type == "description" else None}

    def build(self, goal: str, calls: list[dict[str, Any]], scope: Any = None, evidence_gap: str | None = None, on_delta: Callable[[str], None] | None = None) -> dict[str, Any]:
        invalid = [call["tool"] for call in calls if call["tool"] not in self.allowed_tools]
        if invalid:
            raise ValueError(f"工具不在白名单：{invalid}")
        plan = {"goal": goal, "scope": scope, "calls": calls, "evidenceGap": evidence_gap, "stopCondition": "证据足够、证据冲突或达到最大轮次"}
        try:
            model_plan = self.llm.role("planner", {"goal": goal, "scope": scope, "calls": [{"id": item["id"], "tool": item["tool"]} for item in calls], "evidenceGap": evidence_gap}, on_delta)
            plan["modelPlan"] = model_plan
        except Exception as error:
            plan["modelFallback"] = str(error)
        return plan

    @staticmethod
    def _time_range(question: str) -> tuple[float, float] | None:
        match = re.search(r"(\d+(?:\.\d+)?)\s*分(?:钟)?\s*(?:到|至|-|—|~)\s*(\d+(?:\.\d+)?)\s*分", question)
        if match:
            return float(match.group(1)) * 60, float(match.group(2)) * 60
        match = re.search(r"(\d+(?:\.\d+)?)\s*秒\s*(?:到|至|-|—|~)\s*(\d+(?:\.\d+)?)\s*秒", question)
        return (float(match.group(1)), float(match.group(2))) if match else None
