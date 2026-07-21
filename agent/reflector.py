"""检查证据充分性并控制循环退出。"""
from __future__ import annotations

import json
import re
from typing import Any, Callable

from services import AgentLLMService


class Reflector:
    ALLOWED = {"sufficient", "replan", "conflict", "uncertain"}
    _CONTINUE_ACTION = re.compile(r"^\s*(?:下一轮|继续(?:规划|执行|调用|读取|获取)?|重新规划|补充(?:读取|获取|调用)?)")
    _STATE_ALIASES = {
        "完成": "sufficient",
        "已完成": "sufficient",
        "停止": "sufficient",
        "继续": "replan",
        "重规划": "replan",
        "冲突": "conflict",
        "不确定": "uncertain",
        "无法确认": "uncertain",
    }

    def __init__(self, llm: AgentLLMService):
        self.llm = llm

    def review(
        self,
        default_state: str,
        reason: str,
        observation_summary: dict[str, Any],
        evidence_gap: str | None = None,
        on_delta: Callable[[str], None] | None = None,
        *,
        autonomous: bool = False,
        expected_outcome: str | None = None,
        success_criteria: str | None = None,
        next_agent_focus: str | None = None,
        previous_rounds: list[dict[str, Any]] | None = None,
        acceptance_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if autonomous:
            return self._review_autonomous(
                default_state,
                reason,
                observation_summary,
                evidence_gap,
                on_delta,
                expected_outcome,
                success_criteria,
                next_agent_focus,
                previous_rounds,
                acceptance_context,
            )
        return self._review_guided(default_state, reason, observation_summary, evidence_gap, on_delta)

    def _review_guided(
        self,
        default_state: str,
        reason: str,
        observation_summary: dict[str, Any],
        evidence_gap: str | None,
        on_delta: Callable[[str], None] | None,
    ) -> dict[str, Any]:
        """硬编码辅助模式只审计，循环状态仍由控制器的固定链路决定。"""
        state = default_state if default_state in self.ALLOWED else "uncertain"
        reflection = {"state": state, "reason": reason, "evidenceGap": evidence_gap}
        try:
            reflection["modelReflection"] = self.llm.role(
                "reflector",
                {
                    "controlledState": state,
                    "reason": reason,
                    "observation": observation_summary,
                    "evidenceGap": evidence_gap,
                },
                on_delta,
            )
        except Exception as error:
            reflection["modelFallback"] = str(error)
        return reflection

    def _review_autonomous(
        self,
        proposed_state: str,
        planner_reason: str,
        observation_summary: dict[str, Any],
        evidence_gap: str | None,
        on_delta: Callable[[str], None] | None,
        expected_outcome: str | None,
        success_criteria: str | None,
        next_agent_focus: str | None,
        previous_rounds: list[dict[str, Any]] | None,
        acceptance_context: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """自主规划模式由 ReflectAgent 读取观察事实并实际决定下一轮状态。"""
        task = observation_summary.get("task") if isinstance(observation_summary.get("task"), dict) else {}
        payload = {
            "task": task,
            "plannerProposal": proposed_state if proposed_state in self.ALLOWED else "replan",
            "plannerReason": planner_reason,
            "expectedOutcome": expected_outcome,
            "successCriteria": success_criteria,
            "nextAgentFocus": next_agent_focus,
            "observation": observation_summary,
            "previousRounds": previous_rounds or [],
            "acceptanceProgress": acceptance_context or {},
            "evidenceGap": evidence_gap,
        }
        reflection = {"state": "uncertain", "reason": planner_reason, "evidenceGap": evidence_gap}
        try:
            prompt = self.llm._prompt("reflector_autonomous")
            request = prompt + "\n输入：" + json.dumps(payload, ensure_ascii=False)
            if on_delta and hasattr(self.llm, "complete_text_stream"):
                raw = self.llm.complete_text_stream(request, on_delta)
            else:
                raw = self.llm.complete_text(request)
            parsed = self._parse_autonomous_review(raw)
            parsed = self._enforce_acceptance(parsed, acceptance_context)
            reflection.update(parsed)
            summary = raw.strip()
            if parsed.get("stateCorrection"):
                summary = f"{summary}\n\n[状态一致性修正] {parsed['stateCorrection']}"
            if parsed.get("acceptanceOverride"):
                summary = (
                    f"{summary}\n\n[验收目标约束] {parsed.get('reason')} "
                    f"下一步：{parsed.get('nextAction')}"
                )
            reflection["modelReflection"] = {"summary": summary}
        except Exception as error:
            reflection["reason"] = "ReflectAgent 未返回有效审计结果"
            reflection["evidenceGap"] = evidence_gap or "缺少可用的反思结论"
            reflection["modelFallback"] = str(error)
            reflection = self._enforce_acceptance(reflection, acceptance_context)
        return reflection

    @staticmethod
    def _enforce_acceptance(
        reflection: dict[str, Any],
        acceptance_context: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """以初始意图的验收清单约束退出状态，并生成下一轮证据方向。"""
        if not acceptance_context:
            return reflection
        result = dict(reflection)
        satisfied = bool(acceptance_context.get("acceptanceSatisfied"))
        pending = [str(value) for value in acceptance_context.get("pendingRequirements") or []]
        pending_labels = [str(value) for value in acceptance_context.get("pendingRequirementLabels") or pending]
        next_action = str(acceptance_context.get("nextAction") or "").strip()
        expected = str(acceptance_context.get("expectedOutcome") or "").strip()
        if satisfied:
            if result.get("state") != "conflict":
                result["state"] = "sufficient"
                result["reason"] = f"已满足初始验收目标：{expected or '当前任务验收条件'}"
                result["evidenceGap"] = None
                result["nextAction"] = "停止并返回满足验收目标的结果"
            return result
        if not pending:
            return result
        gap = "、".join(pending_labels)
        result["evidenceGap"] = gap
        result["reason"] = (
            f"尚未满足初始验收目标“{expected or '当前任务'}”；"
            f"仍缺少：{gap}。"
        )
        if next_action.startswith(("下一轮", "继续", "重新规划", "补充")):
            result["state"] = "replan"
            result["nextAction"] = next_action
        else:
            result["state"] = "uncertain"
            result["nextAction"] = "停止并返回当前证据不足的结果"
        result["acceptanceOverride"] = True
        return result

    @classmethod
    def _parse_autonomous_review(cls, text: str) -> dict[str, Any]:
        """解析固定四行输出；格式错误时保守返回 uncertain。"""
        content = str(text or "").strip()

        try:
            value = json.loads(content)
        except json.JSONDecodeError:
            value = None
        if isinstance(value, dict):
            state = str(value.get("state") or value.get("状态") or "uncertain").lower()
            basis = str(value.get("reason") or value.get("依据") or value.get("action") or value.get("动作") or "").strip()
            gap = str(value.get("evidenceGap") or value.get("缺口") or "").strip()
            action = str(value.get("action") or value.get("动作") or "").strip()
            if gap.lower() in {"无", "none", "null", "没有", "无缺口"}:
                gap = ""
            state, correction = cls._normalize_autonomous_state(state, action)
            parsed = {
                "state": state,
                "reason": basis or "ReflectAgent 未给出明确依据",
                "evidenceGap": gap or None,
                "nextAction": action or None,
            }
            if correction:
                parsed["stateCorrection"] = correction
            return parsed

        def field(*names: str) -> str:
            for name in names:
                match = re.search(rf"(?im)^\s*{name}\s*[：:]\s*(.+?)\s*$", content)
                if match:
                    return match.group(1).strip()
            return ""

        state_text = field("状态", "state").lower()
        state_match = re.search(r"\b(sufficient|replan|conflict|uncertain)\b", state_text, re.IGNORECASE)
        state = state_match.group(1).lower() if state_match else "uncertain"
        basis = field("依据", "reason")
        gap = field("缺口", "evidenceGap")
        action = field("动作", "action")
        if gap.lower() in {"无", "none", "null", "没有", "无缺口"}:
            gap = ""
        reason = basis or action or "ReflectAgent 未给出明确依据"
        state, correction = cls._normalize_autonomous_state(state, action)
        parsed = {
            "state": state,
            "reason": reason,
            "evidenceGap": gap or None,
            "nextAction": action or None,
        }
        if correction:
            parsed["stateCorrection"] = correction
        return parsed

    @classmethod
    def _normalize_autonomous_state(cls, state: str, action: str) -> tuple[str, str | None]:
        """防止模型把“继续下一轮”和“无法确认”同时输出而提前终止循环。"""
        normalized = cls._STATE_ALIASES.get(state, state)
        normalized = normalized if normalized in cls.ALLOWED else "uncertain"
        if normalized != "replan" and cls._CONTINUE_ACTION.search(str(action or "")):
            return "replan", "动作明确要求进入下一轮，状态由终止状态更正为 replan。"
        return normalized, None
