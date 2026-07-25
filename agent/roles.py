"""Agent 角色系统提示：从 skills catalog 按需拼装。"""
from __future__ import annotations

from typing import Any

from .skill_loader import catalog_index, compose_skills, select_skill_ids


def role_system_prompt(
    agent_key: str,
    title: str,
    responsibility: str,
    context: dict[str, Any] | None = None,
    extra_ids: list[str] | None = None,
) -> tuple[str, list[str]]:
    """返回 (system_prompt, selected_skill_ids)。"""
    skill_ids = select_skill_ids(agent_key, context or {}, extra_ids=extra_ids or [])
    skills_text = compose_skills(agent_key, skill_ids, include_catalog_index=True)
    available = catalog_index(agent_key, exclude_ids=skill_ids)
    catalog_hint = ""
    if available:
        lines = [f"- `{item['id']}`: {item.get('description') or item.get('title')}" for item in available]
        catalog_hint = "\n未加载可选技能（可用 loadSkill）：\n" + "\n".join(lines)

    work_style = (
        "## 工作方式\n"
        "- 可多轮调用工具收集信息，再给出结论。\n"
        "- 完成后调用对应的移交工具（handoff_*）把控制权交给下一 Agent。\n"
        "- 只使用分配给你的工具，不要编造工具结果。\n"
        "- 回答与 reason 使用简体中文。\n"
    )
    if agent_key == "plan_agent":
        # tool_chains 已 always 注入；仍保留可选 skill 目录，但不鼓励多轮 loadSkill
        work_style = (
            "## 工作方式\n"
            "- 你只有 handoff（及可选 loadSkill），不要执行检索工具。\n"
            "- **第一动作**优先 handoff_to_observe(goal, calls, planHint)；勿空转。\n"
            "- calls 用 $ref 串联；简体中文写 goal/planHint。\n"
            "- 可选 skill 仅在确实缺细则时 loadSkill 一次，然后立即 handoff。\n"
            "- 禁止只输出 JSON 正文而不调用工具。\n"
        )
    prompt = (
        f"你是海域船舶监控系统的{title}。\n"
        f"职责：{responsibility}\n\n"
        f"## 本轮已启用 Skills：{', '.join(skill_ids) or '无'}\n"
        f"{skills_text or '（无 skill 正文）'}\n"
        f"{catalog_hint}\n\n"
        f"{work_style}"
    )
    return prompt, skill_ids


INTENT_RESPONSIBILITY = (
    "理解用户问题：判定时间范围、多目标、舷号/描述、操作类型；"
    "完成后调用 handoff_to_plan，arguments 中携带结构化意图。"
)

PLAN_RESPONSIBILITY = (
    "根据意图规划本轮最小 calls（含 $ref），然后立即调用 handoff_to_observe；"
    "不要执行业务检索工具，不要长篇解释；无法规划时才 handoff_to_reflect。"
)

OBSERVE_RESPONSIBILITY = (
    "严格执行检索与匹配工具，把结果写入工作记忆；"
    "完成后调用 handoff_to_reflect，summary 中简述观察事实。"
)

REFLECT_RESPONSIBILITY = (
    "审计证据是否充分。"
    "视频 0 轨迹且未查库 → replan getRegistry；"
    "已查库有参考图但未 matchImage → replan 视觉匹配（getTrack不带hull→getFrames→matchImage）；"
    "state=replan 时 handoff_to_plan；sufficient/conflict/uncertain 时 handoff_finish。"
)
