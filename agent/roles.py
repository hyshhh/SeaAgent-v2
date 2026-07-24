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

    prompt = (
        f"你是海域船舶监控系统的{title}。\n"
        f"职责：{responsibility}\n\n"
        f"## 本轮已启用 Skills：{', '.join(skill_ids) or '无'}\n"
        f"{skills_text or '（无 skill 正文）'}\n"
        f"{catalog_hint}\n\n"
        "## 工作方式\n"
        "- 可多轮调用工具收集信息，再给出结论。\n"
        "- 完成后调用对应的移交工具（handoff_*）把控制权交给下一 Agent。\n"
        "- 只使用分配给你的工具，不要编造工具结果。\n"
        "- 回答与 reason 使用简体中文。\n"
    )
    return prompt, skill_ids


INTENT_RESPONSIBILITY = (
    "理解用户问题：判定时间范围、多目标、舷号/描述、操作类型；"
    "完成后调用 handoff_to_plan，arguments 中携带结构化意图。"
)

PLAN_RESPONSIBILITY = (
    "根据意图与历史证据规划本轮检索步骤，用 planHint 写清工具顺序与参数要点；"
    "规划完成后必须调用 handoff_to_observe；若无法继续则 handoff_to_reflect。"
    "不要反复 loadSkill；最多查阅一次细则后立即移交。"
)

OBSERVE_RESPONSIBILITY = (
    "严格执行检索与匹配工具，把结果写入工作记忆；"
    "完成后调用 handoff_to_reflect，summary 中简述观察事实。"
)

REFLECT_RESPONSIBILITY = (
    "审计证据是否充分。state=replan 时 handoff_to_plan；"
    "state 为 sufficient/conflict/uncertain 时 handoff_finish 结束循环。"
)
