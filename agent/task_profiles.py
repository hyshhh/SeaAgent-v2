"""任务模式判定的单一事实源（② 最小治理）。

「在库/未在库船舶列表」这类问法（哪些船在库/未在库）的判定条件，此前在
`agent/graph.py` 与 `agent/controller.py` 各写一遍，改一处漏一处。
这里集中定义 membership 模式的判定与问法类型工具，graph / controller
共同调用，避免语义漂移。
"""
from __future__ import annotations

from typing import Any

_MEMBERSHIP_QUESTION_TYPES = frozenset({"registry_in_list", "registry_out_list"})


def registry_membership_list_mode(intent: dict[str, Any] | None) -> str:
    """识别「在库/未在库船舶列表」任务，返回 in/out；其他任务返回空字符串。

    判定依据（与旧版 graph._registry_membership_list_mode 语义一致）：
    - registryRelation 为 in/out；
    - operation 为 list；
    - 没有具体舷号（列表任务面向全体候选）；
    - targetScope 为 both，或 questionType 与 relation 匹配（registry_in_list/out）。
    """
    if not isinstance(intent, dict):
        return ""
    relation = str(intent.get("registryRelation") or "")
    operation = str(intent.get("operation") or "")
    hull = str(intent.get("hullNumber") or "").strip()
    target_scope = str(intent.get("targetScope") or "")
    question_type = str(intent.get("questionType") or "")
    expected_type = f"registry_{relation}_list" if relation in {"in", "out"} else ""
    if (
        relation in {"in", "out"}
        and operation == "list"
        and not hull
        and (target_scope == "both" or question_type == expected_type)
    ):
        return relation
    return ""


def is_membership_question_type(question_type: Any) -> bool:
    """问法类型是否为在库/未在库列表（registry_in_list / registry_out_list）。"""
    return str(question_type or "") in _MEMBERSHIP_QUESTION_TYPES


def relation_for_membership(question_type: Any) -> str:
    """从问法类型反推在库关系：registry_in_list→in，registry_out_list→out，否则空串。"""
    qt = str(question_type or "")
    if qt == "registry_in_list":
        return "in"
    if qt == "registry_out_list":
        return "out"
    return ""
