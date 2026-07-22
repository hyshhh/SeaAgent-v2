"""解析问答中以分隔符列出的多个独立船舶目标。"""
from __future__ import annotations

import re
from typing import Any

_QUERY_WORDS = ("查", "找", "有没有", "是否", "出现", "存在", "列出")
_SPLIT_PATTERN = re.compile(r"[，,、；;]+|\s+(?:和|及|以及)\s+")
_LEADING_PATTERN = re.compile(r"^(?:请|麻烦|帮我|帮忙)?\s*(?:查找|查一下|找一下|帮我查一下|帮我找一下|帮我找|查询|寻找)\s*")
_TRAILING_PATTERN = re.compile(r"(?:是否(?:已经)?(?:出现|存在)(?:[／/或](?:出现|存在))?|有没有(?:出现|存在)?|是否出现过|是否存在过|出现了吗|存在吗|出现过|存在)?\s*[？?。！!]*$")
_HULL_PATTERN = re.compile(r"[0-9A-Za-z-]{3,16}$")


def extract_target_items(question: str) -> list[dict[str, str | None]]:
    """从用户问题中提取逗号、顿号或并列词分隔的独立目标。"""
    text = re.sub(r"\s+", " ", str(question or "")).strip()
    if not text or not any(word in text for word in _QUERY_WORDS):
        return []
    if not _SPLIT_PATTERN.search(text):
        return []
    text = _LEADING_PATTERN.sub("", text)
    text = _TRAILING_PATTERN.sub("", text).strip()
    parts = [part.strip(" \t\r\n：:，,、；;。！？?!") for part in _SPLIT_PATTERN.split(text)]
    return _deduplicate(_build_item(part, index + 1) for index, part in enumerate(parts))


def normalize_target_items(value: Any) -> list[dict[str, str | None]]:
    """校验模型给出的多目标数组，禁止模型把多个目标合并为一个字符串。"""
    if not isinstance(value, list):
        return []
    raw_items = []
    for index, item in enumerate(value):
        if isinstance(item, str):
            raw_items.append(_build_item(item, index + 1))
            continue
        if not isinstance(item, dict):
            continue
        label = str(item.get("label") or item.get("targetText") or item.get("description") or item.get("hullNumber") or "").strip()
        built = _build_item(label, index + 1)
        kind = str(item.get("targetKind") or item.get("kind") or built.get("kind") or "").strip().lower()
        if kind == "hull" and _HULL_PATTERN.fullmatch(str(item.get("hullNumber") or label).replace(" ", "")):
            hull = str(item.get("hullNumber") or label).replace(" ", "").upper()
            built = {"targetId": f"target-{index + 1}", "label": hull, "kind": "hull", "hullNumber": hull, "description": None}
        raw_items.append(built)
    return _deduplicate(raw_items)


def _build_item(value: str, index: int) -> dict[str, str | None]:
    label = _LEADING_PATTERN.sub("", str(value or "")).strip(" \t\r\n：:，,、；;。！？?!")
    label = re.sub(r"^(?:第\s*\d+\s*(?:艘|条|只)\s*)", "", label).strip()
    explicit = re.sub(r"^(?:舷号|弦号)\s*[:：]?\s*", "", label, flags=re.IGNORECASE).strip()
    compact = explicit.replace(" ", "")
    if _HULL_PATTERN.fullmatch(compact):
        hull = compact.upper()
        return {"targetId": f"target-{index}", "label": hull, "kind": "hull", "hullNumber": hull, "description": None}
    return {"targetId": f"target-{index}", "label": label, "kind": "description", "hullNumber": None, "description": label or None}


def _deduplicate(items: Any) -> list[dict[str, str | None]]:
    result: list[dict[str, str | None]] = []
    seen: set[tuple[str, str]] = set()
    for item in items:
        label = str(item.get("label") or "").strip()
        kind = str(item.get("kind") or "description")
        if not label:
            continue
        key = kind, label.upper() if kind == "hull" else label
        if key in seen:
            continue
        seen.add(key)
        result.append({
            "targetId": f"target-{len(result) + 1}",
            "label": label,
            "kind": kind,
            "hullNumber": item.get("hullNumber") if kind == "hull" else None,
            "description": item.get("description") if kind == "description" else None,
        })
    return result
