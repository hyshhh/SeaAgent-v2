"""解析问答中以分隔符列出的多个独立船舶目标。

切分词、正则与舷号形态均来自 skills/intent_agent/target_parsing.yaml，
本模块只加载规则并执行，不在代码中维护业务词表。
"""
from __future__ import annotations

import re
from functools import lru_cache
from typing import Any

from agent.skill_loader import load_skill_yaml


@lru_cache(maxsize=1)
def _rules() -> dict[str, Any]:
    data = load_skill_yaml("intent_agent", "target_parsing")
    return data if isinstance(data, dict) else {}


@lru_cache(maxsize=1)
def _compiled() -> dict[str, Any]:
    rules = _rules()
    flags = re.IGNORECASE
    return {
        "query_words": tuple(str(w) for w in (rules.get("query_words") or [])),
        "split": re.compile(str(rules.get("split_pattern") or r"[，,、；;]+")),
        "leading": re.compile(str(rules.get("leading_pattern") or r"^"), flags),
        "trailing": re.compile(str(rules.get("trailing_pattern") or r"$")),
        "hull": re.compile(str(rules.get("hull_pattern") or r"[0-9A-Za-z-]{3,16}$")),
        "ordinal": re.compile(str(rules.get("ordinal_prefix_pattern") or r"^")),
        "hull_label": re.compile(str(rules.get("hull_label_prefix_pattern") or r"^"), flags),
        "trim_chars": str(rules.get("trim_chars") or " \t\r\n：:，,、；;。！？?!"),
    }


def extract_target_items(question: str) -> list[dict[str, str | None]]:
    """从用户问题中提取逗号、顿号或并列词分隔的独立目标。"""
    cfg = _compiled()
    text = re.sub(r"\s+", " ", str(question or "")).strip()
    if not text or not any(word in text for word in cfg["query_words"]):
        return []
    if not cfg["split"].search(text):
        return []
    text = cfg["leading"].sub("", text)
    text = cfg["trailing"].sub("", text).strip()
    parts = [part.strip(cfg["trim_chars"]) for part in cfg["split"].split(text)]
    return _deduplicate(_build_item(part, index + 1) for index, part in enumerate(parts))


def normalize_target_items(value: Any) -> list[dict[str, str | None]]:
    """校验模型给出的多目标数组，禁止模型把多个目标合并为一个字符串。"""
    if not isinstance(value, list):
        return []
    raw_items = []
    cfg = _compiled()
    for index, item in enumerate(value):
        if isinstance(item, str):
            raw_items.append(_build_item(item, index + 1))
            continue
        if not isinstance(item, dict):
            continue
        label = str(
            item.get("label")
            or item.get("targetText")
            or item.get("description")
            or item.get("hullNumber")
            or ""
        ).strip()
        built = _build_item(label, index + 1)
        kind = str(item.get("targetKind") or item.get("kind") or built.get("kind") or "").strip().lower()
        candidate = str(item.get("hullNumber") or label).replace(" ", "")
        if kind == "hull" and cfg["hull"].fullmatch(candidate):
            hull = candidate.upper()
            built = {
                "targetId": f"target-{index + 1}",
                "label": hull,
                "kind": "hull",
                "hullNumber": hull,
                "description": None,
            }
        raw_items.append(built)
    return _deduplicate(raw_items)


def extract_hull_number(question: str) -> str | None:
    """从问题中抽取可能的舷号（规则辅助，供 Intent/Plan 工具调用）。"""
    explicit = re.search(r"[舷弦]号\s*[:：]?\s*([0-9A-Za-z-]+)", question, re.I)
    if explicit:
        return explicit.group(1).upper()
    if not any(token in question for token in ("船", "出现", "轨迹", "编号", "时间", "有没有", "是否", "库")):
        return None
    for value in re.findall(r"(?<![\d:：-])([0-9A-Za-z]{3,8})(?![\d:：-])", question):
        if value.isdigit() and len(value) < 3:
            continue
        if re.fullmatch(r"\d{1,2}", value):
            continue
        return value.upper()
    return None


def parse_targets(question: str) -> dict[str, Any]:
    """Intent 工具协议：parseTargets。"""
    return {
        "ok": True,
        "targetItems": extract_target_items(question),
        "hint": "可选参考；请你确认后写入 result.targetItems",
    }


def extract_hull(question: str) -> dict[str, Any]:
    """Intent 工具协议：extractHull。"""
    return {"ok": True, "hullNumber": extract_hull_number(question)}


def _build_item(value: str, index: int) -> dict[str, str | None]:
    cfg = _compiled()
    label = cfg["leading"].sub("", str(value or "")).strip(cfg["trim_chars"])
    label = cfg["ordinal"].sub("", label).strip()
    explicit = cfg["hull_label"].sub("", label).strip()
    compact = explicit.replace(" ", "")
    if cfg["hull"].fullmatch(compact):
        hull = compact.upper()
        return {
            "targetId": f"target-{index}",
            "label": hull,
            "kind": "hull",
            "hullNumber": hull,
            "description": None,
        }
    return {
        "targetId": f"target-{index}",
        "label": label,
        "kind": "description",
        "hullNumber": None,
        "description": label or None,
    }


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
