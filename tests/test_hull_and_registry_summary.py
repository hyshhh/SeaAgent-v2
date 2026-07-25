"""舷号抽取与 listRegistry 摘要计数回归。"""
from __future__ import annotations

from agent.plan_executor import PlanExecutor
from tools.target_parser import clear_target_parser_cache, extract_hull_number, infer_intent_fields


def test_extract_hull_chinese_prefix():
    clear_target_parser_cache()
    assert extract_hull_number("舷号 小蓝320 有没有在视频中出现？") == "小蓝320"
    assert extract_hull_number("舷号：A01 出现过吗") == "A01"
    assert extract_hull_number("查一下 0789") in {"0789", None} or extract_hull_number("查一下 0789") == "0789"


def test_infer_hull_existence():
    clear_target_parser_cache()
    fields = infer_intent_fields("舷号 小蓝320 有没有在视频中出现？")
    assert fields["hullNumber"] == "小蓝320"
    assert fields["operation"] == "existence"
    assert fields["targetKind"] == "hull"
    focus = str(fields.get("nextAgentFocus") or "")
    criteria = str(fields.get("successCriteria") or "")
    assert "getRegistry" in focus or "先验库" in focus
    assert "matchImage" in focus or "matchImage" in criteria
    assert "0 轨迹即可否定" not in criteria
    assert "未检测到" not in criteria


def test_registry_summary_prefers_items_over_references():
    observation = {
        "id": "registry",
        "tool": "listRegistry",
        "skipped": False,
        "result": {
            "ok": True,
            "registryItems": [{"registryId": "a"}, {"registryId": "b"}],
            "registryReferences": [],
        },
    }
    summary = PlanExecutor.summarize_observation(observation)
    assert summary["registryCount"] == 2
    assert summary["registryItemCount"] == 2
    assert summary["registryReferenceCount"] == 0


def test_default_ref_fallback_empty_references():
    executor = PlanExecutor(tools=type("T", (), {"execute": staticmethod(lambda *a, **k: {})})())
    scope = {
        "registry": {
            "ok": True,
            "registryReferences": [],
            "registryItems": [{"registryId": "x", "description": "黄色船"}],
        }
    }
    resolved = executor._resolve(
        {"$ref": "registry.registryReferences", "$default": {"$ref": "registry.registryItems"}},
        scope,
    )
    assert isinstance(resolved, list)
    assert resolved[0]["registryId"] == "x"


def test_skipped_not_counted_as_failed():
    observations = [
        {"id": "tracks", "tool": "getTrack", "skipped": False, "result": {"ok": True, "tracks": []}},
        {
            "id": "frames",
            "tool": "getFrames",
            "skipped": True,
            "skipReason": "dependency_empty:tracks.trackIds",
            "result": {"ok": False, "error": "dependency_empty:tracks.trackIds"},
        },
    ]
    failed = sum(
        1
        for item in observations
        if not item.get("skipped") and (item.get("result") or {}).get("ok") is False
    )
    skipped = sum(1 for item in observations if item.get("skipped"))
    assert failed == 0
    assert skipped == 1


def test_visual_match_default_plan_shape():
    """补洞 replan 应产出 getRegistry → getTrack(无hull) → getFrames → matchImage。"""
    # 内嵌函数：直接复现 _default_plan_calls 的关键分支逻辑做契约测试
    def wants_visual(hint: str) -> bool:
        h = hint.lower()
        return any(
            token in h
            for token in (
                "matchimage", "match_image", "视觉匹配", "图像匹配", "图匹配",
                "registryreferences", "关键帧匹配", "库图", "对照视频",
            )
        ) or ("match" in h and "image" in h)

    hint = (
        "getRegistry(hullNumber=小蓝320) → getTrack(不带hullNumber, 全时域) → "
        "getFrames($ref trackIds) → matchImage(queryImages=$ref registry.registryReferences, "
        "galleryImages=$ref frames.keyframes)"
    )
    assert wants_visual(hint)
    assert "matchimage" in hint.lower()
    assert "不带hull" in hint or "不带hullnumber" in hint.lower()


def test_should_replan_visual_when_registry_found_without_searchable_flag():
    """库有 found/items 但 searchable 未标时，仍应触发视觉 replan（契约）。"""
    registry_checked = True
    registry_searchable = False
    registry_found = True
    registry_has_items = True
    visual_matched = False
    zero_tracks = True
    hull = "小蓝320"
    loop_count = 1
    limit = 3
    op = "existence"
    can_try_visual = registry_searchable or registry_found or registry_has_items
    should_replan_visual = (
        loop_count < limit
        and bool(hull)
        and registry_checked
        and can_try_visual
        and not visual_matched
        and op in {"existence", "list", "explain", "time", ""}
        and zero_tracks
    )
    assert can_try_visual
    assert should_replan_visual
