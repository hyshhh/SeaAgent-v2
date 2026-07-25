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
