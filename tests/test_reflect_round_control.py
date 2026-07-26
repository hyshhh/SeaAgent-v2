import json
from unittest.mock import patch

from langchain_core.messages import AIMessage, ToolMessage

from agent.controller import AgentController
from agent.graph import _build_acceptance_progress, _default_plan_calls, _registry_membership_list_mode, run_sea_agent
from tools.target_parser import clear_target_parser_cache, infer_intent_fields


def _out_fields():
    clear_target_parser_cache()
    return infer_intent_fields("有哪些未在库船出现在视频中？")


def test_registry_out_intent_separates_acceptance_and_current_focus():
    fields = _out_fields()

    assert fields["operation"] == "list"
    assert fields["registryRelation"] == "out"
    assert fields["questionType"] == "registry_out_list"
    assert not fields.get("description")
    assert fields["successCriteria"] != fields["nextAgentFocus"]
    assert "listRegistry" in fields["successCriteria"]
    assert "matchImage" in fields["successCriteria"]
    assert "mismatch" in fields["successCriteria"]
    assert "第一轮" in fields["nextAgentFocus"]
    assert "getTrack" in fields["nextAgentFocus"]
    assert "getFrames" in fields["nextAgentFocus"]


def test_registry_out_first_round_enumerates_tracks_before_full_registry_audit():
    calls = _default_plan_calls(_out_fields(), top_k=1, broad_match_top_k=0)

    assert [call["tool"] for call in calls] == ["getTrack", "getFrames"]


def test_registry_out_replan_reuses_existing_tracks_and_frames_with_custom_call_ids():
    calls = _default_plan_calls(
        _out_fields(),
        top_k=1,
        broad_match_top_k=0,
        replan_hint="补全 listRegistry 与 matchImage",
        working_scope={
            "all_video_tracks": {
                "ok": True,
                "trackIds": ["1", "2"],
                "tracks": [{"trackId": "1"}, {"trackId": "2"}],
            },
            "all_video_frames": {
                "ok": True,
                "keyframes": [
                    {"trackId": "1", "keyframeId": "f1"},
                    {"trackId": "2", "keyframeId": "f2"},
                ],
            },
        },
    )

    assert [call["tool"] for call in calls] == ["listRegistry", "matchImage"]
    assert calls[1]["arguments"]["galleryImages"] == {"$ref": "all_video_frames.keyframes"}
    assert calls[1]["arguments"]["topK"] == 0


def test_registry_out_replan_generates_no_registry_or_match_calls_for_known_zero_tracks():
    calls = _default_plan_calls(
        _out_fields(),
        top_k=1,
        broad_match_top_k=0,
        replan_hint="补全 listRegistry 与 matchImage",
        working_scope={
            "all_video_tracks": {"ok": True, "trackIds": [], "tracks": []},
        },
    )

    assert calls == []


def test_reflect_acceptance_requests_next_round_when_registry_audit_is_missing():
    fields = _out_fields()
    progress = _build_acceptance_progress(
        fields,
        {"getTrack", "getFrames"},
        track_count=11,
        registry_checked=False,
        registry_listed=False,
        registry_has_items=False,
        can_try_visual=False,
        visual_attempted=False,
        match_image_attempted=False,
        match_image_usable=False,
        has_tool_evidence=True,
    )

    assert _registry_membership_list_mode(fields) == "out"
    assert progress["acceptanceSatisfied"] is False
    assert progress["pendingRequirements"] == ["已获取完整先验库名录"]
    assert "listRegistry" in progress["nextAction"]
    assert "matchImage" in progress["nextAction"]


def test_reflect_acceptance_short_circuits_registry_audit_when_video_has_no_tracks():
    progress = _build_acceptance_progress(
        _out_fields(),
        {"getTrack"},
        track_count=0,
        registry_checked=False,
        registry_listed=False,
        registry_has_items=False,
        can_try_visual=False,
        visual_attempted=False,
        match_image_attempted=False,
        match_image_usable=False,
        has_tool_evidence=True,
    )

    assert progress["acceptanceSatisfied"] is True
    assert progress["pendingRequirements"] == []
    assert progress["videoEmptyShortCircuit"] is True
    assert "无需 listRegistry" in progress["nextAction"]
    assert "matchImage" in progress["nextAction"]


def test_reflect_acceptance_stops_with_uncertainty_when_image_inputs_are_blocked():
    progress = _build_acceptance_progress(
        _out_fields(),
        {"getTrack", "getFrames", "listRegistry", "matchImage"},
        track_count=2,
        registry_checked=True,
        registry_listed=True,
        registry_has_items=True,
        can_try_visual=True,
        visual_attempted=True,
        match_image_attempted=True,
        match_image_usable=False,
        has_tool_evidence=True,
        match_image_blocked=True,
    )

    assert progress["acceptanceSatisfied"] is True
    assert progress["pendingRequirements"] == []
    assert progress["terminalState"] == "uncertain"
    assert progress["matchImageBlocked"] is True


def test_reflect_acceptance_passes_after_full_registry_image_match():
    fields = _out_fields()
    progress = _build_acceptance_progress(
        fields,
        {"getTrack", "getFrames", "listRegistry", "matchImage"},
        track_count=11,
        registry_checked=True,
        registry_listed=True,
        registry_has_items=True,
        can_try_visual=True,
        visual_attempted=True,
        match_image_attempted=True,
        match_image_usable=True,
        has_tool_evidence=True,
    )

    assert progress["pendingRequirements"] == []
    assert progress["acceptanceSatisfied"] is True


def test_registry_out_synthesis_returns_only_mismatch_tracks():
    controller = AgentController.__new__(AgentController)
    controller.meta = {
        "operation": "list",
        "targetScope": "both",
        "targetKind": "all",
        "registryRelation": "out",
        "questionType": "registry_out_list",
    }
    controller.working_scope = {
        "tracks": {
            "ok": True,
            "tracks": [
                {"trackId": "1"},
                {"trackId": "2"},
                {"trackId": "3"},
            ],
        },
        "registry": {
            "ok": True,
            "registryItems": [{"registryId": "r1", "hullNumber": "0857"}],
        },
        "match": {
            "ok": True,
            "matches": [
                {"matchedTrackId": "1", "matchedRegistryId": "r1", "embeddingScore": 0.91, "scoreBand": "match"},
                {"matchedTrackId": "2", "matchedRegistryId": "r1", "embeddingScore": 0.31, "scoreBand": "mismatch"},
                {"matchedTrackId": "3", "matchedRegistryId": "r1", "embeddingScore": 0.61, "scoreBand": "uncertain"},
            ],
        },
    }
    controller.tool_records = [
        {"tool": "listRegistry", "ok": True},
        {"tool": "matchImage", "ok": True, "skipped": False},
    ]
    controller.tool_chain = ["getTrack", "getFrames", "listRegistry", "matchImage"]
    controller.rounds = []
    controller.display_limit = 3
    controller.display_record = {"displayId": "display-test", "mode": "lazy"}
    controller.display_groups = []
    controller.session_id = "session-test"
    controller.question = "有哪些未在库船出现在视频中？"
    controller.event_handler = None
    controller._pending_registry_items = []

    result = controller._synthesize("sufficient", "全库对照完成")

    assert result["outOfRegistryCount"] == 1
    assert [item["trackId"] for item in result["tracks"]] == ["2"]
    assert [item["trackId"] for item in result["uncertainTracks"]] == ["3"]
    assert result["inRegistryMatchCount"] == 1
    assert result["uncertainty"] == "uncertain"




def test_registry_out_synthesis_orders_uncertain_by_lowest_best_score_and_hides_full_registry():
    controller = AgentController.__new__(AgentController)
    controller.meta = {
        "operation": "list",
        "targetScope": "both",
        "targetKind": "all",
        "registryRelation": "out",
        "questionType": "registry_out_list",
    }
    controller.working_scope = {
        "tracks": {"ok": True, "tracks": [{"trackId": "1"}, {"trackId": "2"}, {"trackId": "3"}]},
        "registry": {"ok": True, "registryItems": [{"registryId": "r1"}]},
        "match": {
            "ok": True,
            "matchMode": "image_to_image",
            "registryCoverageComplete": False,
            "registryCoverageRatio": 0.5,
            "scoredRegistryCount": 1,
            "totalRegistryCount": 2,
            "unscoredRegistryIds": ["r2"],
            "matches": [
                {"matchedTrackId": "1", "embeddingScore": 0.709, "scoreBand": "uncertain"},
                {"matchedTrackId": "2", "embeddingScore": 0.596, "scoreBand": "uncertain"},
                {"matchedTrackId": "3", "embeddingScore": 0.655, "scoreBand": "uncertain"},
            ],
        },
    }
    controller.tool_records = [{"tool": "matchImage", "ok": True, "skipped": False}]
    controller.tool_chain = ["getTrack", "getFrames", "listRegistry", "matchImage"]
    controller.rounds = []
    controller.display_limit = 3
    controller.display_record = {"displayId": "display-test", "mode": "lazy"}
    controller.display_groups = []
    controller.session_id = "session-test"
    controller.question = "有哪些未在库船出现在视频中？"
    controller.event_handler = None
    controller._pending_registry_items = []

    result = controller._synthesize("sufficient", "全库对照完成")

    assert [item["trackId"] for item in result["uncertainTracks"]] == ["2", "3", "1"]
    assert [item["embeddingScore"] for item in result["matches"]] == [0.596, 0.655, 0.709]
    assert result["registryCoverageComplete"] is False
    assert result["uncertainty"] == "uncertain"
    assert "registryItems" not in result

def test_registry_out_synthesis_does_not_show_full_registry_when_video_has_no_tracks():
    controller = AgentController.__new__(AgentController)
    controller.meta = {
        "operation": "list",
        "targetScope": "both",
        "targetKind": "all",
        "registryRelation": "out",
        "questionType": "registry_out_list",
    }
    controller.working_scope = {
        "tracks": {"ok": True, "tracks": [], "trackIds": []},
        "registry": {
            "ok": True,
            "registryItems": [{"registryId": "r1", "hullNumber": "0857"}],
        },
        "match": {
            "ok": True,
            "matches": [],
            "error": "argument_missing:galleryImages",
            "visualAttempted": True,
        },
    }
    controller.tool_records = [
        {
            "tool": "getTrack",
            "ok": True,
            "skipped": False,
            "result": {"ok": True, "tracks": [], "trackIds": []},
            "summary": {"trackCount": 0},
            "round": 1,
        },
        {"tool": "listRegistry", "ok": True, "skipped": False, "round": 2},
        {
            "tool": "matchImage",
            "ok": True,
            "skipped": False,
            "error": "argument_missing:galleryImages",
            "result": {"error": "argument_missing:galleryImages", "matches": []},
            "round": 2,
        },
    ]
    controller.tool_chain = ["getTrack", "listRegistry", "matchImage"]
    controller.rounds = []
    controller.display_limit = 3
    controller.display_record = None
    controller.display_groups = []
    controller.session_id = "session-zero"
    controller.question = "有哪些未在库船出现在视频中？"
    controller.event_handler = None
    controller._pending_registry_items = []
    controller.tools = type("T", (), {})()

    result = controller._synthesize("uncertain", "旧链路错误地进入了全库匹配")

    assert result["uncertainty"] == "sufficient"
    assert result["outOfRegistryCount"] == 0
    assert result["tracks"] == []
    assert "registryItems" not in result
    assert result["display"]["registryReferenceCount"] == 0
    assert "未检测到船舶轨迹" in result["conclusion"]


def test_reflect_drives_registry_out_query_into_a_second_round_when_handoff_models_fail():
    fields = _out_fields()

    class _FakeAgent:
        def __init__(self, name):
            self.name = name

        def stream(self, *args, **kwargs):
            if self.name == "intent":
                payload = {"ok": True, "handoff": "plan", "intent": fields, "note": ""}
                messages = [
                    AIMessage(
                        content="",
                        tool_calls=[{
                            "name": "handoff_to_plan",
                            "args": {"intent": fields, "note": ""},
                            "id": "intent-handoff",
                            "type": "tool_call",
                        }],
                    ),
                    ToolMessage(
                        content=json.dumps(payload, ensure_ascii=False),
                        tool_call_id="intent-handoff",
                        name="handoff_to_plan",
                    ),
                ]
                yield "values", {"messages": messages}
                return
            raise RuntimeError(f"{self.name} 模拟未移交")

        def invoke(self, *args, **kwargs):
            raise RuntimeError(f"{self.name} 模拟未移交")

    class _FakeTools:
        @staticmethod
        def execute(name, arguments):
            if name == "getTrack":
                tracks = [{"trackId": "1"}, {"trackId": "2"}, {"trackId": "3"}]
                return {
                    "ok": True,
                    "trackIds": ["1", "2", "3"],
                    "tracks": tracks,
                    "returnedTrackCount": 3,
                    "totalTrackCount": 3,
                }
            if name == "getFrames":
                frames = [
                    {"trackId": "1", "keyframeId": "frame-1"},
                    {"trackId": "2", "keyframeId": "frame-2"},
                    {"trackId": "3", "keyframeId": "frame-3"},
                ]
                return {
                    "ok": True,
                    "keyframes": frames,
                    "keyframeIds": ["frame-1", "frame-2", "frame-3"],
                    "keyframesByTrack": {},
                }
            if name == "listRegistry":
                return {
                    "ok": True,
                    "registryItems": [{
                        "registryId": "registry-1",
                        "hullNumber": "0857",
                        "references": [{"referenceId": "ref-1", "registryId": "registry-1"}],
                    }],
                    "registryReferences": [{"referenceId": "ref-1", "registryId": "registry-1"}],
                }
            if name == "matchImage":
                return {
                    "ok": True,
                    "visualAttempted": True,
                    "scoredPairCount": 3,
                    "matches": [
                        {"matchedTrackId": "1", "matchedRegistryId": "registry-1", "embeddingScore": 0.91, "scoreBand": "match"},
                        {"matchedTrackId": "2", "matchedRegistryId": "registry-1", "embeddingScore": 0.31, "scoreBand": "mismatch"},
                        {"matchedTrackId": "3", "matchedRegistryId": "registry-1", "embeddingScore": 0.29, "scoreBand": "mismatch"},
                    ],
                }
            raise AssertionError(name)

    def _fake_create_agent(model, tools, system_prompt, name):
        handoff_tools = [tool for tool in tools if str(getattr(tool, "name", "")).startswith("handoff")]
        assert handoff_tools
        assert all(getattr(tool, "return_direct", False) for tool in handoff_tools)
        if name in {"plan", "reflect"}:
            assert "loadSkill" not in {str(getattr(tool, "name", "")) for tool in tools}
        return _FakeAgent(name)

    with patch("agent.graph.build_chat_model", return_value=object()), patch(
        "agent.graph.create_agent", side_effect=_fake_create_agent
    ):
        state = run_sea_agent(
            "有哪些未在库船出现在视频中？",
            object(),
            _FakeTools(),
            max_rounds=5,
            query_top_k=1,
            broad_match_top_k=0,
        )

    assert state["loop_count"] == 2
    assert len(state["rounds"]) == 2
    assert state["final_state"] == "sufficient"
    assert state["tool_chain"] == [
        "getTrack", "getFrames", "listRegistry", "matchImage"
    ]
    business_records = [
        record for record in state["tool_records"]
        if record.get("tool") in {"getTrack", "getFrames", "listRegistry", "matchImage"}
    ]
    assert [(record["tool"], record["round"]) for record in business_records] == [
        ("getTrack", 1), ("getFrames", 1), ("listRegistry", 2), ("matchImage", 2)
    ]
    assert state["reflection"]["acceptanceProgress"]["acceptanceSatisfied"] is True


def test_reflect_finishes_registry_out_query_in_first_round_when_video_has_no_tracks():
    fields = _out_fields()
    executed_tools = []

    class _FakeAgent:
        def __init__(self, name):
            self.name = name

        def stream(self, *args, **kwargs):
            if self.name == "intent":
                payload = {"ok": True, "handoff": "plan", "intent": fields, "note": ""}
                yield "values", {
                    "messages": [
                        AIMessage(
                            content="",
                            tool_calls=[{
                                "name": "handoff_to_plan",
                                "args": {"intent": fields, "note": ""},
                                "id": "intent-zero",
                                "type": "tool_call",
                            }],
                        ),
                        ToolMessage(
                            content=json.dumps(payload, ensure_ascii=False),
                            tool_call_id="intent-zero",
                            name="handoff_to_plan",
                        ),
                    ]
                }
                return
            raise RuntimeError(f"{self.name} 模拟未移交")

        def invoke(self, *args, **kwargs):
            raise RuntimeError(f"{self.name} 模拟未移交")

    class _FakeTools:
        @staticmethod
        def execute(name, arguments):
            executed_tools.append(name)
            if name == "getTrack":
                return {
                    "ok": True,
                    "trackIds": [],
                    "tracks": [],
                    "returnedTrackCount": 0,
                    "totalTrackCount": 0,
                }
            raise AssertionError(f"零轨迹后不应执行 {name}")

    def _fake_create_agent(model, tools, system_prompt, name):
        return _FakeAgent(name)

    with patch("agent.graph.build_chat_model", return_value=object()), patch(
        "agent.graph.create_agent", side_effect=_fake_create_agent
    ):
        state = run_sea_agent(
            "有哪些未在库船出现在视频中？",
            object(),
            _FakeTools(),
            max_rounds=5,
            query_top_k=1,
            broad_match_top_k=0,
        )

    assert state["loop_count"] == 1
    assert state["final_state"] == "sufficient"
    assert executed_tools == ["getTrack"]
    assert state["tool_chain"] == ["getTrack"]
    assert state["reflection"]["acceptanceProgress"]["videoEmptyShortCircuit"] is True
    assert "无需查询整库" in state["final_reason"] or "没有未在库船舶" in state["final_reason"]
