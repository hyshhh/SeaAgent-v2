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
        "getTrack", "getFrames", "listRegistry", "getTrack", "getFrames", "matchImage"
    ]
    assert state["reflection"]["acceptanceProgress"]["acceptanceSatisfied"] is True
