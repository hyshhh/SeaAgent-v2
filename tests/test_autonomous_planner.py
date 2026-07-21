import json
import unittest
from datetime import datetime, timedelta, timezone

from agent.controller import AgentController
from agent.observer import Observer
from agent.planner import Planner
from agent.reflector import Reflector
from services.vlm_service import _extract_json


class _InvalidPlanLLM:
    def __init__(self) -> None:
        self.requests: list[str] = []

    def _prompt(self, key: str) -> str:
        self.asserted_key = key
        return "test"

    def complete_text(self, prompt: str) -> str:
        self.requests.append(prompt)
        return json.dumps(
            {
                "goal": "错误空参数计划",
                "calls": [
                    {"id": "frames", "tool": "getFrames", "arguments": {"trackIds": []}},
                    {
                        "id": "match",
                        "tool": "matchText",
                        "arguments": {"description": "黄色无人艇", "galleryImages": [], "topK": None},
                    },
                ],
                "proposedState": "replan",
                "reason": "错误计划",
                "evidenceGap": "无",
                "answerHint": "",
            },
            ensure_ascii=False,
        )

class _ValidPlanLLM(_InvalidPlanLLM):
    def complete_text(self, prompt: str) -> str:
        self.requests.append(prompt)
        return json.dumps(
            {
                "goal": "有效引用计划",
                "calls": [
                    {"id": "tracks", "tool": "getTrack", "arguments": {"timeRange": None, "offset": 0, "limit": 60}},
                    {"id": "frames", "tool": "getFrames", "arguments": {"trackIds": {"$ref": "tracks.trackIds"}}},
                    {
                        "id": "match",
                        "tool": "matchText",
                        "arguments": {"description": "黄色无人艇", "galleryImages": {"$ref": "frames.keyframes"}, "topK": 3},
                    },
                ],
                "proposedState": "replan",
                "reason": "有效计划",
                "evidenceGap": "灰区核验",
                "answerHint": "",
            },
            ensure_ascii=False,
        )


class _RepairPlanLLM(_InvalidPlanLLM):
    def complete_text(self, prompt: str) -> str:
        self.requests.append(prompt)
        if len(self.requests) == 1:
            return super().complete_text(prompt)
        return _ValidPlanLLM.complete_text(self, prompt)


class _StringReferencePlanLLM(_InvalidPlanLLM):
    def complete_text(self, prompt: str) -> str:
        self.requests.append(prompt)
        return json.dumps(
            {
                "goal": "兼容模型生成的类引用字符串",
                "calls": [
                    {
                        "id": "frames",
                        "tool": "getFrames",
                        "arguments": {"trackIds": ["$ref: 'tracks.trackIds'"]},
                    },
                    {
                        "id": "matchText",
                        "tool": "matchText",
                        "arguments": {
                            "description": "黄色无人艇",
                            "galleryImages": ["$ref: 'frames.keyframes' "],
                            "topK": 10,
                        },
                    },
                ],
                "proposedState": "replan",
                "reason": "读取已有轨迹的关键帧后执行描述匹配",
                "evidenceGap": None,
                "answerHint": "",
            },
            ensure_ascii=False,
        )

class _CompleteJsonPlanLLM:
    def __init__(self) -> None:
        self.json_calls = 0
        self.text_calls = 0

    def _prompt(self, key: str) -> str:
        return "test"

    def complete_json(self, prompt: str) -> dict[str, object]:
        self.json_calls += 1
        return {
            "goal": "结构化计划",
            "calls": [{"id": "tracks", "tool": "getTrack", "arguments": {"limit": 10}}],
            "proposedState": "replan",
            "reason": "先读取轨迹",
        }

    def complete_text(self, prompt: str) -> str:
        self.text_calls += 1
        raise AssertionError("结构化接口存在时不应调用文本接口")

class _ReflectLLM:
    def _prompt(self, key: str) -> str:
        self.key = key
        return "test"

    def complete_text(self, prompt: str) -> str:
        self.prompt = prompt
        return "状态：replan\n依据：当前只有候选匹配，未完成灰区核验\n缺口：目标船片段\n动作：下一轮读取候选轨迹片段"


class _Repository:
    def get_track(self, track_id: str) -> dict[str, object] | None:
        return {"trackId": str(track_id), "finalDescription": "仓库轨迹"}


class _FailingTools:
    def execute(self, name: str, arguments: dict[str, object]) -> dict[str, object]:
        return {"ok": False, "error": "should_not_execute", "tool": name}

class AutonomousPlannerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.intent = {
            "targetScope": "track_memory",
            "targetKind": "description",
            "description": "黄色无人艇",
            "operation": "existence",
            "registryRelation": "any",
            "timeRange": None,
            "retrievalPageSize": 60,
            "maxRounds": 5,
        }

    def test_invalid_empty_args_are_auto_completed_for_description(self) -> None:
        """空 trackIds/gallery 不再直接停死，而是补全最小描述检索链。"""
        llm = _InvalidPlanLLM()
        planner = Planner(llm, AgentController.TOOL_NAMES)

        plan = planner.decide_tools("视频中有没有黄色无人艇？", self.intent)

        self.assertEqual([call["tool"] for call in plan["calls"]], ["getTrack", "getFrames", "matchText"])
        self.assertEqual(plan["calls"][1]["arguments"]["trackIds"], {"$ref": "tracks.trackIds"})
        self.assertEqual(plan["calls"][2]["arguments"]["galleryImages"], {"$ref": "frames.keyframes"})
        self.assertEqual(plan["proposedState"], "replan")

    def test_replanned_calls_are_executed_when_valid(self) -> None:
        """无效计划可被自动补全或二次规划为可执行链。"""
        llm = _RepairPlanLLM()
        planner = Planner(llm, AgentController.TOOL_NAMES)

        plan = planner.decide_tools("视频中有没有黄色无人艇？", self.intent)

        self.assertEqual([call["tool"] for call in plan["calls"]], ["getTrack", "getFrames", "matchText"])
        self.assertEqual(plan["calls"][1]["arguments"]["trackIds"], {"$ref": "tracks.trackIds"})
        self.assertIn(plan["calls"][2]["tool"], {"matchText"})

    def test_valid_dependent_references_are_preserved(self) -> None:
        planner = Planner(_ValidPlanLLM(), AgentController.TOOL_NAMES)

        plan = planner.decide_tools("视频中有没有黄色无人艇？", self.intent)

        self.assertNotIn("planRepair", plan)
        self.assertEqual([call["tool"] for call in plan["calls"]], ["getTrack", "getFrames", "matchText"])
    def test_structured_plan_interface_is_preferred(self) -> None:
        llm = _CompleteJsonPlanLLM()
        planner = Planner(llm, AgentController.TOOL_NAMES)

        plan = planner.decide_tools("视频中有没有船？", self.intent)

        self.assertEqual(llm.json_calls, 1)
        self.assertEqual(llm.text_calls, 0)
        self.assertEqual(plan["calls"][0]["tool"], "getTrack")

    def test_missing_get_track_is_auto_inserted_before_get_frames(self) -> None:
        class _OnlyFramesLLM(_InvalidPlanLLM):
            def complete_text(self, prompt: str) -> str:
                self.requests.append(prompt)
                return json.dumps(
                    {
                        "goal": "缺上游轨迹",
                        "calls": [
                            {"id": "frames", "tool": "getFrames", "arguments": {"trackIds": {"$ref": "tracks.trackIds"}}},
                            {
                                "id": "match",
                                "tool": "matchText",
                                "arguments": {
                                    "description": "黄色无人艇",
                                    "galleryImages": {"$ref": "frames.keyframes"},
                                    "topK": 5,
                                },
                            },
                        ],
                        "proposedState": "replan",
                        "reason": "缺 getTrack",
                        "evidenceGap": None,
                        "answerHint": "",
                    },
                    ensure_ascii=False,
                )

        planner = Planner(_OnlyFramesLLM(), AgentController.TOOL_NAMES)
        plan = planner.decide_tools("视频中有没有黄色无人艇？", self.intent)
        self.assertEqual([call["tool"] for call in plan["calls"]], ["getTrack", "getFrames", "matchText"])
        self.assertEqual(plan["calls"][0]["arguments"]["limit"], 60)

    def test_string_references_are_normalized_before_validation(self) -> None:
        planner = Planner(_StringReferencePlanLLM(), AgentController.TOOL_NAMES)

        plan = planner.decide_tools(
            "视频中有没有黄色无人艇？",
            self.intent,
            memory_scope={"tracks": {"trackIds": ["1", "2"]}},
        )

        self.assertNotIn("planRepair", plan)
        self.assertEqual(plan["calls"][0]["arguments"]["trackIds"], {"$ref": "tracks.trackIds"})
        self.assertEqual(plan["calls"][1]["arguments"]["galleryImages"], {"$ref": "frames.keyframes"})
    def test_reference_field_must_be_declared_by_prior_tool(self) -> None:
        planner = Planner(_ValidPlanLLM(), AgentController.TOOL_NAMES)
        valid, issue = planner._calls_are_executable(
            [
                {"id": "tracks", "tool": "getTrack", "arguments": {}},
                {"id": "frames", "tool": "getFrames", "arguments": {"trackIds": {"$ref": "tracks.unknownField"}}},
            ],
            {},
            "replan",
        )

        self.assertFalse(valid)
        self.assertIn("unknownField", issue or "")

    def test_planner_receives_available_result_structure(self) -> None:
        llm = _ValidPlanLLM()
        planner = Planner(llm, AgentController.TOOL_NAMES)

        planner.decide_tools(
            "视频中有没有黄色无人艇？",
            self.intent,
            memory_scope={
                "tracks": {
                    "ok": True,
                    "trackIds": ["1"],
                    "tracks": [{"trackId": "1"}],
                    "hasMore": False,
                }
            },
        )

        self.assertIn("availableResults", llm.requests[0])
        self.assertIn("hasMore", llm.requests[0])

    def test_observer_skips_matching_when_gallery_is_empty(self) -> None:
        observer = Observer(_ReflectLLM(), _FailingTools())
        result = observer.execute(
            {
                "calls": [
                    {"id": "match", "tool": "matchText", "arguments": {"description": "黄色无人艇", "galleryImages": {"$ref": "frames.keyframes"}}},
                ]
            },
            context={"frames": {"ok": True, "keyframes": []}},
        )

        self.assertTrue(result["observations"][0]["skipped"])
        self.assertIn("dependency_empty", result["observations"][0]["skipReason"])
    def test_observer_stops_dependent_tools_after_failed_reference(self) -> None:
        observer = Observer(_ReflectLLM(), _FailingTools())
        result = observer.execute(
            {
                "calls": [
                    {"id": "frames", "tool": "getFrames", "arguments": {"trackIds": {"$ref": "tracks.trackIds"}}},
                    {"id": "match", "tool": "matchText", "arguments": {"description": "黄色无人艇", "galleryImages": {"$ref": "frames.keyframes"}}},
                ]
            },
            context={"tracks": {"ok": False, "error": "track_query_failed"}},
        )

        self.assertTrue(all(item["skipped"] for item in result["observations"]))
        self.assertEqual(result["summary"]["skippedCount"], 2)
        self.assertIn("dependency_failed", result["observations"][0]["skipReason"])

    def test_reflector_repairs_terminal_state_with_continue_action(self) -> None:
        reflection = Reflector._parse_autonomous_review(
            "状态：sufficient\n依据：当前仍需补充证据\n缺口：关键帧\n动作：下一轮读取关键帧"
        )

        self.assertEqual(reflection["state"], "replan")
    def test_nested_match_reference_can_supply_clip_track_id(self) -> None:
        planner = Planner(_ValidPlanLLM(), AgentController.TOOL_NAMES)
        valid, issue = planner._calls_are_executable(
            [
                {"id": "clip", "tool": "getClip", "arguments": {"trackId": {"$ref": "match.matches.0.matchedTrackId"}}},
            ],
            {"match": {"ok": True, "matches": [{"matchedTrackId": "11"}]}},
            "replan",
        )

        self.assertTrue(valid, issue)
    def test_verify_target_requires_a_supported_evidence_pair(self) -> None:
        planner = Planner(_ValidPlanLLM(), AgentController.TOOL_NAMES)
        valid, issue = planner._calls_are_executable(
            [{"id": "verify", "tool": "verifyTarget", "arguments": {"description": "黄色无人艇"}}],
            {},
            "replan",
        )

        self.assertFalse(valid)
        self.assertIn("verifyTarget", issue or "")

    def test_match_image_requires_one_registry_and_one_keyframe_input(self) -> None:
        planner = Planner(_ValidPlanLLM(), AgentController.TOOL_NAMES)
        valid, issue = planner._calls_are_executable(
            [
                {"id": "frames", "tool": "getFrames", "arguments": {"trackIds": ["1"]}},
                {
                    "id": "invalidMatch",
                    "tool": "matchImage",
                    "arguments": {
                        "queryImages": {"$ref": "frames.keyframes"},
                        "galleryImages": {"$ref": "frames.keyframes"},
                    },
                },
            ],
            {},
            "replan",
        )

        self.assertFalse(valid)
        self.assertIn("matchImage", issue or "")

    def test_match_image_accepts_registry_references_and_keyframes(self) -> None:
        planner = Planner(_ValidPlanLLM(), AgentController.TOOL_NAMES)
        valid, issue = planner._calls_are_executable(
            [
                {"id": "registry", "tool": "getRegistry", "arguments": {"hullNumber": "小黑03"}},
                {"id": "frames", "tool": "getFrames", "arguments": {"trackIds": ["1"]}},
                {
                    "id": "match",
                    "tool": "matchImage",
                    "arguments": {
                        "queryImages": {"$ref": "registry.registryReferences"},
                        "galleryImages": {"$ref": "frames.keyframes"},
                    },
                },
            ],
            {},
            "replan",
        )

        self.assertTrue(valid, issue)

    def test_autonomous_reflector_parses_and_uses_its_own_state(self) -> None:
        reflector = Reflector(_ReflectLLM())

        reflection = reflector.review(
            "sufficient",
            "PlanAgent 建议停止",
            {"calls": [{"tool": "matchText", "matchCount": 1}]},
            autonomous=True,
            success_criteria="完成灰区核验",
        )

        self.assertEqual(reflection["state"], "replan")
        self.assertEqual(reflection["evidenceGap"], "目标船片段")

    def test_autonomous_reflector_repairs_uncertain_with_next_round_action(self) -> None:
        reflection = Reflector._parse_autonomous_review(
            "状态：uncertain\n依据：未完成图像匹配\n缺口：全轨迹关键帧\n动作：下一轮读取全部轨迹后调用 matchImage"
        )

        self.assertEqual(reflection["state"], "replan")
        self.assertIn("更正", reflection["stateCorrection"])
        self.assertEqual(reflection["nextAction"], "下一轮读取全部轨迹后调用 matchImage")

    def test_autonomous_history_keeps_reflector_next_action(self) -> None:
        controller = object.__new__(AgentController)
        controller.rounds = [{
            "roundId": "round-1",
            "plan": {"goal": "先查直接轨迹", "calls": []},
            "observed": {"calls": []},
            "reflection": {
                "state": "replan",
                "reason": "需要全轨迹匹配",
                "evidenceGap": "关键帧",
                "nextAction": "读取全部轨迹并调用 matchImage",
            },
        }]

        history = controller._autonomous_history()

        self.assertEqual(history[0]["nextAction"], "读取全部轨迹并调用 matchImage")

    def test_match_results_only_materialize_matched_tracks(self) -> None:
        controller = object.__new__(AgentController)
        controller.repository = _Repository()
        tracks = [
            {"trackId": "track-1", "finalDescription": "灰色货船"},
            {"trackId": "track-2", "finalDescription": "黄色无人艇"},
        ]
        matches = [{"matchedTrackId": "track-2", "embeddingScore": 0.91, "scoreBand": "match"}]

        result = controller._tracks_for_matches(matches, tracks)

        self.assertEqual([item["trackId"] for item in result], ["track-2"])
        self.assertEqual(result[0]["embeddingScore"], 0.91)


    def test_execution_blueprint_is_complete_and_stable(self) -> None:
        blueprint = Planner.build_execution_blueprint({
            "targetScope": "track_memory",
            "targetKind": "all",
            "operation": "list",
            "registryRelation": "out",
        })

        self.assertEqual(
            [step["stepId"] for step in blueprint],
            [
                "plan-track-scope",
                "plan-hull-exact",
                "plan-registry-catalog",
                "plan-keyframes",
                "plan-registry-match",
                "plan-gray-verify",
            ],
        )
        self.assertEqual(len({step["stepId"] for step in blueprint}), len(blueprint))
        self.assertTrue(blueprint[-1]["optional"])

    def test_tool_event_maps_to_existing_blueprint_step(self) -> None:
        controller = object.__new__(AgentController)
        controller.plan_blueprint = Planner.build_execution_blueprint({
            "targetScope": "track_memory",
            "targetKind": "description",
            "operation": "existence",
            "registryRelation": "any",
        })
        events: list[dict[str, object]] = []
        controller.event_handler = events.append

        controller._emit_observer_tool_event(2, {"phase": "running", "id": "frames", "tool": "getFrames"})

        self.assertEqual(events[0]["planStepId"], "plan-keyframes")
        self.assertEqual(events[0]["round"], 2)

class JsonExtractionTest(unittest.TestCase):
    def test_extracts_json_from_explanation_and_code_fence(self) -> None:
        value = _extract_json('说明文字\n```json\n{"goal":"读取轨迹","calls":[]}\n```')

        self.assertEqual(value["goal"], "读取轨迹")

    def test_prefers_plan_object_when_multiple_objects_exist(self) -> None:
        value = _extract_json('{"note":"提示"}\n{"goal":"读取轨迹","calls":[{"tool":"getTrack"}]}')

        self.assertEqual(value["goal"], "读取轨迹")

    def test_repairs_trailing_comma(self) -> None:
        value = _extract_json('{"goal":"读取轨迹","calls":[],}')

        self.assertEqual(value["calls"], [])
class PlannerTimeRangeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.timezone = timezone(timedelta(hours=8))
        self.now = datetime(2026, 7, 20, 10, 58, 23, tzinfo=self.timezone)

    def assert_range(self, question: str, expected_start: datetime, expected_end: datetime) -> None:
        result = Planner._time_range(question, now=self.now)

        self.assertIsNotNone(result)
        self.assertEqual(datetime.fromtimestamp(result[0], self.timezone), expected_start)
        self.assertEqual(datetime.fromtimestamp(result[1], self.timezone), expected_end)

    def test_yesterday_afternoon_with_parentheses(self) -> None:
        self.assert_range(
            "查找一下昨（天下午）4点-5点有哪些在库船出现？",
            datetime(2026, 7, 19, 16, 0, tzinfo=self.timezone),
            datetime(2026, 7, 19, 17, 0, tzinfo=self.timezone),
        )

    def test_yesterday_afternoon_clock_format(self) -> None:
        self.assert_range(
            "昨天下午4:00-5:00有哪些船？",
            datetime(2026, 7, 19, 16, 0, tzinfo=self.timezone),
            datetime(2026, 7, 19, 17, 0, tzinfo=self.timezone),
        )

    def test_evening_range_can_cross_midnight(self) -> None:
        self.assert_range(
            "今天晚上11点到1点有哪些船？",
            datetime(2026, 7, 20, 23, 0, tzinfo=self.timezone),
            datetime(2026, 7, 21, 1, 0, tzinfo=self.timezone),
        )


if __name__ == "__main__":
    unittest.main()
