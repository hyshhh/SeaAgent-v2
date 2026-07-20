import json
import unittest
from datetime import datetime, timedelta, timezone

from agent.controller import AgentController
from agent.planner import Planner
from agent.reflector import Reflector


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

    def test_invalid_plan_stops_without_hardcoded_tool_chain(self) -> None:
        llm = _InvalidPlanLLM()
        planner = Planner(llm, AgentController.TOOL_NAMES)

        plan = planner.decide_tools("视频中有没有黄色无人艇？", self.intent)

        self.assertIn("planRepair", plan)
        self.assertEqual(plan["calls"], [])
        self.assertEqual(plan["proposedState"], "uncertain")
        self.assertEqual(len(llm.requests), 2)
        self.assertIn("planValidationError", llm.requests[1])

    def test_replanned_calls_are_executed_when_valid(self) -> None:
        llm = _RepairPlanLLM()
        planner = Planner(llm, AgentController.TOOL_NAMES)

        plan = planner.decide_tools("视频中有没有黄色无人艇？", self.intent)

        self.assertIn("planRepair", plan)
        self.assertEqual([call["tool"] for call in plan["calls"]], ["getTrack", "getFrames", "matchText"])
        self.assertEqual(plan["calls"][1]["arguments"]["trackIds"], {"$ref": "tracks.trackIds"})

    def test_valid_dependent_references_are_preserved(self) -> None:
        planner = Planner(_ValidPlanLLM(), AgentController.TOOL_NAMES)

        plan = planner.decide_tools("视频中有没有黄色无人艇？", self.intent)

        self.assertNotIn("planRepair", plan)
        self.assertEqual([call["tool"] for call in plan["calls"]], ["getTrack", "getFrames", "matchText"])

    def test_verify_target_requires_a_supported_evidence_pair(self) -> None:
        planner = Planner(_ValidPlanLLM(), AgentController.TOOL_NAMES)
        valid, issue = planner._calls_are_executable(
            [{"id": "verify", "tool": "verifyTarget", "arguments": {"description": "黄色无人艇"}}],
            {},
            "replan",
        )

        self.assertFalse(valid)
        self.assertIn("verifyTarget", issue or "")

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
