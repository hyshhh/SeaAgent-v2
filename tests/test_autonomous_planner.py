import json
import unittest

from agent.controller import AgentController
from agent.planner import Planner


class _InvalidPlanLLM:
    def _prompt(self, key: str) -> str:
        self.asserted_key = key
        return "test"

    def complete_text(self, prompt: str) -> str:
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

    def test_empty_dependent_arguments_use_controlled_fallback(self) -> None:
        planner = Planner(_InvalidPlanLLM(), AgentController.TOOL_NAMES)

        plan = planner.decide_tools("视频中有没有黄色无人艇？", self.intent)

        self.assertIn("planRepair", plan)
        self.assertEqual([call["tool"] for call in plan["calls"]], ["getTrack", "getFrames", "matchText"])
        self.assertEqual(plan["calls"][1]["arguments"]["trackIds"], {"$ref": "tracks.trackIds"})
        self.assertEqual(plan["calls"][2]["arguments"]["galleryImages"], {"$ref": "frames.keyframes"})

    def test_valid_dependent_references_are_preserved(self) -> None:
        planner = Planner(_ValidPlanLLM(), AgentController.TOOL_NAMES)

        plan = planner.decide_tools("视频中有没有黄色无人艇？", self.intent)

        self.assertNotIn("planRepair", plan)
        self.assertEqual([call["tool"] for call in plan["calls"]], ["getTrack", "getFrames", "matchText"])


if __name__ == "__main__":
    unittest.main()
