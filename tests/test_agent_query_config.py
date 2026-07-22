import unittest

from pydantic import ValidationError

from agent.controller import AgentController
from web.models import AgentQuery


class _EvidenceRepository:
    def __init__(self) -> None:
        self.records = []

    def add_evidence(self, evidence_id, record_id, result, source) -> None:
        self.records.append((evidence_id, record_id, result, source))


class AgentQueryConfigTest(unittest.TestCase):
    def test_query_accepts_optional_top_k(self) -> None:
        self.assertIsNone(AgentQuery(question="测试").top_k)
        self.assertEqual(AgentQuery(question="测试", top_k=8).top_k, 8)

    def test_query_rejects_out_of_range_top_k(self) -> None:
        with self.assertRaises(ValidationError):
            AgentQuery(question="测试", top_k=0)
        with self.assertRaises(ValidationError):
            AgentQuery(question="测试", top_k=21)

    def test_query_top_k_only_overrides_match_tools(self) -> None:
        controller = object.__new__(AgentController)
        controller.query_top_k = 7
        plan = {
            "calls": [
                {"id": "tracks", "tool": "getTrack", "arguments": {"limit": 60}},
                {"id": "text", "tool": "matchText", "arguments": {"topK": 3}},
                {"id": "image", "tool": "matchImage", "arguments": {}},
            ]
        }

        controller._apply_query_top_k(plan)

        self.assertEqual(plan["calls"][0]["arguments"], {"limit": 60})
        self.assertEqual(plan["calls"][1]["arguments"]["topK"], 7)
        self.assertEqual(plan["calls"][2]["arguments"]["topK"], 7)

    def test_tool_record_keeps_round_call_and_summary(self) -> None:
        controller = object.__new__(AgentController)
        controller.repository = _EvidenceRepository()
        controller.tool_chain = []
        controller.tool_records = []
        observed = {
            "observations": [
                {
                    "id": "tracks",
                    "tool": "getTrack",
                    "result": {"ok": True, "tracks": [{"trackId": "1"}, {"trackId": "2"}]},
                }
            ]
        }

        controller._store_observations("round-id", observed, 2)

        self.assertEqual(controller.tool_chain, ["getTrack(tracks)"])
        self.assertEqual(controller.tool_records[0]["round"], 2)
        self.assertEqual(controller.tool_records[0]["tool"], "getTrack")
        self.assertEqual(controller.tool_records[0]["id"], "tracks")
        self.assertEqual(controller.tool_records[0]["trackCount"], 2)


if __name__ == "__main__":
    unittest.main()
