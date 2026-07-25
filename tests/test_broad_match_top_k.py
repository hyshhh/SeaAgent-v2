from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from agent.graph import (
    _apply_retrieval_limits,
    _default_plan_calls,
    run_sea_agent,
)
from tools.service import ToolService


class _FakeVectorIndex:
    def __init__(self, vectors):
        self.vectors = vectors

    def get_many(self, ids):
        return {int(item): self.vectors[int(item)] for item in ids if int(item) in self.vectors}


class BroadMatchTopKTest(unittest.TestCase):
    def test_default_broad_plan_does_not_truncate_by_normal_top_k(self):
        calls = _default_plan_calls(
            {
                "operation": "list",
                "targetScope": "both",
                "registryRelation": "in",
                "questionType": "registry_in_list",
            },
            top_k=2,
            broad_match_top_k=0,
        )

        self.assertEqual([call["tool"] for call in calls], ["listRegistry", "getTrack", "getFrames", "matchImage"])
        self.assertEqual(calls[1]["arguments"]["limit"], 0)
        self.assertEqual(calls[3]["arguments"]["topK"], 0)

    def test_broad_plan_uses_its_own_positive_limit(self):
        calls = _default_plan_calls(
            {
                "operation": "list",
                "targetScope": "both",
                "registryRelation": "in",
                "questionType": "registry_in_list",
            },
            top_k=2,
            broad_match_top_k=7,
        )

        self.assertEqual(calls[3]["arguments"]["topK"], 7)
        self.assertEqual(calls[1]["arguments"]["limit"], 0)

    def test_normal_description_matching_keeps_normal_top_k(self):
        calls = _default_plan_calls(
            {
                "operation": "list",
                "targetScope": "track_memory",
                "description": "灰色船体",
            },
            top_k=2,
            broad_match_top_k=7,
        )

        self.assertEqual([call["tool"] for call in calls], ["getTrack", "getFrames", "matchText"])
        self.assertEqual(calls[2]["arguments"]["topK"], 2)

    def test_model_broad_plan_is_forced_to_full_track_and_independent_top_k(self):
        calls = [
            {"id": "registry", "tool": "listRegistry", "arguments": {}},
            {"id": "tracks", "tool": "getTrack", "arguments": {"offset": 0, "limit": 12}},
            {"id": "frames", "tool": "getFrames", "arguments": {"trackIds": {"$ref": "tracks.trackIds"}}},
            {
                "id": "match",
                "tool": "matchImage",
                "arguments": {
                    "queryImages": {"$ref": "registry.registryReferences"},
                    "galleryImages": {"$ref": "frames.keyframes"},
                    "topK": 2,
                },
            },
        ]

        normalized = _apply_retrieval_limits(calls, broad_match_top_k=9)

        self.assertEqual(normalized[1]["arguments"]["limit"], 0)
        self.assertEqual(normalized[3]["arguments"]["topK"], 9)
        self.assertEqual(calls[1]["arguments"]["limit"], 12)
        self.assertEqual(calls[3]["arguments"]["topK"], 2)

    def test_single_registry_match_keeps_normal_limit(self):
        calls = [
            {"id": "registry", "tool": "getRegistry", "arguments": {"hullNumber": "0857"}},
            {"id": "tracks", "tool": "getTrack", "arguments": {"hullNumber": "0857", "limit": 60}},
            {"id": "frames", "tool": "getFrames", "arguments": {"trackIds": {"$ref": "tracks.trackIds"}}},
            {
                "id": "match",
                "tool": "matchImage",
                "arguments": {
                    "queryImages": {"$ref": "registry.registryReferences"},
                    "galleryImages": {"$ref": "frames.keyframes"},
                    "topK": 2,
                },
            },
        ]

        normalized = _apply_retrieval_limits(calls, broad_match_top_k=9)

        self.assertEqual(normalized, calls)

    def test_match_image_zero_returns_all_and_positive_truncates(self):
        service = ToolService.__new__(ToolService)
        service.settings = {
            "image_match": 0.72,
            "image_exclude": 0.52,
        }
        service.vectors = SimpleNamespace(
            keyframes=_FakeVectorIndex({
                1: np.array([1.0, 0.0], dtype=np.float32),
                2: np.array([0.9, 0.1], dtype=np.float32),
                3: np.array([0.8, 0.2], dtype=np.float32),
            }),
            registry=_FakeVectorIndex({
                101: np.array([1.0, 0.0], dtype=np.float32),
            }),
        )
        query_images = [{
            "referenceId": "ref-1",
            "registryId": "registry-1",
            "registryVectorId": 101,
        }]
        gallery_images = [
            {"trackId": "track-1", "keyframeId": "frame-1", "keyframeVectorId": 1},
            {"trackId": "track-2", "keyframeId": "frame-2", "keyframeVectorId": 2},
            {"trackId": "track-3", "keyframeId": "frame-3", "keyframeVectorId": 3},
        ]

        unlimited = service.matchImage(query_images, gallery_images, topK=0)
        limited = service.matchImage(query_images, gallery_images, topK=2)

        self.assertEqual(len(unlimited["matches"]), 3)
        self.assertEqual(len(limited["matches"]), 2)
        self.assertEqual([item["matchedTrackId"] for item in limited["matches"]], ["track-1", "track-2"])

    def test_run_sea_agent_passes_broad_top_k_separately(self):
        class _FakeApp:
            def invoke(self, initial, config):
                return initial

        with patch("agent.graph.build_sea_agent_graph", return_value=_FakeApp()) as builder:
            state = run_sea_agent(
                "测试广泛匹配",
                object(),
                object(),
                query_top_k=4,
                broad_match_top_k=0,
            )

        self.assertEqual(builder.call_args.kwargs["query_top_k"], 4)
        self.assertEqual(builder.call_args.kwargs["broad_match_top_k"], 0)
        self.assertEqual(state["query_top_k"], 4)
        self.assertEqual(state["broad_match_top_k"], 0)


if __name__ == "__main__":
    unittest.main()
