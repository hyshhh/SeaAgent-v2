import unittest

import numpy as np

from agent.controller import AgentController
from tools.service import ToolService


class _RegistryRepository:
    def references_by_ids(self, reference_ids):
        records = {
            "ref-003-a": {"referenceId": "ref-003-a", "registryId": "registry-003-a"},
            "ref-003-b": {"referenceId": "ref-003-b", "registryId": "registry-003-b"},
            "ref-004": {"referenceId": "ref-004", "registryId": "registry-004"},
        }
        return [records[value] for value in reference_ids if value in records]

    def list_registry(self):
        return [
            {"registryId": "registry-003-a", "hullNumber": "003"},
            {"registryId": "registry-003-b", "hullNumber": "003"},
            {"registryId": "registry-004", "hullNumber": "004"},
        ]


class _DisplayTools:
    def getFrames(self, track_ids):
        return {
            "keyframesByTrack": {
                str(track_id): {
                    "keyframes": [{"keyframeId": f"frame-{track_id}", "retentionScore": 1.0}]
                }
                for track_id in track_ids
            }
        }

    def _representative_registry_reference_ids(self, reference_ids):
        return list(dict.fromkeys(reference_ids))


class EvidenceDisplayTest(unittest.TestCase):
    def test_crop_is_scaled_to_fixed_canvas(self):
        crop = np.full((40, 80, 3), 255, dtype=np.uint8)
        canvas = ToolService._fit_crop_to_canvas(crop, (640, 360))
        self.assertEqual(canvas.shape, (360, 640, 3))
        self.assertGreater(np.count_nonzero(canvas == 255), crop.size)

    def test_image_input_flattens_registry_and_keyframe_groups(self):
        records = ToolService._flatten_image_records([
            {"references": [{"referenceId": "ref-1"}]},
            {"keyframesByTrack": {"1": {"keyframes": [{"keyframeId": "frame-1"}]}}},
        ])

        self.assertEqual(records, [{"referenceId": "ref-1"}, {"keyframeId": "frame-1"}])

    def test_registry_evidence_keeps_one_image_per_hull(self):
        service = ToolService.__new__(ToolService)
        service.repository = _RegistryRepository()
        selected = service._representative_registry_reference_ids(["ref-003-a", "ref-003-b", "ref-004"])
        self.assertEqual(selected, ["ref-003-a", "ref-004"])

    def test_all_tracks_receive_lazy_evidence_groups(self):
        controller = AgentController.__new__(AgentController)
        controller.tools = _DisplayTools()
        controller.meta = {}
        controller.display_record = None
        controller.display_groups = []
        tracks = [{"trackId": str(index)} for index in range(1, 99)]
        controller._display_tracks(tracks, include_clips=True, include_registry=False)
        self.assertEqual(len(controller.display_groups), len(tracks))
        self.assertEqual([item["clipTrackId"] for item in controller.display_groups], [str(index) for index in range(1, 99)])
        self.assertTrue(all(len(item["keyframeIds"]) == 1 for item in controller.display_groups))

    def test_each_track_keeps_its_own_registry_reference(self):
        controller = AgentController.__new__(AgentController)
        controller.tools = _DisplayTools()
        controller.meta = {}
        controller.display_record = None
        controller.display_groups = []
        tracks = [
            {"trackId": "1", "registryReferenceIds": ["ref-003-a"]},
            {"trackId": "2", "registryReferenceIds": ["ref-003-a"]},
        ]

        controller._display_tracks(tracks, include_clips=False, include_registry=True)

        self.assertEqual(
            [item["registryReferenceIds"] for item in controller.display_groups],
            [["ref-003-a"], ["ref-003-a"]],
        )


    def test_autonomous_count_only_displays_deduplicated_representatives(self):
        controller = AgentController.__new__(AgentController)
        controller.tools = _DisplayTools()
        controller.meta = {"registryRelation": "any", "targetScope": "track_memory"}
        controller.session_id = "session-test"
        controller.question = "一共出现多少艘船？"
        controller.rounds = []
        controller.tool_chain = []
        controller.tool_records = []
        controller.display_record = None
        controller.display_groups = []
        controller.display_limit = 3
        controller.event_handler = None
        controller.working_scope = {
            "tracksSnapshot": {
                "tracks": [
                    {"trackId": "1", "startTime": 0.0, "endTime": 1.0},
                    {"trackId": "2", "startTime": 2.0, "endTime": 3.0},
                    {"trackId": "3", "startTime": 4.0, "endTime": 5.0},
                ]
            },
            "countFrames": {
                "keyframesByTrack": {
                    "1": {"keyframes": [{"keyframeId": "frame-1", "retentionScore": 0.4}]},
                    "2": {"keyframes": [{"keyframeId": "frame-2", "retentionScore": 0.9}]},
                    "3": {"keyframes": [{"keyframeId": "frame-3", "retentionScore": 0.8}]},
                }
            },
            "deduplicatedCount": {
                "highThresholdShipCount": 2,
                "lowThresholdShipCount": 2,
                "highGroups": [["1", "2"], ["3"]],
            },
        }

        result = controller._finish_autonomous(None, "sufficient", "去重完成")

        self.assertEqual([item["trackId"] for item in result["tracks"]], ["2", "3"])
        self.assertEqual([item["trackId"] for item in result["displayGroups"]], ["2", "3"])
        self.assertEqual(result["display"]["trackCount"], 2)
        self.assertEqual(result["evidence"]["keyframeIds"], ["frame-2", "frame-3"])


if __name__ == "__main__":
    unittest.main()
