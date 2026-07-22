import unittest

from agent.acceptance import build_acceptance_progress, compact_acceptance
from agent.controller import AgentController
from agent.reflector import Reflector


class _ReflectLLM:
    def _prompt(self, key: str) -> str:
        return "test"

    def complete_text(self, prompt: str) -> str:
        return (
            "状态：sufficient\n"
            "依据：轨迹已经读取完整\n"
            "缺口：无\n"
            "动作：停止并返回结果"
        )


class AcceptanceProgressTest(unittest.TestCase):
    def setUp(self) -> None:
        self.intent = {
            "targetScope": "track_memory",
            "targetKind": "all",
            "operation": "list",
            "registryRelation": "out",
            "expectedOutcome": "列出视频中未在库船舶",
            "successCriteria": "已完成在库/未在库判定并得到轨迹列表",
            "timeRange": None,
        }
        self.track_one = {
            "trackId": "1",
            "finalHullNumber": "001",
            "finalMatchType": "confirmed",
        }
        self.track_two = {
            "trackId": "2",
            "finalHullNumber": None,
            "finalMatchType": "unknown",
        }
        self.registry = {
            "ok": True,
            "registryItems": [{"registryId": "r1", "hullNumber": "001"}],
            "registryReferences": [
                {
                    "referenceId": "ref1",
                    "registryId": "r1",
                    "registryVectorId": 10,
                    "isEmbedded": True,
                }
            ],
            "registryReferenceIds": ["ref1"],
            "unsearchableRegistryIds": [],
        }

    def _complete_tracks_scope(self) -> dict:
        return {
            "tracks1": {
                "ok": True,
                "trackIds": ["1"],
                "tracks": [self.track_one],
                "totalTrackCount": 2,
                "returnedTrackCount": 1,
                "offset": 0,
                "limit": 1,
                "hasMore": True,
                "nextOffset": 1,
            },
            "tracks2": {
                "ok": True,
                "trackIds": ["2"],
                "tracks": [self.track_two],
                "totalTrackCount": 2,
                "returnedTrackCount": 1,
                "offset": 1,
                "limit": 1,
                "hasMore": False,
                "nextOffset": None,
            },
            "registryCatalog": self.registry,
        }

    def test_complete_track_list_does_not_satisfy_registry_relation(self) -> None:
        progress = build_acceptance_progress(self.intent, self._complete_tracks_scope())

        self.assertTrue(progress["trackScopeComplete"])
        self.assertFalse(progress["acceptanceSatisfied"])
        self.assertEqual(progress["pendingRequirements"], ["exact_hull_classification"])
        self.assertIn("matchHull", progress["nextAction"])

    def test_reflector_overrides_early_sufficient_state(self) -> None:
        progress = build_acceptance_progress(self.intent, self._complete_tracks_scope())
        reflection = Reflector(_ReflectLLM()).review(
            "sufficient",
            "轨迹读取完整",
            {"task": {"successCriteria": self.intent["successCriteria"]}, "calls": []},
            autonomous=True,
            expected_outcome=self.intent["expectedOutcome"],
            success_criteria=self.intent["successCriteria"],
            next_agent_focus="先完成关系分类",
            previous_rounds=[],
            acceptance_context=compact_acceptance(progress),
        )

        self.assertEqual(reflection["state"], "replan")
        self.assertEqual(reflection["evidenceGap"], "稳定舷号精确查库")
        self.assertIn("matchHull", reflection["nextAction"])
        self.assertTrue(reflection["acceptanceOverride"])

    def test_full_relation_chain_reaches_acceptance(self) -> None:
        scope = self._complete_tracks_scope()
        scope["exactHull"] = {
            "ok": True,
            "matchedHullNumbers": ["001"],
            "unmatchedHullNumbers": [],
            "exactMatches": {"001": [{"registryId": "r1"}]},
        }
        progress = build_acceptance_progress(self.intent, scope)
        self.assertEqual(progress["pendingRequirements"], ["keyframe_evidence"])
        self.assertEqual(progress["remainingTrackIds"], ["2"])

        scope["remainingFrames"] = {
            "ok": True,
            "keyframeIds": ["k2"],
            "keyframes": [
                {
                    "keyframeId": "k2",
                    "trackId": "2",
                    "keyframeVectorId": 20,
                    "isEmbedded": True,
                }
            ],
            "keyframesByTrack": {
                "2": {
                    "keyframeIds": ["k2"],
                    "keyframes": [{"keyframeId": "k2", "trackId": "2"}],
                }
            },
            "unsearchableTrackIds": [],
        }
        progress = build_acceptance_progress(self.intent, scope)
        self.assertEqual(progress["pendingRequirements"], ["registry_image_classification"])

        scope["registryImageMatch"] = {
            "ok": True,
            "matchMode": "image_to_image",
            "matches": [
                {
                    "matchedTrackId": "2",
                    "matchedRegistryId": "r1",
                    "embeddingScore": 0.1,
                    "scoreBand": "mismatch",
                    "matchedKeyframeIds": ["k2"],
                    "matchedRegistryReferenceIds": ["ref1"],
                }
            ],
        }
        progress = build_acceptance_progress(self.intent, scope)

        self.assertTrue(progress["acceptanceSatisfied"])
        self.assertEqual(progress["inRegistryTrackIds"], ["1"])
        self.assertEqual(progress["outOfRegistryTrackIds"], ["2"])
        self.assertEqual(progress["unresolvedTrackIds"], [])

    def test_autonomous_finish_returns_only_out_of_registry_tracks(self) -> None:
        scope = self._complete_tracks_scope()
        scope["exactHull"] = {
            "ok": True,
            "matchedHullNumbers": ["001"],
            "unmatchedHullNumbers": [],
            "exactMatches": {"001": [{"registryId": "r1"}]},
        }
        scope["remainingFrames"] = {
            "ok": True,
            "keyframeIds": ["k2"],
            "keyframes": [{"keyframeId": "k2", "trackId": "2"}],
            "keyframesByTrack": {"2": {"keyframes": [{"keyframeId": "k2", "trackId": "2"}]}},
            "unsearchableTrackIds": [],
        }
        scope["registryImageMatch"] = {
            "ok": True,
            "matchMode": "image_to_image",
            "matches": [
                {
                    "matchedTrackId": "2",
                    "matchedRegistryId": "r1",
                    "embeddingScore": 0.1,
                    "scoreBand": "mismatch",
                    "matchedKeyframeIds": ["k2"],
                    "matchedRegistryReferenceIds": ["ref1"],
                }
            ],
        }
        controller = AgentController.__new__(AgentController)
        controller.meta = self.intent
        controller.working_scope = scope
        controller.display_limit = 3
        controller._finish = lambda conclusion, tracks, reason, state, extra=None, display=None: {
            "conclusion": conclusion,
            "tracks": tracks,
            "reason": reason,
            "state": state,
            "extra": extra or {},
        }

        result = controller._finish_autonomous(None, "sufficient", "done")

        self.assertEqual({str(item["trackId"]) for item in result["tracks"]}, {"2"})
        self.assertEqual(result["state"], "sufficient")
        self.assertEqual(result["extra"]["inRegistryTrackIds"], ["1"])
        self.assertEqual(result["extra"]["outOfRegistryTrackIds"], ["2"])

    def test_controller_repairs_plan_to_acceptance_next_step(self) -> None:
        controller = AgentController.__new__(AgentController)
        controller.meta = self.intent
        controller.working_scope = self._complete_tracks_scope()
        controller.retrieval_page_size = 60
        progress = build_acceptance_progress(self.intent, controller.working_scope)
        controller.working_scope["acceptance"] = progress

        repaired = controller._align_plan_with_acceptance(
            {
                "goal": "轨迹已经读完，准备停止",
                "calls": [],
                "proposedState": "sufficient",
                "reason": "轨迹列表完整",
            },
            progress,
        )

        self.assertEqual(repaired["proposedState"], "replan")
        self.assertEqual(repaired["calls"][0]["tool"], "matchHull")
        self.assertEqual(
            repaired["calls"][0]["arguments"]["hullNumberArray"],
            {"$ref": "acceptance.confirmedHullNumbers"},
        )


    def test_hull_query_requires_image_match_when_registry_exists_without_direct_hit(self) -> None:
        intent = {
            "targetScope": "track_memory",
            "targetKind": "hull",
            "operation": "existence",
            "registryRelation": "any",
            "hullNumber": "小白07",
        }
        scope = {
            "directHullTracks": {
                "ok": True,
                "queryHullNumber": "小白07",
                "trackIds": [],
                "tracks": [],
                "totalTrackCount": 0,
                "hasMore": False,
            },
            "targetHullRegistry": {
                "ok": True,
                "found": True,
                "searchable": True,
                "hullNumber": "小白07",
                "registryItems": [{"registryId": "r7", "hullNumber": "小白07"}],
                "registryReferences": [{"referenceId": "ref7", "registryId": "r7"}],
            },
        }

        progress = build_acceptance_progress(intent, scope)
        self.assertFalse(progress["acceptanceSatisfied"])
        self.assertEqual(progress["pendingRequirements"], ["hull_track_scope"])

        scope["tracksPage0"] = {
            "ok": True,
            "queryHullNumber": None,
            "queryFinalMatchType": None,
            "trackIds": ["1", "2"],
            "tracks": [{"trackId": "1"}, {"trackId": "2"}],
            "totalTrackCount": 2,
            "hasMore": False,
        }
        progress = build_acceptance_progress(intent, scope)
        self.assertEqual(progress["pendingRequirements"], ["hull_keyframe_evidence"])

        scope["hullSearchFrames"] = {
            "ok": True,
            "keyframes": [{"keyframeId": "k1", "trackId": "1"}, {"keyframeId": "k2", "trackId": "2"}],
            "keyframesByTrack": {
                "1": {"keyframes": [{"keyframeId": "k1", "trackId": "1"}]},
                "2": {"keyframes": [{"keyframeId": "k2", "trackId": "2"}]},
            },
            "unsearchableTrackIds": [],
        }
        progress = build_acceptance_progress(intent, scope)
        self.assertEqual(progress["pendingRequirements"], ["hull_image_classification"])

        scope["hullImageMatch"] = {
            "ok": True,
            "matchMode": "image_to_image",
            "matches": [{"matchedTrackId": "2", "matchedRegistryId": "r7", "scoreBand": "match"}],
        }
        progress = build_acceptance_progress(intent, scope)
        self.assertTrue(progress["acceptanceSatisfied"])
        self.assertEqual(progress["pendingRequirements"], [])

    def test_hull_query_accepts_confirmed_direct_track_without_image_match(self) -> None:
        intent = {
            "targetScope": "track_memory",
            "targetKind": "hull",
            "operation": "existence",
            "registryRelation": "any",
            "hullNumber": "320",
        }
        scope = {
            "directHullTracks": {
                "ok": True,
                "queryHullNumber": "320",
                "trackIds": ["3"],
                "tracks": [{"trackId": "3", "finalHullNumber": "320", "finalMatchType": "confirmed"}],
                "totalTrackCount": 1,
                "hasMore": False,
            },
            "targetHullRegistry": {
                "ok": True,
                "found": False,
                "searchable": False,
                "hullNumber": "320",
                "registryItems": [],
                "registryReferences": [],
            },
        }

        progress = build_acceptance_progress(intent, scope)
        self.assertTrue(progress["acceptanceSatisfied"])
        self.assertEqual(progress["directConfirmedTrackIds"], ["3"])

    def test_hull_track_scope_rebinds_filtered_get_track_to_full_scope(self) -> None:
        controller = AgentController.__new__(AgentController)
        controller.meta = {"hullNumber": "无极", "timeRange": None}
        controller.working_scope = {}
        controller.retrieval_page_size = 60
        acceptance = {
            "acceptanceSatisfied": False,
            "pendingRequirements": ["hull_track_scope"],
            "pendingRequirementLabels": ["舷号回退检索的完整轨迹范围"],
            "expectedTrackCount": None,
            "trackCount": 0,
            "nextAction": "读取完整轨迹范围",
        }
        plan = {
            "calls": [{
                "id": "wrongFilteredTracks",
                "tool": "getTrack",
                "arguments": {"hullNumber": "无极", "offset": 0, "limit": 60},
            }],
            "proposedState": "replan",
        }

        repaired = controller._align_plan_with_acceptance(plan, acceptance)

        self.assertEqual(repaired["calls"][0]["id"], "tracksSnapshot")
        self.assertEqual(repaired["calls"][0]["arguments"]["limit"], 0)
        self.assertNotIn("hullNumber", repaired["calls"][0]["arguments"])

    def test_hull_keyframe_step_binds_all_tracks_and_appends_match_image(self) -> None:
        controller = AgentController.__new__(AgentController)
        controller.meta = {"hullNumber": "无极", "timeRange": None}
        controller.retrieval_page_size = 60
        controller.working_scope = {
            "targetHullRegistry": {
                "hullNumber": "无极",
                "registryReferences": [{"referenceId": "ref-1"}],
            }
        }
        acceptance = {
            "acceptanceSatisfied": False,
            "pendingRequirements": ["hull_keyframe_evidence"],
            "pendingRequirementLabels": ["舷号回退检索的正式关键帧"],
            "hullSearchTrackIds": ["1", "2"],
            "nextAction": "读取关键帧并匹配",
        }
        plan = {
            "calls": [{
                "id": "wrongFrames",
                "tool": "getFrames",
                "arguments": {"trackIds": ["1"]},
            }],
            "proposedState": "replan",
        }

        repaired = controller._align_plan_with_acceptance(plan, acceptance)

        self.assertEqual([call["tool"] for call in repaired["calls"]], ["getFrames", "matchImage"])
        self.assertEqual(repaired["calls"][0]["arguments"]["trackIds"], {"$ref": "acceptance.hullSearchTrackIds"})
        self.assertEqual(repaired["calls"][1]["arguments"]["galleryImages"], {"$ref": "hullSearchFrames.keyframes"})

    def test_acceptance_recovery_executes_required_tool_after_regular_round_limit(self) -> None:
        controller = AgentController.__new__(AgentController)
        controller.max_rounds = 1
        controller.acceptance_recovery_rounds = 2
        controller.rounds = []
        controller.working_scope = {}
        controller.meta = dict(self.intent)
        controller.retrieval_page_size = 60
        forced_plans = []

        def fake_round(history, forced_plan=None):
            if forced_plan is None:
                controller.working_scope.update(self._complete_tracks_scope())
                record = {"reflection": {"state": "replan", "reason": "需要精确查库"}}
            else:
                forced_plans.append(forced_plan)
                record = {"reflection": {"state": "sufficient", "reason": "已执行验收工具"}}
            controller.rounds.append(record)
            return record

        controller._round_autonomous = fake_round
        controller._finish_autonomous = lambda last_round, state, reason: {"state": state, "reason": reason}

        result = controller._answer_autonomous()

        self.assertEqual(result["state"], "sufficient")
        self.assertEqual(forced_plans[0]["calls"][0]["tool"], "matchHull")
        self.assertIn("验收", forced_plans[0]["planRepair"])

    def test_track_snapshot_fallback_reads_all_current_tracks(self) -> None:
        controller = AgentController.__new__(AgentController)
        controller.meta = {"timeRange": None}
        controller.working_scope = {
            "partialTracks": {
                "trackIds": ["1", "2"],
                "tracks": [],
                "queryHullNumber": None,
                "queryFinalMatchType": None,
                "hasMore": True,
                "nextOffset": 2,
            }
        }
        acceptance = {
            "pendingRequirements": ["complete_track_scope"],
            "expectedTrackCount": 2,
            "trackCount": 2,
        }

        calls = controller._acceptance_fallback_calls(acceptance)

        self.assertEqual(calls[0]["id"], "tracksSnapshot")
        self.assertEqual(calls[0]["arguments"]["offset"], 0)
        self.assertEqual(calls[0]["arguments"]["limit"], 0)

    def test_count_acceptance_requires_snapshot_then_dedup(self) -> None:
        intent = {
            "targetScope": "track_memory",
            "targetKind": "all",
            "operation": "count",
            "registryRelation": "any",
        }
        partial_scope = {
            "tracks": {
                "ok": True,
                "tracks": [{"trackId": "1"}],
                "trackIds": ["1"],
                "totalTrackCount": 2,
                "hasMore": True,
            }
        }
        partial = build_acceptance_progress(intent, partial_scope)
        self.assertEqual(partial["pendingRequirements"], ["complete_track_scope"])

        full_scope = {
            "tracksSnapshot": {
                "ok": True,
                "tracks": [{"trackId": "1"}, {"trackId": "2"}],
                "trackIds": ["1", "2"],
                "totalTrackCount": 2,
                "hasMore": False,
            }
        }
        before_dedup = build_acceptance_progress(intent, full_scope)
        self.assertEqual(before_dedup["pendingRequirements"], ["deduplicated_count"])

        full_scope["deduplicatedCount"] = {
            "ok": True,
            "highThresholdShipCount": 2,
            "lowThresholdShipCount": 2,
            "countStability": "stable",
        }
        after_dedup = build_acceptance_progress(intent, full_scope)
        self.assertTrue(after_dedup["acceptanceSatisfied"])

    def test_empty_count_plan_is_repaired_to_dedup_chain(self) -> None:
        controller = AgentController.__new__(AgentController)
        controller.meta = {"timeRange": None}
        controller.working_scope = {
            "tracksSnapshot": {
                "ok": True,
                "tracks": [{"trackId": "1"}],
                "trackIds": ["1"],
                "queryHullNumber": None,
                "queryFinalMatchType": None,
                "hasMore": False,
            }
        }
        acceptance = {
            "acceptanceSatisfied": False,
            "pendingRequirements": ["deduplicated_count"],
            "pendingRequirementLabels": ["跨轨迹去重数量"],
            "nextAction": "读取全部轨迹关键帧并执行去重",
        }

        repaired = controller._align_plan_with_acceptance({"calls": [], "proposedState": "uncertain"}, acceptance)

        self.assertEqual([call["tool"] for call in repaired["calls"]], ["getFrames", "dedupTracks"])
        self.assertEqual(repaired["proposedState"], "replan")

    def test_count_fallback_binds_snapshot_to_frames_and_dedup(self) -> None:
        controller = AgentController.__new__(AgentController)
        controller.meta = {"timeRange": None}
        controller.working_scope = {
            "tracksSnapshot": {
                "ok": True,
                "tracks": [{"trackId": "1"}, {"trackId": "2"}],
                "trackIds": ["1", "2"],
                "queryHullNumber": None,
                "queryFinalMatchType": None,
                "hasMore": False,
            }
        }
        acceptance = {"pendingRequirements": ["deduplicated_count"]}

        calls = controller._acceptance_fallback_calls(acceptance)

        self.assertEqual([call["tool"] for call in calls], ["getFrames", "dedupTracks"])
        self.assertEqual(calls[0]["arguments"]["trackIds"], {"$ref": "tracksSnapshot.trackIds"})
        self.assertEqual(calls[1]["arguments"]["tracks"], {"$ref": "tracksSnapshot.tracks"})
        self.assertEqual(calls[1]["arguments"]["keyframesByTrack"], {"$ref": "countFrames.keyframesByTrack"})



if __name__ == "__main__":
    unittest.main()
