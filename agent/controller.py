"""五类海域监控问答的闭环控制器。"""
from __future__ import annotations
import uuid
from pathlib import Path
from typing import Any, Callable
from config import load_config
from memory import MemoryRepository
from services import AgentLLMService, QwenMultimodalEmbedder
from tools import ToolService
from vector_store import VectorCatalog
from .observer import Observer
from .planner import Planner
from .reflector import Reflector

class AgentController:
    TOOL_NAMES = {"getTrack", "getFrames", "getClip", "getRegistry", "matchHull", "listRegistry", "matchText", "matchImage", "verifyTarget", "showEvidence", "dedupTracks"}

    def __init__(self, config: dict[str, Any] | None = None, repository: MemoryRepository | None = None, tools: ToolService | None = None, llm: AgentLLMService | None = None, embedder: QwenMultimodalEmbedder | None = None, vectors: VectorCatalog | None = None, event_handler: Callable[[dict[str, Any]], None] | None = None):
        self.config = config or load_config()
        self.repository = repository or MemoryRepository(self.config)
        self.llm = llm or AgentLLMService(self.config)
        self.tools = tools or ToolService(self.config, self.repository, embedder, self.llm, vectors)
        self.planner = Planner(self.llm, self.TOOL_NAMES)
        self.observer = Observer(self.llm, self.tools)
        self.reflector = Reflector(self.llm)
        settings = self.config["pipeline"]["agent"]
        self.max_rounds = int(settings.get("max_rounds", 3))
        self.display_limit = int(settings.get("display_limit", 3))
        self.session_id = ""
        self.question = ""
        self.meta: dict[str, Any] = {}
        self.rounds: list[dict[str, Any]] = []
        self.tool_chain: list[str] = []
        self.display_record: dict[str, Any] | None = None
        self.display_groups: list[dict[str, Any]] = []
        self.event_handler = event_handler

    def _emit(self, event_type: str, title: str, message: str, **payload: Any) -> None:
        if not self.event_handler:
            return
        try:
            self.event_handler({"type": event_type, "title": title, "message": message, **payload})
        except Exception:
            pass

    def answer(self, question: str) -> dict[str, Any]:
        self.session_id = f"session-{uuid.uuid4().hex[:12]}"
        self.question = question.strip()
        self._emit("status", "创建问答会话", "正在解析问题类型与查询范围")
        self.meta = self.planner.classify(self.question)
        scope = list(self.meta["timeRange"]) if self.meta.get("timeRange") else None
        self._emit("classification", "完成任务识别", "已确定问题类型与检索范围", questionType=self.meta.get("questionType"), queryScope=scope)
        self.rounds, self.tool_chain = [], []
        self.display_record, self.display_groups = None, []
        self.repository.add_session(self.session_id, {"question": self.question, **self.meta})
        try:
            handlers = {"hull": self._answer_hull, "description": self._answer_description, "registry_description": self._answer_registry_description, "out_of_registry": lambda: self._answer_registry(False), "in_registry": lambda: self._answer_registry(True), "count": self._answer_count}
            result = handlers[self.meta["questionType"]]()
        except Exception as error:
            result = self._finish("执行失败", [], f"工具链执行失败：{error}", "uncertain", extra={"error": str(error)})
        self.repository.finish_session(self.session_id, self._session_audit_result(result))
        return result

    def _answer_hull(self) -> dict[str, Any]:
        hull = self.meta.get("hullNumber")
        if not hull:
            return self._finish("无法确认", [], "问题中未解析到舷号", "uncertain")
        first = self._round("查询轨迹记忆中的聚合舷号", [{"id": "directTracks", "tool": "getTrack", "arguments": {"hullNumber": hull}}], "replan", "先检查轨迹级舷号是否稳定命中")
        direct = self._result(first, "directTracks")
        confirmed = [track for track in direct.get("tracks", []) if track["finalMatchType"] == "confirmed"]
        if confirmed:
            return self._finish("确认出现", confirmed, "轨迹级舷号聚合状态为 confirmed", "sufficient", display={"tracks": confirmed, "includeClips": True})
        direct_candidates = [track for track in direct.get("tracks", []) if track["finalMatchType"] in {"candidate", "conflict"}]
        second = self._round("读取目标库项并匹配全视频正式关键帧", [
            {"id": "hullRegistry", "tool": "getRegistry", "arguments": {"hullNumber": hull}},
            {"id": "allTracks", "tool": "getTrack", "condition": {"ref": "hullRegistry.searchable", "equals": True}, "arguments": {}},
            {"id": "allFrames", "tool": "getFrames", "condition": {"ref": "hullRegistry.searchable", "equals": True}, "arguments": {"trackIds": {"$ref": "allTracks.trackIds"}}},
            {"id": "hullImageMatch", "tool": "matchImage", "condition": {"ref": "hullRegistry.searchable", "equals": True}, "arguments": {"queryImages": {"$ref": "hullRegistry.registryReferences"}, "galleryImages": {"$ref": "allFrames.keyframes"}, "topK": 3}},
        ], "sufficient", "库项可检索时返回图像匹配的前三条候选轨迹")
        registry = self._result(second, "hullRegistry")
        if not registry.get("searchable"):
            conclusion = "无法确认" if direct_candidates else "未找到可靠证据"
            reason = "存在非稳定直接候选，但先验库不可检索" if direct_candidates else "轨迹无稳定命中且先验库不存在或不可检索"
            return self._finish(conclusion, direct_candidates, reason, "uncertain" if direct_candidates else "sufficient", display={"tracks": direct_candidates})
        matches = self._result(second, "hullImageMatch").get("matches", [])[:3]
        track_map = {item["trackId"]: item for item in self._result(second, "allTracks").get("tracks", [])}
        tracks = [self._with_match(track_map[item["matchedTrackId"]], item) for item in matches if item["matchedTrackId"] in track_map]
        unsearchable = self._result(second, "allFrames").get("unsearchableTrackIds", [])
        if tracks:
            return self._finish("找到候选轨迹", tracks, "先验库参考图返回前三条相似轨迹，不执行灰区核验", "uncertain", display={"tracks": tracks, "includeRegistry": True})
        return self._finish("无法确认" if unsearchable else "未找到可靠证据", [], "存在不可检索轨迹" if unsearchable else "库参考图未召回候选轨迹", "uncertain" if unsearchable else "sufficient")

    def _answer_description(self) -> dict[str, Any]:
        description = self._description_target()
        first = self._round("按用户描述检索全视频正式关键帧", [
            {"id": "descriptionTracks", "tool": "getTrack", "arguments": {}},
            {"id": "descriptionFrames", "tool": "getFrames", "arguments": {"trackIds": {"$ref": "descriptionTracks.trackIds"}}},
            {"id": "textMatch", "tool": "matchText", "arguments": {"description": description, "galleryImages": {"$ref": "descriptionFrames.keyframes"}}},
        ], "replan", "高分直接回答，灰区进入视觉核验")
        tracks_result = self._result(first, "descriptionTracks")
        frame_result = self._result(first, "descriptionFrames")
        matches = self._result(first, "textMatch").get("matches", [])
        track_map = {item["trackId"]: item for item in tracks_result.get("tracks", [])}
        matched = [item for item in matches if item["scoreBand"] == "match"]
        if matched:
            tracks = [self._with_match(track_map[item["matchedTrackId"]], item) for item in matched[:self.display_limit] if item["matchedTrackId"] in track_map]
            return self._finish("确认出现", tracks, "描述与正式关键帧的统一特征分数达到匹配阈值", "sufficient", display={"tracks": tracks, "includeClips": True})
        uncertain = [item for item in matches if item["scoreBand"] == "uncertain"][:self.display_limit]
        if not uncertain:
            missing = frame_result.get("unsearchableTrackIds", [])
            return self._finish("无法确认" if missing else "未发现", [], "存在不可检索轨迹" if missing else "全部可检索轨迹均低于排除阈值", "uncertain" if missing else "sufficient")
        calls = []
        for index, item in enumerate(uncertain):
            calls.extend([
                {"id": f"verifyDescription{index}", "tool": "verifyTarget", "arguments": {"description": description, "keyframeIds": item["matchedKeyframeIds"]}},
                {"id": f"descriptionClip{index}", "tool": "getClip", "condition": {"ref": f"verifyDescription{index}.decision", "equals": "uncertain"}, "arguments": {"trackId": item["matchedTrackId"]}},
                {"id": f"verifyDescriptionClip{index}", "tool": "verifyTarget", "condition": {"ref": f"verifyDescription{index}.decision", "equals": "uncertain"}, "arguments": {"description": description, "shipSegmentIds": {"$ref": f"descriptionClip{index}.shipSegmentId", "$list": True}}},
            ])
        second = self._round("核验描述检索的灰区视觉证据", calls, "uncertain", "关键帧仍不确定时只读取该轨迹的目标船片段")
        verified, unresolved = [], []
        for index, item in enumerate(uncertain):
            frame_decision = self._result(second, f"verifyDescription{index}").get("decision", "uncertain")
            clip_result = self._result(second, f"descriptionClip{index}")
            final_decision = self._result(second, f"verifyDescriptionClip{index}").get("decision", frame_decision)
            track = self._with_match(track_map[item["matchedTrackId"]], item) if item["matchedTrackId"] in track_map else None
            if track and clip_result.get("shipSegmentId"):
                track["shipSegmentIds"] = [clip_result["shipSegmentId"]]
            if track and final_decision == "match":
                track["scoreBand"] = "verified"
                verified.append(track)
            elif track and final_decision == "uncertain":
                unresolved.append(track)
        if verified:
            return self._finish("确认出现", verified, "灰区视觉证据经模型核验后符合目标描述", "sufficient", display={"tracks": verified, "includeClips": True})
        missing = frame_result.get("unsearchableTrackIds", [])
        if unresolved or missing:
            return self._finish("无法确认", unresolved, "灰区证据或不可检索轨迹未形成稳定结论", "uncertain", display={"tracks": unresolved, "includeClips": True})
        return self._finish("未发现", [], "灰区候选经视觉核验均不符合目标描述", "sufficient")

    def _answer_registry_description(self) -> dict[str, Any]:
        description = self._description_target()
        first = self._round("判断先验库中是否存在符合描述的库项", [
            {"id": "registryCatalog", "tool": "listRegistry", "arguments": {}},
            {"id": "registryTextMatch", "tool": "matchText", "arguments": {"description": description, "galleryImages": {"$ref": "registryCatalog.registryReferences"}, "topK": 3}},
        ], "replan", "先判断目标属于先验库还是视频轨迹，再根据匹配分数决定是否需要补充核验")
        match_result = self._result(first, "registryTextMatch")
        matches = match_result.get("matches", [])
        confirmed = [item for item in matches if item.get("scoreBand") == "match"]
        uncertain = [item for item in matches if item.get("scoreBand") == "uncertain"]
        if confirmed:
            return self._finish("先验库中存在符合描述的库项", [], "文本与先验库参考图完成特征匹配", "sufficient", extra={"registryMatches": confirmed, "registryDescription": description})
        if uncertain:
            return self._finish("无法确认", [], "先验库存在相似但未达到确定阈值的库项", "uncertain", extra={"registryMatches": uncertain, "registryDescription": description})
        return self._finish("先验库中未找到符合描述的库项", [], "先验库参考图均未达到匹配阈值", "sufficient", extra={"registryMatches": [], "registryDescription": description})

    def _answer_registry(self, want_in_registry: bool) -> dict[str, Any]:
        time_range = self.meta.get("timeRange")
        first = self._round("筛选时间范围轨迹并执行舷号精确查库", [
            {"id": "rangeTracks", "tool": "getTrack", "arguments": {"timeRange": time_range}},
            {"id": "exactHull", "tool": "matchHull", "arguments": {"hullNumberArray": {"$ref": "rangeTracks.tracks", "$map": "finalHullNumber", "$compact": True}}},
        ], "replan", "精确命中直接判为在库，其余轨迹进入图像匹配")
        tracks = self._result(first, "rangeTracks").get("tracks", [])
        exact_result = self._result(first, "exactHull")
        exact_hulls = set(exact_result.get("matchedHullNumbers", []))
        exact_tracks = [self._attach_exact_registry(item, exact_result) for item in tracks if item["finalMatchType"] == "confirmed" and item.get("finalHullNumber") in exact_hulls]
        remaining = [item for item in tracks if item["trackId"] not in {track["trackId"] for track in exact_tracks}]
        if not remaining:
            result_tracks = exact_tracks if want_in_registry else []
            return self._finish("查询完成" if result_tracks else "未发现", result_tracks, "全部轨迹已由舷号精确查库完成分类", "sufficient", extra={"unresolvedTracks": []}, display={"tracks": result_tracks, "includeClips": True, "includeRegistry": True})
        remaining_ids = [item["trackId"] for item in remaining]
        second = self._round("将剩余轨迹与完整先验库参考图匹配", [
            {"id": "fullRegistry", "tool": "listRegistry", "arguments": {}},
            {"id": "remainingFrames", "tool": "getFrames", "condition": {"ref": "fullRegistry.registryItems"}, "arguments": {"trackIds": remaining_ids}},
            {"id": "registryImageMatch", "tool": "matchImage", "condition": {"ref": "fullRegistry.registryItems"}, "arguments": {"queryImages": {"$ref": "remainingFrames.keyframes"}, "galleryImages": {"$ref": "fullRegistry.registryReferences"}, "topK": 3}},
        ], "replan", "匹配和排除阈值直接分类，灰区进入核验")
        registry = self._result(second, "fullRegistry")
        frame_result = self._result(second, "remainingFrames")
        if not registry.get("registryItems"):
            if want_in_registry:
                return self._finish("未发现", [], "先验库为空，不存在可匹配库项", "sufficient")
            return self._finish("查询完成", remaining, "先验库为空，查询范围内轨迹均判为未在库", "sufficient", display={"tracks": remaining, "includeClips": True})
        matches = self._result(second, "registryImageMatch").get("matches", [])
        by_track = {track_id: [item for item in matches if item["matchedTrackId"] == track_id] for track_id in remaining_ids}
        library_complete = bool(registry.get("registryItems")) and not registry.get("unsearchableRegistryIds")
        image_in: list[tuple[dict[str, Any], dict[str, Any]]] = []
        image_out: list[tuple[dict[str, Any], dict[str, Any]]] = []
        uncertain: list[tuple[dict[str, Any], dict[str, Any]]] = []
        unresolved: list[dict[str, Any]] = []
        for track in remaining:
            candidates = by_track.get(track["trackId"], [])
            direct_match = next((item for item in candidates if item["scoreBand"] == "match"), None)
            gray_match = next((item for item in candidates if item["scoreBand"] == "uncertain"), None)
            if direct_match:
                image_in.append((track, direct_match))
            elif candidates and all(item["scoreBand"] == "mismatch" for item in candidates) and library_complete:
                image_out.append((track, candidates[0]))
            elif gray_match:
                uncertain.append((track, gray_match))
            else:
                unresolved.append(track)
        if uncertain:
            calls = []
            for index, (track, match) in enumerate(uncertain):
                calls.extend([
                    {"id": f"verifyRegistry{index}", "tool": "verifyTarget", "arguments": {"registryReferenceIds": match["matchedRegistryReferenceIds"], "keyframeIds": match["matchedKeyframeIds"]}},
                    {"id": f"registryClip{index}", "tool": "getClip", "condition": {"ref": f"verifyRegistry{index}.decision", "equals": "uncertain"}, "arguments": {"trackId": track["trackId"], "timeRange": time_range}},
                    {"id": f"verifyRegistryClip{index}", "tool": "verifyTarget", "condition": {"ref": f"verifyRegistry{index}.decision", "equals": "uncertain"}, "arguments": {"registryReferenceIds": match["matchedRegistryReferenceIds"], "shipSegmentIds": {"$ref": f"registryClip{index}.shipSegmentId", "$list": True}}},
                ])
            third = self._round("核验轨迹与库项图像匹配的灰区", calls, "uncertain", "关键帧仍不确定时只读取该轨迹的目标船片段")
            for index, pair in enumerate(uncertain):
                frame_decision = self._result(third, f"verifyRegistry{index}").get("decision", "uncertain")
                clip_result = self._result(third, f"registryClip{index}")
                final_decision = self._result(third, f"verifyRegistryClip{index}").get("decision", frame_decision)
                if clip_result.get("shipSegmentId"):
                    pair[1]["shipSegmentIds"] = [clip_result["shipSegmentId"]]
                if final_decision == "match":
                    image_in.append(pair)
                elif final_decision == "mismatch" and library_complete:
                    image_out.append(pair)
                else:
                    unresolved.append(pair[0])
        in_tracks = exact_tracks + [self._with_match(track, match) for track, match in image_in]
        out_tracks = [self._with_match(track, match) for track, match in image_out]
        result_tracks = in_tracks if want_in_registry else out_tracks
        if result_tracks:
            conclusion = "部分确认" if unresolved else "查询完成"
        else:
            conclusion = "无法确认" if unresolved else "未发现"
        reason = "先执行舷号精确查库，再对剩余轨迹执行库参考图匹配和灰区核验"
        extra = {"unresolvedTracks": unresolved, "unsearchableTrackIds": frame_result.get("unsearchableTrackIds", []), "exactInRegistryTrackIds": [item["trackId"] for item in exact_tracks]}
        return self._finish(conclusion, result_tracks, reason, "uncertain" if unresolved else "sufficient", extra=extra, display={"tracks": result_tracks, "includeClips": True, "includeRegistry": True})

    def _answer_count(self) -> dict[str, Any]:
        time_range = self.meta.get("timeRange")
        first = self._round("筛选时间范围轨迹并执行跨轨迹去重", [
            {"id": "countTracks", "tool": "getTrack", "arguments": {"timeRange": time_range}},
            {"id": "countFrames", "tool": "getFrames", "arguments": {"trackIds": {"$ref": "countTracks.trackIds"}}},
            {"id": "dedupResult", "tool": "dedupTracks", "arguments": {"tracks": {"$ref": "countTracks.tracks"}, "keyframesByTrack": {"$ref": "countFrames.keyframesByTrack"}}},
        ], "sufficient", "两种阈值分组只用于报告计数敏感性")
        tracks = self._result(first, "countTracks").get("tracks", [])
        dedup = self._result(first, "dedupResult")
        extra = {"statistics": dedup}
        if not tracks:
            return self._finish("数量为 0", [], "查询范围内没有轨迹", "sufficient", extra=extra)
        frame_groups = self._result(first, "countFrames").get("keyframesByTrack", {})
        representatives = []
        for group in dedup.get("highGroups", [])[:self.display_limit]:
            if not group:
                continue
            track_id = group[0]
            frames = frame_groups.get(track_id, {}).get("keyframes", [])
            best = max(frames, key=lambda item: item.get("retentionScore", 0), default=None)
            track = next((item for item in tracks if item["trackId"] == track_id), None)
            if track:
                representatives.append(dict(track, matchedKeyframeIds=[best["keyframeId"]] if best else []))
        conclusion = f"高阈值计数 {dedup.get('highThresholdShipCount', len(tracks))}，低阈值计数 {dedup.get('lowThresholdShipCount', len(tracks))}"
        uncertainty = "sufficient" if dedup.get("countStability") == "stable" else "uncertain"
        reason = f"计数状态为 {dedup.get('countStability', 'unknown')}，两种结果表示阈值敏感性"
        return self._finish(conclusion, tracks, reason, uncertainty, extra=extra, display={"tracks": representatives})

    def _round(self, goal: str, calls: list[dict[str, Any]], default_state: str, reason: str, evidence_gap: str | None = None) -> dict[str, Any]:
        if len(self.rounds) >= self.max_rounds:
            raise RuntimeError("达到最大推理轮次")
        round_number = len(self.rounds) + 1
        self._emit("agent_start", "PlanAgent", "正在分析目标并组织工具计划", round=round_number, role="planner")
        plan = self.planner.build(goal, calls, self.meta.get("timeRange"), evidence_gap, lambda delta: self._emit("agent_delta", "PlanAgent", "", round=round_number, role="planner", delta=delta), intent=self.meta)
        public_plan = self._public_plan(plan, calls)
        self._emit("agent_end", "PlanAgent", "规划完成", round=round_number, role="planner", calls=public_plan["calls"], modelSummary=plan.get("modelPlan"), fallback=plan.get("modelFallback"))

        self._emit("agent_start", "ObserveAgent", "正在执行工具并整理轨迹证据", round=round_number, role="observer")
        observed = self.observer.execute(plan, on_delta=lambda delta: self._emit("agent_delta", "ObserveAgent", "", round=round_number, role="observer", delta=delta))
        self._emit("agent_end", "ObserveAgent", "观察完成", round=round_number, role="observer", calls=observed["summary"].get("calls", []), modelSummary=observed["summary"].get("modelObservation"), fallback=observed["summary"].get("modelFallback"))

        self._emit("agent_start", "ReflectAgent", "正在检查证据充分性、冲突与后续动作", round=round_number, role="reflector")
        reflection = self.reflector.review(default_state, reason, observed["summary"], evidence_gap, lambda delta: self._emit("agent_delta", "ReflectAgent", "", round=round_number, role="reflector", delta=delta))
        self._emit("agent_end", "ReflectAgent", reflection.get("reason", reason), round=round_number, role="reflector", state=reflection.get("state"), evidenceGap=reflection.get("evidenceGap"), modelSummary=reflection.get("modelReflection"), fallback=reflection.get("modelFallback"))
        round_id = f"round-{uuid.uuid4().hex[:12]}"
        self.repository.add_round(round_id, self.session_id, public_plan, reflection)
        self._store_observations(round_id, observed)
        record = {"roundId": round_id, "plan": public_plan, "observation": observed["summary"], "reflection": reflection, "scope": observed["scope"]}
        self.rounds.append(record)
        return record

    def _display_tracks(self, tracks: list[dict[str, Any]], include_clips: bool = True, include_registry: bool = False) -> None:
        primary = tracks[:self.display_limit]
        if not primary or self.display_record is not None:
            return
        calls, layouts = [], []
        for index, track in enumerate(primary):
            track_id = str(track["trackId"])
            keyframe_ids = self._ids(track, "matchedKeyframeIds", "queryKeyframeIds", "keyframeIds")[:3]
            segment_ids = self._ids(track, "shipSegmentIds")[:1]
            reference_ids = self._ids(track, "matchedRegistryReferenceIds", "queryRegistryReferenceIds", "registryReferenceIds")[:6] if include_registry else []
            frame_call = f"displayFrames{index}"
            clip_call = f"displayClip{index}"
            show_call = f"displayTrack{index}"
            if not keyframe_ids:
                calls.append({"id": frame_call, "tool": "getFrames", "arguments": {"trackIds": [track_id]}})
            if include_clips and not segment_ids:
                calls.append({"id": clip_call, "tool": "getClip", "arguments": {"trackId": track_id, "timeRange": self.meta.get("timeRange")}})
            arguments: dict[str, Any] = {"keyframeIds": keyframe_ids or {"$ref": f"{frame_call}.keyframeIds"}}
            if include_clips:
                arguments["shipSegmentIds"] = segment_ids or {"$ref": f"{clip_call}.shipSegmentId", "$list": True}
            if reference_ids:
                arguments["registryReferenceIds"] = reference_ids
            calls.append({"id": show_call, "tool": "showEvidence", "arguments": arguments})
            layouts.append({"trackId": track_id, "showCallId": show_call, "clipCallId": clip_call if include_clips and not segment_ids else None, "includeClip": include_clips})
        self._display("展示主轨迹证据", calls)
        scope = self.display_record.get("scope", {}) if self.display_record else {}
        for layout in layouts:
            shown = scope.get(layout["showCallId"], {})
            clip_result = scope.get(layout["clipCallId"], {}) if layout.get("clipCallId") else {}
            segment_ids = shown.get("shownShipSegmentIds", [])
            clip_error = None
            if layout.get("includeClip") and not segment_ids:
                clip_error = clip_result.get("error") or "clip_unavailable"
            self.display_groups.append({"trackId": layout["trackId"], "keyframeIds": shown.get("shownKeyframeIds", []), "shipSegmentIds": segment_ids, "registryReferenceIds": shown.get("shownRegistryReferenceIds", []), "clipError": clip_error})

    def _display(self, goal: str, calls: list[dict[str, Any]]) -> None:
        if self.display_record is not None or not calls:
            return
        try:
            self._emit("evidence", "整理视觉证据", "正在读取关键帧、目标船片段与先验库参考图")
            plan = self.planner.build(goal, calls, self.meta.get("timeRange"))
            observed = self.observer.execute(plan)
            display_id = f"display-{uuid.uuid4().hex[:12]}"
            self._append_tool_chain(observed)
            self.display_record = {"displayId": display_id, "plan": self._public_plan(plan, calls), "observation": observed["summary"], "scope": observed["scope"]}
        except Exception as error:
            self.display_record = {"ok": False, "error": str(error), "scope": {}}

    def _store_observations(self, record_id: str, observed: dict[str, Any]) -> None:
        for observation in observed["observations"]:
            if observation.get("skipped"):
                continue
            evidence_id = f"evidence-{uuid.uuid4().hex[:12]}"
            audit_result = Observer._summarize_observation(observation)
            self.repository.add_evidence(evidence_id, record_id, audit_result, {"tool": observation["tool"], "callId": observation["id"]})
            self.tool_chain.append(f"{observation['tool']}({observation['id']})")

    @staticmethod
    def _session_audit_result(result: dict[str, Any]) -> dict[str, Any]:
        tracks = result.get("tracks", [])
        compact_tracks = [
            {key: item.get(key) for key in ("trackId", "startTime", "endTime", "finalHullNumber", "finalMatchType", "embeddingScore", "scoreBand") if item.get(key) not in (None, "")}
            for item in tracks[:10]
        ]
        return {
            "sessionId": result.get("sessionId"),
            "question": result.get("question"),
            "questionType": result.get("questionType"),
            "conclusion": result.get("conclusion"),
            "answerText": result.get("answerText"),
            "queryScope": result.get("queryScope"),
            "uncertainty": result.get("uncertainty"),
            "trackCount": len(tracks),
            "tracks": compact_tracks,
            "toolChain": result.get("toolChain", [])[:30],
            "evidence": result.get("evidence", {}),
            "roundCount": len(result.get("rounds", [])),
        }

    def _append_tool_chain(self, observed: dict[str, Any]) -> None:
        for observation in observed["observations"]:
            if not observation.get("skipped"):
                self.tool_chain.append(f"{observation['tool']}({observation['id']})")

    def _finish(self, conclusion: str, tracks: list[dict[str, Any]], reason: str, state: str, extra: dict[str, Any] | None = None, display: dict[str, Any] | None = None) -> dict[str, Any]:
        if display and display.get("tracks"):
            self._display_tracks(display["tracks"], display.get("includeClips", True) is not False, bool(display.get("includeRegistry")))
        primary = [item["trackId"] for item in tracks[:self.display_limit]]
        result = {"sessionId": self.session_id, "question": self.question, "questionType": self.meta.get("questionType"), "conclusion": conclusion, "answerText": f"{conclusion}。{reason}", "queryScope": list(self.meta["timeRange"]) if self.meta.get("timeRange") else None, "toolChain": self.tool_chain, "tracks": tracks, "evidence": self._collect_evidence(), "displayGroups": self.display_groups, "display": self._public_display(), "uncertainty": state, "primaryTrackIds": primary, "remainingTrackIds": [item["trackId"] for item in tracks[self.display_limit:]], "rounds": [{key: value for key, value in item.items() if key != "scope"} for item in self.rounds]}
        if extra:
            result.update(extra)
        self._emit("synthesis", "生成最终回答", reason, conclusion=conclusion, state=state, trackCount=len(tracks))
        return result

    def _collect_evidence(self) -> dict[str, list[str]]:
        if self.display_groups:
            return {key: list(dict.fromkeys(value for group in self.display_groups for value in group[key])) for key in ("keyframeIds", "shipSegmentIds", "registryReferenceIds")}
        collected = {"keyframeIds": [], "shipSegmentIds": [], "registryReferenceIds": []}
        key_map = {"keyframeIds": "keyframeIds", "queryKeyframeIds": "keyframeIds", "matchedKeyframeIds": "keyframeIds", "shipSegmentIds": "shipSegmentIds", "registryReferenceIds": "registryReferenceIds", "queryRegistryReferenceIds": "registryReferenceIds", "matchedRegistryReferenceIds": "registryReferenceIds"}
        def visit(value: Any) -> None:
            if isinstance(value, dict):
                for key, item in value.items():
                    if key == "shipSegmentId" and item:
                        collected["shipSegmentIds"].append(str(item))
                    elif key in key_map and isinstance(item, list):
                        collected[key_map[key]].extend(str(entry) for entry in item if entry)
                    visit(item)
            elif isinstance(value, list):
                for item in value:
                    visit(item)
        for round_item in self.rounds:
            visit(round_item.get("scope", {}))
        return {key: list(dict.fromkeys(values)) for key, values in collected.items()}

    def _public_display(self) -> dict[str, Any] | None:
        if not self.display_record:
            return None
        return {key: value for key, value in self.display_record.items() if key != "scope"}

    @staticmethod
    def _result(record: dict[str, Any], call_id: str) -> dict[str, Any]:
        value = record.get("scope", {}).get(call_id, {})
        return value if isinstance(value, dict) else {}

    def _public_plan(self, plan: dict[str, Any], calls: list[dict[str, Any]]) -> dict[str, Any]:
        public = {key: value for key, value in plan.items() if key != "calls"}
        public["calls"] = [{"id": call["id"], "tool": call["tool"], "arguments": self._compact_arguments(call.get("arguments", {})), "condition": call.get("condition")} for call in calls]
        return public

    @staticmethod
    def _compact_arguments(arguments: Any) -> Any:
        if isinstance(arguments, dict):
            if "$ref" in arguments:
                return arguments
            return {key: AgentController._compact_arguments(value) for key, value in arguments.items()}
        if isinstance(arguments, list) and arguments and isinstance(arguments[0], dict):
            return [item.get("keyframeId") or item.get("referenceId") or item.get("trackId") for item in arguments]
        return arguments

    @staticmethod
    def _ids(item: dict[str, Any], *keys: str) -> list[str]:
        values = []
        for key in keys:
            value = item.get(key, [])
            values.extend(value if isinstance(value, list) else [value] if value else [])
        return list(dict.fromkeys(str(value) for value in values if value))

    @staticmethod
    def _with_match(track: dict[str, Any], match: dict[str, Any]) -> dict[str, Any]:
        result = dict(track)
        for key in ("matchedRegistryId", "embeddingScore", "scoreBand", "queryKeyframeIds", "queryRegistryReferenceIds", "matchedKeyframeIds", "matchedRegistryReferenceIds", "shipSegmentIds"):
            if key in match:
                result[key] = match[key]
        return result

    @staticmethod
    def _attach_exact_registry(track: dict[str, Any], exact_result: dict[str, Any]) -> dict[str, Any]:
        result = dict(track)
        items = exact_result.get("exactMatches", {}).get(track.get("finalHullNumber"), [])
        result["matchedRegistryIds"] = [item["registryId"] for item in items]
        result["registryReferenceIds"] = [reference["referenceId"] for item in items for reference in item.get("references", []) if Path(reference.get("imagePath", "")).is_file()]
        result["registryDecision"] = "exact"
        return result

    def _description_target(self) -> str:
        question = self.question
        for prefix in ("数据库中有没有出现", "数据库中是否出现", "先验库中有没有出现", "先验库中是否出现", "视频中有没有出现", "视频中是否出现", "有没有出现", "找一下", "查找一下"):
            question = question.replace(prefix, "")
        return question.strip("？?。 ") or self.question
