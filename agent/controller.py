"""面向轨迹记忆的闭环智能体控制器。"""
from __future__ import annotations
import uuid
from pathlib import Path
from typing import Any, Callable
from config import load_config
from memory import MemoryRepository, normalize_hull_number
from services import AgentLLMService, QwenMultimodalEmbedder
from tools import ToolService
from vector_store import VectorCatalog
from .acceptance import build_acceptance_progress, compact_acceptance
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
        self.acceptance_recovery_rounds = int(settings.get("acceptance_recovery_rounds", 2))
        self.display_limit = int(settings.get("display_limit", 3))
        self.plan_mode = str(settings.get("plan_mode", "guided")).strip().lower()
        if self.plan_mode not in {"guided", "autonomous"}:
            self.plan_mode = "guided"
        self.retrieval_page_size = int(settings.get("retrieval_page_size", 60))
        self.session_id = ""
        self.question = ""
        self.meta: dict[str, Any] = {}
        self.rounds: list[dict[str, Any]] = []
        self.tool_chain: list[str] = []
        self.tool_records: list[dict[str, Any]] = []
        self.display_record: dict[str, Any] | None = None
        self.display_groups: list[dict[str, Any]] = []
        self.working_scope: dict[str, Any] = {}
        self.plan_blueprint: list[dict[str, Any]] = []
        self.event_handler = event_handler

    def _emit(self, event_type: str, title: str, message: str, **payload: Any) -> None:
        if not self.event_handler:
            return
        try:
            self.event_handler({"type": event_type, "title": title, "message": message, **payload})
        except Exception:
            pass

    def _plan_step_id_for_tool(self, tool: Any) -> str | None:
        tool_name = str(tool or "")
        for step in self.plan_blueprint:
            if tool_name in step.get("tools", []):
                return str(step.get("stepId") or "") or None
        return None

    def _emit_observer_tool_event(self, round_number: int, event: dict[str, Any]) -> None:
        self._emit(
            "agent_tool",
            "ObserveAgent",
            "",
            round=round_number,
            role="observer",
            planStepId=self._plan_step_id_for_tool(event.get("tool")),
            **event,
        )

    def answer(self, question: str, top_k: int | None = None) -> dict[str, Any]:
        self.session_id = f"session-{uuid.uuid4().hex[:12]}"
        self.question = question.strip()
        self._emit("status", "IntentAgent", "正在按规则表解析用户意图")
        self.meta = self.planner.classify(self.question)
        self.meta = self._guard_meta(self.meta)
        self.plan_blueprint = self.planner.build_execution_blueprint(self.meta)
        self.meta["planBlueprint"] = self.plan_blueprint
        scope = list(self.meta["timeRange"]) if self.meta.get("timeRange") else None
        self._emit(
            "classification",
            "IntentAgent 意图识别完成",
            "已选择规则并编译检索策略",
            questionType=self.meta.get("questionType"),
            strategy=self.meta.get("strategy"),
            operation=self.meta.get("operation"),
            targetScope=self.meta.get("targetScope"),
            targetKind=self.meta.get("targetKind"),
            registryRelation=self.meta.get("registryRelation"),
            description=self.meta.get("description"),
            hullNumber=self.meta.get("hullNumber"),
            selectedRules=self.meta.get("selectedRules") or [],
            intentSource=self.meta.get("intentSource"),
            intentConfidence=self.meta.get("intentConfidence"),
            expectedOutcome=self.meta.get("expectedOutcome"),
            successCriteria=self.meta.get("successCriteria"),
            nextAgentFocus=self.meta.get("nextAgentFocus"),
            queryScope=scope,
            planBlueprint=self.plan_blueprint,
        )
        self.rounds, self.tool_chain, self.tool_records = [], [], []
        self.display_record, self.display_groups = None, []
        self.working_scope = {}
        agent_settings = self.config.get("pipeline", {}).get("agent", {})
        self.plan_mode = str(agent_settings.get("plan_mode", self.plan_mode)).strip().lower()
        if self.plan_mode not in {"guided", "autonomous"}:
            self.plan_mode = "guided"
        self.max_rounds = int(agent_settings.get("max_rounds", self.max_rounds))
        self.acceptance_recovery_rounds = int(agent_settings.get("acceptance_recovery_rounds", self.acceptance_recovery_rounds))
        self.display_limit = int(agent_settings.get("display_limit", self.display_limit))
        self.retrieval_page_size = int(agent_settings.get("retrieval_page_size", getattr(self, "retrieval_page_size", 60)))
        default_top_k = int(self.config.get("pipeline", {}).get("retrieval", {}).get("top_k", 3))
        self.query_top_k = max(1, min(20, int(top_k if top_k is not None else default_top_k)))
        self.meta["planMode"] = self.plan_mode
        self.meta["maxRounds"] = self.max_rounds
        self.meta["acceptanceRecoveryRounds"] = self.acceptance_recovery_rounds
        self.meta["retrievalPageSize"] = self.retrieval_page_size
        self.meta["retrievalTopK"] = self.query_top_k
        self.repository.add_session(self.session_id, {"question": self.question, **self.meta})
        self._emit("status", "PlanAgent", f"当前规划模式：{'硬编码辅助' if self.plan_mode == 'guided' else '自主规划（PlanAgent规划，ObserveAgent执行）'}", planMode=self.plan_mode)
        try:
            if self.plan_mode == "autonomous":
                result = self._answer_autonomous()
                result = self._finalize_answer(result)
                self.repository.finish_session(self.session_id, self._session_audit_result(result))
                return result
            handlers = {
                "hull": self._answer_hull,
                "registry_hull": self._answer_registry_hull,
                "description": self._answer_description,
                "registry_description": self._answer_registry_description,
                "cross_reference": self._answer_cross_reference,
                "track_list": self._answer_track_list,
                "registry_list": self._answer_registry_list,
                "relation_description": self._answer_relation_description,
                "out_of_registry": lambda: self._answer_registry(False),
                "in_registry": lambda: self._answer_registry(True),
                "count": self._answer_count,
                "description_count": self._answer_description_count,
                "registry_count": self._answer_registry_count,
                "registry_description_count": self._answer_registry_description_count,
            }
            question_type = self.meta.get("questionType")
            if question_type not in handlers:
                raise ValueError(f"未知问题策略：{question_type}")
            result = handlers[question_type]()
            result = self._finalize_answer(result)
        except Exception as error:
            result = self._finish("执行失败", [], f"工具链执行失败：{error}", "uncertain", extra={"error": str(error)})
        self.repository.finish_session(self.session_id, self._session_audit_result(result))
        return result


    def _answer_registry_hull(self) -> dict[str, Any]:
        hull = self.meta.get("hullNumber")
        if not hull:
            return self._finish("无法确认", [], "问题中没有明确舷号", "uncertain")
        round_result = self._round("按舷号查询先验库库项", [{"id": "registryHull", "tool": "getRegistry", "arguments": {"hullNumber": hull}}], "sufficient", "返回全部匹配库项及其参考图")
        result = self._result(round_result, "registryHull")
        items = result.get("registryItems", [])
        refs = result.get("registryReferenceIds", [])
        if not items:
            return self._finish("先验库中未找到", [], f"没有找到舷号 {hull} 对应的库项", "sufficient", extra={"registryItems": [], "registryReferenceIds": []})
        return self._finish("先验库中找到匹配库项", [], f"舷号 {hull} 匹配到 {len(items)} 个库项", "sufficient", extra={"registryItems": items, "registryReferenceIds": refs})

    def _answer_registry_list(self) -> dict[str, Any]:
        round_result = self._round("读取先验库并列出库项", [{"id": "registryCatalog", "tool": "listRegistry", "arguments": {}}], "sufficient", "返回先验库全部库项")
        result = self._result(round_result, "registryCatalog")
        items = result.get("registryItems", [])
        return self._finish("先验库查询完成", [], f"当前先验库共有 {len(items)} 个库项", "sufficient", extra={"registryItems": items, "registryReferenceIds": result.get("registryReferenceIds", [])})

    def _answer_track_list(self) -> dict[str, Any]:
        time_range = self.meta.get("timeRange")
        round_result = self._round("按时间范围读取视频轨迹列表", [{"id": "trackList", "tool": "getTrack", "arguments": {"timeRange": time_range}}], "sufficient", "列出符合时间和轨迹条件的船舶")
        result = self._result(round_result, "trackList")
        tracks = result.get("tracks", [])
        return self._finish("轨迹查询完成", tracks, f"查询到 {len(tracks)} 条轨迹", "sufficient", extra={"totalTrackCount": result.get("totalTrackCount", len(tracks))}, display={"tracks": tracks, "includeClips": True})

    def _answer_description_count(self) -> dict[str, Any]:
        """按描述筛选轨迹，再对命中轨迹做跨轨迹去重统计。"""
        description = self.meta.get("description") or self._description_target()
        page_size = 0
        offset, has_more = 0, True
        track_map: dict[str, dict[str, Any]] = {}
        confirmed_map: dict[str, dict[str, Any]] = {}
        uncertain_map: dict[str, dict[str, Any]] = {}
        missing_track_ids: set[str] = set()
        searched_count = 0
        while has_more and len(self.rounds) < max(1, self.max_rounds - 1):
            page_number = 1 if page_size == 0 else offset // page_size + 1
            track_call = f"countDescTracks{page_number}"
            frame_call = f"countDescFrames{page_number}"
            match_call = f"countDescMatch{page_number}"
            current = self._round(
                f"读取第 {page_number} 页轨迹并按描述筛选",
                [
                    {"id": track_call, "tool": "getTrack", "arguments": {"timeRange": self.meta.get("timeRange"), "offset": offset, "limit": page_size}},
                    {"id": frame_call, "tool": "getFrames", "arguments": {"trackIds": {"$ref": f"{track_call}.trackIds"}}},
                    {"id": match_call, "tool": "matchText", "arguments": {"description": description, "galleryImages": {"$ref": f"{frame_call}.keyframes"}, "topK": 20}},
                ],
                "replan",
                "当前页筛选完成后，若 hasMore=true 则继续读取下一页",
            )
            tracks_result = self._result(current, track_call)
            frames_result = self._result(current, frame_call)
            matches = self._result(current, match_call).get("matches", [])
            page_tracks = tracks_result.get("tracks", [])
            searched_count += len(page_tracks)
            track_map.update({str(item["trackId"]): item for item in page_tracks})
            missing_track_ids.update(str(value) for value in frames_result.get("unsearchableTrackIds", []))
            for item in matches:
                track_id = str(item.get("matchedTrackId") or "")
                if track_id not in track_map:
                    continue
                band = item.get("scoreBand")
                if band == "match":
                    previous = confirmed_map.get(track_id)
                    if previous is None or float(item.get("embeddingScore") or 0) > float(previous.get("embeddingScore") or 0):
                        confirmed_map[track_id] = self._with_match(track_map[track_id], item)
                elif band == "uncertain":
                    previous = uncertain_map.get(track_id)
                    if previous is None or float(item.get("embeddingScore") or 0) > float(previous.get("embeddingScore") or 0):
                        uncertain_map[track_id] = item
            has_more = bool(tracks_result.get("hasMore"))
            next_offset = tracks_result.get("nextOffset")
            if not has_more or next_offset is None:
                break
            offset = int(next_offset)

        confirmed = list(confirmed_map.values())
        uncertain_count = len(uncertain_map)
        if not confirmed:
            state = "uncertain" if uncertain_count or missing_track_ids or has_more else "sufficient"
            reason = f"目标描述：{description}；未找到达到匹配阈值的轨迹"
            if uncertain_count:
                reason += f"；另有 {uncertain_count} 条灰区轨迹"
            if has_more:
                reason += "；仍有轨迹未读取"
            return self._finish(
                "符合描述的船舶数量为 0",
                [],
                reason,
                state,
                extra={
                    "description": description,
                    "trackCount": 0,
                    "dedupShipCount": 0,
                    "uncertainMatchCount": uncertain_count,
                    "searchedTrackCount": searched_count,
                    "hasMore": has_more,
                    "unsearchableTrackIds": sorted(missing_track_ids),
                },
            )

        if len(self.rounds) >= self.max_rounds:
            return self._finish(
                f"符合描述的轨迹数量为 {len(confirmed)}",
                confirmed,
                f"目标描述：{description}；已找到 {len(confirmed)} 条轨迹，但没有剩余轮次完成去重",
                "uncertain",
                extra={
                    "description": description,
                    "trackCount": len(confirmed),
                    "dedupShipCount": None,
                    "uncertainMatchCount": uncertain_count,
                    "searchedTrackCount": searched_count,
                    "hasMore": has_more,
                    "unsearchableTrackIds": sorted(missing_track_ids),
                },
                display={"tracks": confirmed, "includeClips": True},
            )

        dedup_round = self._round(
            "对描述命中轨迹执行跨轨迹去重",
            [
                {"id": "countDescConfirmedFrames", "tool": "getFrames", "arguments": {"trackIds": [item["trackId"] for item in confirmed]}},
                {"id": "countDescDedup", "tool": "dedupTracks", "arguments": {"tracks": confirmed, "keyframesByTrack": {"$ref": "countDescConfirmedFrames.keyframesByTrack"}}},
            ],
            "sufficient",
            "高阈值分组对应保守船舶数，低阈值分组对应敏感船舶数",
        )
        frames_result = self._result(dedup_round, "countDescConfirmedFrames")
        dedup = self._result(dedup_round, "countDescDedup")
        high_count = int(dedup.get("highThresholdShipCount", len(confirmed)))
        low_count = int(dedup.get("lowThresholdShipCount", len(confirmed)))
        stability = dedup.get("countStability", "unknown")
        state = "sufficient" if stability == "stable" and not uncertain_count and not missing_track_ids and not has_more else "uncertain"
        reason = (
            f"目标描述：{description}；匹配轨迹 {len(confirmed)} 条；"
            f"去重后高阈值船舶数 {high_count}、低阈值船舶数 {low_count}；"
            f"计数状态 {stability}"
        )
        if uncertain_count:
            reason += f"；另有 {uncertain_count} 条灰区轨迹未计入"
        representatives = []
        frame_groups = frames_result.get("keyframesByTrack", {})
        for group in dedup.get("highGroups", [])[: self.display_limit]:
            if not group:
                continue
            track_id = str(group[0])
            frames = frame_groups.get(track_id, {}).get("keyframes", [])
            best = max(frames, key=lambda item: item.get("retentionScore", 0), default=None)
            track = next((item for item in confirmed if str(item["trackId"]) == track_id), None)
            if track:
                representatives.append(dict(track, matchedKeyframeIds=[best["keyframeId"]] if best else []))
        return self._finish(
            f"符合描述的船舶数量约为 {high_count}",
            confirmed,
            reason,
            state,
            extra={
                "description": description,
                "trackCount": len(confirmed),
                "dedupShipCount": high_count,
                "lowThresholdShipCount": low_count,
                "uncertainMatchCount": uncertain_count,
                "searchedTrackCount": searched_count,
                "hasMore": has_more,
                "unsearchableTrackIds": sorted(missing_track_ids),
                "statistics": dedup,
            },
            display={"tracks": representatives or confirmed, "includeClips": True},
        )

    def _answer_registry_count(self) -> dict[str, Any]:
        round_result = self._round("统计先验库库项数量", [{"id": "registryCatalog", "tool": "listRegistry", "arguments": {}}], "sufficient", "按库项而不是参考图数量统计")
        result = self._result(round_result, "registryCatalog")
        items = result.get("registryItems", [])
        return self._finish(f"先验库共有 {len(items)} 个库项", [], "按库项编号去重统计", "sufficient", extra={"registryItems": items, "registryReferenceIds": result.get("registryReferenceIds", [])})

    def _answer_registry_description_count(self) -> dict[str, Any]:
        description = self.meta.get("description") or self._description_target()
        first = self._round("按描述检索先验库并统计库项", [
            {"id": "registryCatalog", "tool": "listRegistry", "arguments": {}},
            {"id": "registryDescriptionMatch", "tool": "matchText", "arguments": {"description": description, "galleryImages": {"$ref": "registryCatalog.registryReferences"}, "topK": 20}},
        ], "sufficient", "按库项编号去重，不能把同一库项的多张参考图重复计数")
        catalog = self._result(first, "registryCatalog")
        matches = self._result(first, "registryDescriptionMatch").get("matches", [])
        confirmed = [item for item in matches if item.get("scoreBand") == "match"]
        uncertain = [item for item in matches if item.get("scoreBand") == "uncertain"]
        return self._finish(
            f"符合描述的先验库库项数量为 {len(confirmed)}",
            [],
            f"目标描述：{description}；按库项去重后确定匹配 {len(confirmed)} 个",
            "sufficient" if not uncertain else "uncertain",
            extra={"description": description, "registryMatches": confirmed, "uncertainRegistryMatches": uncertain, "registryReferenceIds": catalog.get("registryReferenceIds", [])},
        )

    def _answer_cross_reference(self) -> dict[str, Any]:
        """跨记忆问题：先按自然语言条件筛轨迹，再与先验库建立对应关系。"""
        description = (self.meta.get("description") or "").strip()
        time_range = self.meta.get("timeRange")
        if description:
            page_size = max(1, int(self.config["pipeline"]["agent"].get("retrieval_page_size", 60)))
            offset, has_more = 0, True
            track_map: dict[str, dict[str, Any]] = {}
            confirmed_map: dict[str, dict[str, Any]] = {}
            while has_more and len(self.rounds) < max(1, self.max_rounds - 1):
                page_number = offset // page_size + 1
                track_call = f"crossTracks{page_number}"
                frame_call = f"crossFrames{page_number}"
                match_call = f"crossTextMatch{page_number}"
                current = self._round(
                    f"按描述筛选第 {page_number} 页轨迹",
                    [
                        {"id": track_call, "tool": "getTrack", "arguments": {"timeRange": time_range, "offset": offset, "limit": page_size}},
                        {"id": frame_call, "tool": "getFrames", "arguments": {"trackIds": {"$ref": f"{track_call}.trackIds"}}},
                        {"id": match_call, "tool": "matchText", "arguments": {"description": description, "galleryImages": {"$ref": f"{frame_call}.keyframes"}, "topK": 10}},
                    ],
                    "replan",
                    "先得到描述命中轨迹，再与先验库建立对应关系",
                )
                tracks_result = self._result(current, track_call)
                matches = self._result(current, match_call).get("matches", [])
                page_tracks = tracks_result.get("tracks", [])
                track_map.update({str(item["trackId"]): item for item in page_tracks})
                for item in matches:
                    track_id = str(item.get("matchedTrackId") or "")
                    if track_id not in track_map or item.get("scoreBand") != "match":
                        continue
                    previous = confirmed_map.get(track_id)
                    if previous is None or float(item.get("embeddingScore") or 0) > float(previous.get("embeddingScore") or 0):
                        confirmed_map[track_id] = self._with_match(track_map[track_id], item)
                has_more = bool(tracks_result.get("hasMore"))
                next_offset = tracks_result.get("nextOffset")
                if not has_more or next_offset is None:
                    break
                offset = int(next_offset)
            candidate_tracks = list(confirmed_map.values())
            if not candidate_tracks:
                return self._finish("未建立可靠对应关系", [], f"描述“{description}”未命中可检索轨迹", "sufficient")
        else:
            first = self._round(
                "读取视频轨迹作为跨记忆候选",
                [{"id": "crossTracks", "tool": "getTrack", "arguments": {"timeRange": time_range, "limit": 60}}],
                "replan",
                "先取时间范围内轨迹，再匹配先验库",
            )
            candidate_tracks = self._result(first, "crossTracks").get("tracks", [])
            if not candidate_tracks:
                return self._finish("证据不足", [], "查询范围内没有可比较的轨迹", "uncertain")

        if len(self.rounds) >= self.max_rounds:
            return self._finish("无法确认", candidate_tracks, "已筛出候选轨迹，但没有剩余轮次完成先验库对应", "uncertain", display={"tracks": candidate_tracks, "includeClips": True})

        second = self._round(
            "将候选轨迹与先验库参考图建立对应关系",
            [
                {"id": "crossCandidateFrames", "tool": "getFrames", "arguments": {"trackIds": [item["trackId"] for item in candidate_tracks]}},
                {"id": "crossRegistry", "tool": "listRegistry", "arguments": {}},
                {"id": "crossImageMatch", "tool": "matchImage", "arguments": {"queryImages": {"$ref": "crossCandidateFrames.keyframes"}, "galleryImages": {"$ref": "crossRegistry.registryReferences"}, "topK": 3}},
            ],
            "sufficient",
            "返回可解释的轨迹-库项对应关系",
        )
        registry = self._result(second, "crossRegistry")
        if not registry.get("registryReferences"):
            return self._finish("证据不足", candidate_tracks, "先验库缺少可检索参考图", "uncertain", display={"tracks": candidate_tracks, "includeClips": True})
        matches = self._result(second, "crossImageMatch").get("matches", [])
        track_map = {str(item["trackId"]): item for item in candidate_tracks}
        linked = []
        for item in matches:
            track_id = str(item.get("matchedTrackId") or item.get("queryTrackId") or "")
            if track_id not in track_map:
                track_id = str(item.get("queryTrackId") or item.get("matchedTrackId") or "")
            if track_id not in track_map:
                continue
            if item.get("scoreBand") not in {"match", "uncertain"}:
                continue
            linked.append(self._with_match(track_map[track_id], item))
        best: dict[str, dict[str, Any]] = {}
        for track in linked:
            track_id = str(track["trackId"])
            previous = best.get(track_id)
            if previous is None or float(track.get("embeddingScore") or 0) > float(previous.get("embeddingScore") or 0):
                best[track_id] = track
        linked = list(best.values())
        if linked:
            certain = all(item.get("scoreBand") == "match" for item in linked)
            return self._finish(
                "已建立跨记忆对应关系",
                linked,
                f"返回 {len(linked)} 条轨迹与先验库的对应候选" + (f"；筛选条件：{description}" if description else ""),
                "sufficient" if certain else "uncertain",
                extra={"description": description or None},
                display={"tracks": linked, "includeClips": True, "includeRegistry": True},
            )
        return self._finish(
            "未建立可靠对应关系",
            candidate_tracks[: self.display_limit],
            "候选轨迹与先验库参考图均未达到匹配阈值",
            "sufficient",
            extra={"description": description or None},
            display={"tracks": candidate_tracks[: self.display_limit], "includeClips": True},
        )

    def _answer_relation_description(self) -> dict[str, Any]:
        """描述 + 在库/库外：先按描述筛轨迹，再执行库关系认证。"""
        description = (self.meta.get("description") or self._description_target()).strip()
        want_in_registry = self.meta.get("registryRelation") != "out"
        page_size = max(1, int(self.config["pipeline"]["agent"].get("retrieval_page_size", 60)))
        offset, has_more = 0, True
        track_map: dict[str, dict[str, Any]] = {}
        confirmed_map: dict[str, dict[str, Any]] = {}
        while has_more and len(self.rounds) < max(1, self.max_rounds - 1):
            page_number = offset // page_size + 1
            track_call = f"relationTracks{page_number}"
            frame_call = f"relationFrames{page_number}"
            match_call = f"relationTextMatch{page_number}"
            current = self._round(
                f"按描述筛选第 {page_number} 页轨迹",
                [
                    {"id": track_call, "tool": "getTrack", "arguments": {"timeRange": self.meta.get("timeRange"), "offset": offset, "limit": page_size}},
                    {"id": frame_call, "tool": "getFrames", "arguments": {"trackIds": {"$ref": f"{track_call}.trackIds"}}},
                    {"id": match_call, "tool": "matchText", "arguments": {"description": description, "galleryImages": {"$ref": f"{frame_call}.keyframes"}, "topK": 20}},
                ],
                "replan",
                "先得到描述命中轨迹，再执行在库/库外认证",
            )
            tracks_result = self._result(current, track_call)
            matches = self._result(current, match_call).get("matches", [])
            page_tracks = tracks_result.get("tracks", [])
            track_map.update({str(item["trackId"]): item for item in page_tracks})
            for item in matches:
                track_id = str(item.get("matchedTrackId") or "")
                if track_id not in track_map or item.get("scoreBand") != "match":
                    continue
                previous = confirmed_map.get(track_id)
                if previous is None or float(item.get("embeddingScore") or 0) > float(previous.get("embeddingScore") or 0):
                    confirmed_map[track_id] = self._with_match(track_map[track_id], item)
            has_more = bool(tracks_result.get("hasMore"))
            next_offset = tracks_result.get("nextOffset")
            if not has_more or next_offset is None:
                break
            offset = int(next_offset)

        candidate_tracks = list(confirmed_map.values())
        if not candidate_tracks:
            relation = "在库" if want_in_registry else "库外"
            return self._finish(
                f"未发现符合描述的{relation}船舶",
                [],
                f"描述“{description}”未命中可检索轨迹",
                "sufficient",
                extra={"relationDescription": description, "registryRelation": "in" if want_in_registry else "out"},
            )

        original_get_track = self.tools.getTrack

        def limited_get_track(timeRange=None, hullNumber=None, finalMatchType=None, offset=0, limit=0):
            selected = candidate_tracks
            if hullNumber:
                selected = [item for item in selected if str(item.get("finalHullNumber") or "").upper() == str(hullNumber).upper()]
            start = max(0, int(offset or 0))
            page = max(0, min(200, int(limit or 0)))
            page_items = selected[start:start + page] if page else selected[start:]
            next_offset = start + len(page_items)
            return {
                "ok": True,
                "queryScope": list(timeRange) if timeRange else None,
                "trackIds": [item["trackId"] for item in page_items],
                "tracks": page_items,
                "totalTrackCount": len(selected),
                "returnedTrackCount": len(page_items),
                "offset": start,
                "limit": page,
                "hasMore": next_offset < len(selected),
                "nextOffset": next_offset if next_offset < len(selected) else None,
            }

        self.tools.getTrack = limited_get_track  # type: ignore[method-assign]
        try:
            result = self._answer_registry(want_in_registry)
        finally:
            self.tools.getTrack = original_get_track  # type: ignore[method-assign]
        result["relationDescription"] = description
        result["description"] = description
        if result.get("answerText"):
            result["answerText"] = f"{result['answerText']}；筛选条件：{description}"
        return result

    def _answer_hull(self) -> dict[str, Any]:
        hull = self.meta.get("hullNumber")
        if not hull:
            return self._finish("无法确认", [], "问题中未解析到舷号", "uncertain")
        first = self._round("查询轨迹记忆中的聚合舷号", [{"id": "directTracks", "tool": "getTrack", "arguments": {"hullNumber": hull, "timeRange": self.meta.get("timeRange")}}], "replan", "先检查轨迹级舷号是否稳定命中")
        direct = self._result(first, "directTracks")
        confirmed = [track for track in direct.get("tracks", []) if track["finalMatchType"] == "confirmed"]
        if confirmed:
            return self._finish("确认出现", confirmed, "轨迹级舷号聚合状态为 confirmed", "sufficient", display={"tracks": confirmed, "includeClips": True, "includeRegistry": self.meta.get("operation") in {"explain", "time"}})
        direct_candidates = [track for track in direct.get("tracks", []) if track["finalMatchType"] in {"candidate", "conflict"}]
        second = self._round("读取目标库项并匹配全视频正式关键帧", [
            {"id": "hullRegistry", "tool": "getRegistry", "arguments": {"hullNumber": hull}},
            {"id": "allTracks", "tool": "getTrack", "condition": {"ref": "hullRegistry.searchable", "equals": True}, "arguments": {"timeRange": self.meta.get("timeRange")}},
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
        description = self.meta.get("description") or self._description_target()
        page_size = 0
        offset, has_more = 0, True
        track_map: dict[str, dict[str, Any]] = {}
        uncertain_by_track: dict[str, dict[str, Any]] = {}
        missing_track_ids: set[str] = set()
        searched_count = 0
        while has_more and len(self.rounds) < max(1, self.max_rounds - 1):
            page_number = 1 if page_size == 0 else offset // page_size + 1
            track_call, frame_call, match_call = f"descriptionTracks{page_number}", f"descriptionFrames{page_number}", f"textMatch{page_number}"
            current = self._round("读取当前全部轨迹并执行描述检索", [
                {"id": track_call, "tool": "getTrack", "arguments": {"timeRange": self.meta.get("timeRange"), "offset": offset, "limit": page_size}},
                {"id": frame_call, "tool": "getFrames", "arguments": {"trackIds": {"$ref": f"{track_call}.trackIds"}}},
                {"id": match_call, "tool": "matchText", "arguments": {"description": description, "galleryImages": {"$ref": f"{frame_call}.keyframes"}, "topK": 6}},
            ], "replan", "当前页无充分证据且 hasMore=true 时继续读取下一页")
            tracks_result = self._result(current, track_call)
            frame_result = self._result(current, frame_call)
            matches = self._result(current, match_call).get("matches", [])
            page_tracks = tracks_result.get("tracks", [])
            searched_count += len(page_tracks)
            track_map.update({str(item["trackId"]): item for item in page_tracks})
            missing_track_ids.update(str(value) for value in frame_result.get("unsearchableTrackIds", []))
            matched = [item for item in matches if item.get("scoreBand") == "match"]
            if matched:
                tracks = [self._with_match(track_map[str(item["matchedTrackId"])], item) for item in matched[:self.display_limit] if str(item["matchedTrackId"]) in track_map]
                return self._finish("确认出现", tracks, f"读取当前全部 {searched_count} 条轨迹后找到达到匹配阈值的证据", "sufficient", extra={"searchedTrackCount": searched_count}, display={"tracks": tracks, "includeClips": True})
            for item in matches:
                if item.get("scoreBand") != "uncertain":
                    continue
                track_id = str(item.get("matchedTrackId"))
                previous = uncertain_by_track.get(track_id)
                if previous is None or float(item.get("embeddingScore") or 0) > float(previous.get("embeddingScore") or 0):
                    uncertain_by_track[track_id] = item
            has_more = bool(tracks_result.get("hasMore"))
            next_offset = tracks_result.get("nextOffset")
            if not has_more or next_offset is None:
                break
            offset = int(next_offset)

        uncertain = sorted(uncertain_by_track.values(), key=lambda item: float(item.get("embeddingScore") or 0), reverse=True)[:self.display_limit]
        if not uncertain:
            if has_more:
                return self._finish("无法确认", [], "达到检索轮次上限但仍有轨迹尚未读取", "uncertain", extra={"searchedTrackCount": searched_count, "hasMore": True})
            return self._finish("无法确认" if missing_track_ids else "未发现", [], "存在不可检索轨迹" if missing_track_ids else "全部已读取轨迹均低于排除阈值", "uncertain" if missing_track_ids else "sufficient", extra={"searchedTrackCount": searched_count})
        if len(self.rounds) >= self.max_rounds:
            candidates = [self._with_match(track_map[str(item["matchedTrackId"])], item) for item in uncertain if str(item["matchedTrackId"]) in track_map]
            return self._finish("无法确认", candidates, "已完成分页检索，但没有剩余轮次核验灰区证据", "uncertain", extra={"searchedTrackCount": searched_count, "hasMore": has_more}, display={"tracks": candidates, "includeClips": True})
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
            track_id = str(item["matchedTrackId"])
            track = self._with_match(track_map[track_id], item) if track_id in track_map else None
            if track and clip_result.get("shipSegmentId"):
                track["shipSegmentIds"] = [clip_result["shipSegmentId"]]
            if track and final_decision == "match":
                track["scoreBand"] = "verified"
                verified.append(track)
            elif track and final_decision == "uncertain":
                unresolved.append(track)
        if verified:
            return self._finish("确认出现", verified, "全量检索后的灰区视觉证据经模型核验后符合目标描述", "sufficient", extra={"searchedTrackCount": searched_count}, display={"tracks": verified, "includeClips": True})
        if unresolved or missing_track_ids or has_more:
            return self._finish("无法确认", unresolved, "灰区证据、不可检索轨迹或未读取页面仍存在不确定性", "uncertain", extra={"searchedTrackCount": searched_count, "hasMore": has_more}, display={"tracks": unresolved, "includeClips": True})
        return self._finish("未发现", [], "当前全部轨迹候选经视觉核验均不符合目标描述", "sufficient", extra={"searchedTrackCount": searched_count})

    def _answer_registry_description(self) -> dict[str, Any]:
        description = self.meta.get("description") or self._description_target()
        first = self._round(
            "判断先验库中是否存在符合描述的库项",
            [
                {"id": "registryCatalog", "tool": "listRegistry", "arguments": {}},
                {"id": "registryTextMatch", "tool": "matchText", "arguments": {"description": description, "galleryImages": {"$ref": "registryCatalog.registryReferences"}, "topK": 3}},
            ],
            "replan",
            "先做文本-库图特征匹配；高分直接确认，灰区再对库参考图做视觉核验",
        )
        catalog = self._result(first, "registryCatalog")
        match_result = self._result(first, "registryTextMatch")
        matches = self._enrich_registry_matches(match_result.get("matches", []), catalog)
        # 文本-图像匹配的中分段不可过度信任：分数刚过阈值或与第二名接近时，降为灰区走视觉核验
        text_match = float(self.config["pipeline"]["retrieval"].get("text_match", 0.62))
        for item in matches:
            score = float(item.get("embeddingScore") or 0)
            rank_gap = item.get("rankGap")
            weak_match = item.get("scoreBand") == "match" and (
                score < text_match + 0.05
                or (isinstance(rank_gap, (int, float)) and float(rank_gap) < 0.03 and score < text_match + 0.10)
            )
            if weak_match:
                item["scoreBand"] = "uncertain"
                item["weakMatch"] = True
        confirmed = [item for item in matches if item.get("scoreBand") == "match"]
        uncertain = [item for item in matches if item.get("scoreBand") == "uncertain"]
        rejected = [item for item in matches if item.get("scoreBand") == "mismatch"]

        if confirmed:
            refs = []
            for item in confirmed:
                refs.extend(item.get("registryReferenceIds") or item.get("matchedRegistryReferenceIds") or [])
            if refs and len(self.rounds) < self.max_rounds:
                self._round(
                    "展示命中的先验库参考图证据",
                    [{"id": "registryShow", "tool": "showEvidence", "arguments": {"registryReferenceIds": list(dict.fromkeys(refs))[:6]}}],
                    "sufficient",
                    "返回已确认库项及参考图",
                )
            hit_text = self._format_registry_hits(confirmed)
            return self._finish(
                "先验库中存在符合描述的库项",
                [],
                f"目标描述：{description}；命中 {len(confirmed)} 个库项：{hit_text}",
                "sufficient",
                extra={
                    "registryMatches": confirmed,
                    "registryItems": [{
                        "registryId": item.get("registryId"),
                        "hullNumber": item.get("hullNumber"),
                        "description": item.get("description"),
                        "embeddingScore": item.get("embeddingScore"),
                        "scoreBand": item.get("scoreBand"),
                        "registryReferenceIds": item.get("registryReferenceIds") or [],
                    } for item in confirmed],
                    "registryDescription": description,
                    "registryReferenceIds": list(dict.fromkeys(refs)),
                    "hitHullNumbers": [item.get("hullNumber") for item in confirmed if item.get("hullNumber")],
                },
            )

        if uncertain and len(self.rounds) < self.max_rounds:
            calls = []
            for index, item in enumerate(uncertain[: self.display_limit]):
                ref_ids = item.get("registryReferenceIds") or item.get("matchedRegistryReferenceIds") or []
                if not ref_ids:
                    continue
                calls.append({
                    "id": f"verifyRegistryDesc{index}",
                    "tool": "verifyTarget",
                    "arguments": {"description": description, "registryReferenceIds": ref_ids[:3]},
                })
            if calls:
                second = self._round(
                    "对先验库描述匹配灰区做视觉核验",
                    calls,
                    "uncertain",
                    "用库参考图核对是否真正符合文字描述，避免仅因相似度灰区直接放弃",
                )
                verified, still_uncertain = [], []
                for index, item in enumerate(uncertain[: self.display_limit]):
                    ref_ids = item.get("registryReferenceIds") or item.get("matchedRegistryReferenceIds") or []
                    if not ref_ids:
                        still_uncertain.append(item)
                        continue
                    decision = self._result(second, f"verifyRegistryDesc{index}").get("decision", "uncertain")
                    enriched = dict(item)
                    enriched["verifyDecision"] = decision
                    if decision == "match":
                        enriched["scoreBand"] = "verified"
                        verified.append(enriched)
                    elif decision == "mismatch":
                        rejected.append(enriched)
                    else:
                        still_uncertain.append(enriched)
                if verified:
                    refs = []
                    for item in verified:
                        refs.extend(item.get("registryReferenceIds") or item.get("matchedRegistryReferenceIds") or [])
                    if refs and len(self.rounds) < self.max_rounds:
                        self._round(
                            "展示核验通过的先验库参考图",
                            [{"id": "registryVerifiedShow", "tool": "showEvidence", "arguments": {"registryReferenceIds": list(dict.fromkeys(refs))[:6]}}],
                            "sufficient",
                            "返回视觉核验通过的库项",
                        )
                    hit_text = self._format_registry_hits(verified)
                    return self._finish(
                        "先验库中存在符合描述的库项",
                        [],
                        f"目标描述：{description}；灰区库项经视觉核验确认：{hit_text}",
                        "sufficient",
                        extra={
                            "registryMatches": verified,
                            "registryItems": [{
                                "registryId": item.get("registryId"),
                                "hullNumber": item.get("hullNumber"),
                                "description": item.get("description"),
                                "embeddingScore": item.get("embeddingScore"),
                                "scoreBand": item.get("scoreBand"),
                                "registryReferenceIds": item.get("registryReferenceIds") or [],
                            } for item in verified],
                            "registryDescription": description,
                            "registryReferenceIds": list(dict.fromkeys(refs)),
                            "hitHullNumbers": [item.get("hullNumber") for item in verified if item.get("hullNumber")],
                            "rejectedRegistryMatches": rejected,
                        },
                    )
                if still_uncertain:
                    refs = []
                    for item in still_uncertain:
                        refs.extend(item.get("registryReferenceIds") or item.get("matchedRegistryReferenceIds") or [])
                    hit_text = self._format_registry_hits(still_uncertain)
                    return self._finish(
                        "无法确认",
                        [],
                        f"目标描述：{description}；相似库项仍无法确认：{hit_text}",
                        "uncertain",
                        extra={
                            "registryMatches": still_uncertain,
                            "registryItems": [{
                                "registryId": item.get("registryId"),
                                "hullNumber": item.get("hullNumber"),
                                "description": item.get("description"),
                                "embeddingScore": item.get("embeddingScore"),
                                "scoreBand": item.get("scoreBand"),
                                "registryReferenceIds": item.get("registryReferenceIds") or [],
                            } for item in still_uncertain],
                            "registryDescription": description,
                            "registryReferenceIds": list(dict.fromkeys(refs)),
                            "hitHullNumbers": [item.get("hullNumber") for item in still_uncertain if item.get("hullNumber")],
                            "rejectedRegistryMatches": rejected,
                        },
                    )

        if uncertain:
            refs = []
            for item in uncertain:
                refs.extend(item.get("registryReferenceIds") or item.get("matchedRegistryReferenceIds") or [])
            hit_text = self._format_registry_hits(uncertain)
            return self._finish(
                "无法确认",
                [],
                f"目标描述：{description}；存在灰区匹配但没有剩余轮次完成视觉核验：{hit_text}",
                "uncertain",
                extra={
                    "registryMatches": uncertain,
                    "registryItems": [{
                        "registryId": item.get("registryId"),
                        "hullNumber": item.get("hullNumber"),
                        "description": item.get("description"),
                        "embeddingScore": item.get("embeddingScore"),
                        "scoreBand": item.get("scoreBand"),
                        "registryReferenceIds": item.get("registryReferenceIds") or [],
                    } for item in uncertain],
                    "registryDescription": description,
                    "registryReferenceIds": list(dict.fromkeys(refs)),
                    "hitHullNumbers": [item.get("hullNumber") for item in uncertain if item.get("hullNumber")],
                },
            )
        return self._finish(
            "先验库中未找到符合描述的库项",
            [],
            f"目标描述：{description}；特征匹配均未达到接受阈值",
            "sufficient",
            extra={"registryMatches": [], "registryItems": [], "registryDescription": description, "rejectedRegistryMatches": rejected, "hitHullNumbers": []},
        )

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
        # 范围查询返回全部命中，不被 topk/display_limit 截断；topk 只用于单轨迹的库项候选数
        result_tracks = in_tracks if want_in_registry else out_tracks
        hit_count = len(result_tracks)
        unresolved_count = len(unresolved)
        if result_tracks:
            conclusion = f"{'在库' if want_in_registry else '未在库'}船舶 {hit_count} 条"
            if unresolved_count:
                conclusion = f"部分确认：{conclusion}"
        else:
            conclusion = "无法确认" if unresolved_count else f"未发现{'在库' if want_in_registry else '未在库'}船舶"
        reason = (
            f"范围内轨迹 {len(tracks)} 条；舷号精确在库 {len(exact_tracks)} 条；"
            f"图像确认在库 {len(image_in)} 条、未在库 {len(image_out)} 条；"
            f"仍不确定 {unresolved_count} 条。"
            "topk 仅限制单条轨迹匹配的库项候选数，不限制范围命中返回数量。"
        )
        extra = {
            "unresolvedTracks": unresolved,
            "unsearchableTrackIds": frame_result.get("unsearchableTrackIds", []),
            "exactInRegistryTrackIds": [item["trackId"] for item in exact_tracks],
            "inRegistryCount": len(in_tracks),
            "outOfRegistryCount": len(out_tracks),
            "hitCount": hit_count,
            "totalTrackCount": len(tracks),
        }
        return self._finish(
            conclusion,
            result_tracks,
            reason,
            "uncertain" if unresolved_count else "sufficient",
            extra=extra,
            display={"tracks": result_tracks, "includeClips": True, "includeRegistry": True},
        )

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
        representatives = self._deduplication_representatives(tracks, dedup)
        conclusion = f"高阈值计数 {dedup.get('highThresholdShipCount', len(tracks))}，低阈值计数 {dedup.get('lowThresholdShipCount', len(tracks))}"
        uncertainty = "sufficient" if dedup.get("countStability") == "stable" else "uncertain"
        reason = f"计数状态为 {dedup.get('countStability', 'unknown')}，两种结果表示阈值敏感性"
        return self._finish(conclusion, representatives, reason, uncertainty, extra=extra, display={"tracks": representatives, "includeClips": True})

    def _answer_autonomous(self) -> dict[str, Any]:
        """自主规划模式：PlanAgent 决定工具，ReflectAgent 根据证据决定是否继续。"""
        final_state = "uncertain"
        final_reason = "自主规划未形成充分证据"
        last_round: dict[str, Any] | None = None
        total_round_limit = self.max_rounds + max(0, self.acceptance_recovery_rounds)
        while len(self.rounds) < total_round_limit:
            history = self._autonomous_history()
            forced_plan = None
            if len(self.rounds) >= self.max_rounds:
                acceptance = build_acceptance_progress(self.meta, self.working_scope)
                self.working_scope["acceptance"] = acceptance
                recovery_calls = self._acceptance_fallback_calls(acceptance)
                if acceptance.get("acceptanceSatisfied") or not recovery_calls:
                    break
                forced_plan = {
                    "goal": "完成尚未满足的验收步骤",
                    "intent": self.meta,
                    "calls": recovery_calls,
                    "proposedState": "replan",
                    "reason": acceptance.get("nextAction") or "执行验收补偿工具",
                    "evidenceGap": "、".join(acceptance.get("pendingRequirementLabels") or []),
                    "answerHint": "",
                    "planMode": "autonomous",
                    "planRepair": "常规轮次已用完，控制器按初始验收目标补充执行必要工具",
                }
            round_result = self._round_autonomous(history, forced_plan=forced_plan)
            last_round = round_result
            state = str(round_result.get("reflection", {}).get("state") or "uncertain")
            reason = str(round_result.get("reflection", {}).get("reason") or final_reason)
            final_state, final_reason = state, reason
            if state == "replan":
                continue
            break
        if final_state == "replan" and len(self.rounds) >= total_round_limit:
            final_state = "uncertain"
            final_reason = f"已达到常规轮次 {self.max_rounds} 与验收补偿轮次 {self.acceptance_recovery_rounds}，仍未满足验收标准"
        return self._finish_autonomous(last_round, final_state, final_reason)

    def _acceptance_fallback_calls(self, acceptance: dict[str, Any]) -> list[dict[str, Any]]:
        """根据 ReflectAgent 的验收缺口生成最小下一步计划。"""
        pending = list(acceptance.get("pendingRequirements") or [])
        if not pending:
            return []
        requirement = str(pending[0])
        if requirement == "hull_lookup":
            hull_number = self.meta.get("hullNumber")
            return [
                {"id": "directHullTracks", "tool": "getTrack", "arguments": {"hullNumber": hull_number, "timeRange": self.meta.get("timeRange")}},
                {"id": "targetHullRegistry", "tool": "getRegistry", "arguments": {"hullNumber": hull_number}},
            ]
        if requirement in {"complete_track_scope", "hull_track_scope"}:
            return [{
                "id": "tracksSnapshot",
                "tool": "getTrack",
                "arguments": {
                    "timeRange": self.meta.get("timeRange"),
                    "offset": 0,
                    "limit": 0,
                },
            }]
        if requirement == "exact_hull_classification":
            calls = [{
                "id": "exactHull",
                "tool": "matchHull",
                "arguments": {"hullNumberArray": {"$ref": "acceptance.confirmedHullNumbers"}},
            }]
            if not acceptance.get("registryLoaded"):
                calls.append({"id": "registryCatalog", "tool": "listRegistry", "arguments": {}})
            return calls
        if requirement == "registry_catalog":
            return [{"id": "registryCatalog", "tool": "listRegistry", "arguments": {}}]
        if requirement == "hull_keyframe_evidence":
            registry_id = next((
                str(key) for key, value in reversed(list(self.working_scope.items()))
                if isinstance(value, dict)
                and normalize_hull_number(value.get("hullNumber")) == normalize_hull_number(self.meta.get("hullNumber"))
                and value.get("registryReferences")
            ), None)
            calls = [{
                "id": "hullSearchFrames",
                "tool": "getFrames",
                "arguments": {"trackIds": {"$ref": "acceptance.hullSearchTrackIds"}},
            }]
            if registry_id:
                calls.append({
                    "id": "hullImageMatch",
                    "tool": "matchImage",
                    "arguments": {
                        "queryImages": {"$ref": f"{registry_id}.registryReferences"},
                        "galleryImages": {"$ref": "hullSearchFrames.keyframes"},
                        "topK": 3,
                    },
                })
            return calls
        if requirement == "hull_image_classification":
            frame_id = next((
                str(key) for key, value in reversed(list(self.working_scope.items()))
                if isinstance(value, dict) and value.get("keyframes")
            ), None)
            registry_id = next((
                str(key) for key, value in reversed(list(self.working_scope.items()))
                if isinstance(value, dict)
                and normalize_hull_number(value.get("hullNumber")) == normalize_hull_number(self.meta.get("hullNumber"))
                and value.get("registryReferences")
            ), None)
            if frame_id and registry_id:
                return [{
                    "id": "hullImageMatch",
                    "tool": "matchImage",
                    "arguments": {
                        "queryImages": {"$ref": f"{registry_id}.registryReferences"},
                        "galleryImages": {"$ref": f"{frame_id}.keyframes"},
                        "topK": 3,
                    },
                }]
        if requirement == "keyframe_evidence":
            calls = [{
                "id": "remainingFrames",
                "tool": "getFrames",
                "arguments": {"trackIds": {"$ref": "acceptance.remainingTrackIds"}},
            }]
            registry_id = next((
                str(key) for key, value in reversed(list(self.working_scope.items()))
                if isinstance(value, dict) and "registryReferences" in value and "hullNumber" not in value
            ), None)
            if registry_id:
                calls.append({
                    "id": "registryImageMatch",
                    "tool": "matchImage",
                    "arguments": {
                        "queryImages": {"$ref": "remainingFrames.keyframes"},
                        "galleryImages": {"$ref": f"{registry_id}.registryReferences"},
                        "topK": 3,
                    },
                })
            return calls
        if requirement == "deduplicated_count":
            track_result_id = next((
                str(key)
                for key, value in reversed(list(self.working_scope.items()))
                if isinstance(value, dict)
                and isinstance(value.get("tracks"), list)
                and "trackIds" in value
                and not normalize_hull_number(value.get("queryHullNumber"))
                and not value.get("queryFinalMatchType")
            ), None)
            if not track_result_id:
                return []
            return [
                {
                    "id": "countFrames",
                    "tool": "getFrames",
                    "arguments": {"trackIds": {"$ref": f"{track_result_id}.trackIds"}},
                },
                {
                    "id": "deduplicatedCount",
                    "tool": "dedupTracks",
                    "arguments": {
                        "tracks": {"$ref": f"{track_result_id}.tracks"},
                        "keyframesByTrack": {"$ref": "countFrames.keyframesByTrack"},
                    },
                },
            ]
        if requirement == "target_match":
            description = self.meta.get("description") or self._description_target()
            if self.meta.get("targetScope") == "registry":
                return [
                    {"id": "registryCatalog", "tool": "listRegistry", "arguments": {}},
                    {
                        "id": "descriptionMatch",
                        "tool": "matchText",
                        "arguments": {
                            "description": description,
                            "galleryImages": {"$ref": "registryCatalog.registryReferences"},
                            "topK": self.query_top_k,
                        },
                    },
                ]
            track_ids = list(acceptance.get("trackIds") or [])
            if not track_ids:
                return []
            return [
                {"id": "descriptionFrames", "tool": "getFrames", "arguments": {"trackIds": track_ids}},
                {
                    "id": "descriptionMatch",
                    "tool": "matchText",
                    "arguments": {
                        "description": description,
                        "galleryImages": {"$ref": "descriptionFrames.keyframes"},
                        "topK": self.query_top_k,
                    },
                },
            ]
        if requirement == "registry_image_classification":
            frame_id = next((
                str(key) for key, value in reversed(list(self.working_scope.items()))
                if isinstance(value, dict) and value.get("keyframes")
            ), None)
            registry_id = next((
                str(key) for key, value in reversed(list(self.working_scope.items()))
                if isinstance(value, dict) and "registryReferences" in value and "hullNumber" not in value
            ), None)
            if frame_id and registry_id:
                return [{
                    "id": "registryImageMatch",
                    "tool": "matchImage",
                    "arguments": {
                        "queryImages": {"$ref": f"{frame_id}.keyframes"},
                        "galleryImages": {"$ref": f"{registry_id}.registryReferences"},
                        "topK": 3,
                    },
                }]
        return []

    def _align_plan_with_acceptance(
        self,
        plan: dict[str, Any],
        acceptance: dict[str, Any],
    ) -> dict[str, Any]:
        """确保 PlanAgent 落实 ReflectAgent 根据初始验收目标给出的下一步。"""
        if acceptance.get("acceptanceSatisfied"):
            return plan
        pending = list(acceptance.get("pendingRequirements") or [])
        expected_tools = {
            "complete_track_scope": {"getTrack"},
            "hull_lookup": {"getTrack", "getRegistry"},
            "hull_track_scope": {"getTrack"},
            "hull_keyframe_evidence": {"getFrames"},
            "hull_image_classification": {"matchImage"},
            "exact_hull_classification": {"matchHull"},
            "registry_catalog": {"listRegistry"},
            "keyframe_evidence": {"getFrames"},
            "registry_image_classification": {"matchImage"},
            "deduplicated_count": {"dedupTracks"},
            "target_match": {"matchText"},
            "gray_verification": {"verifyTarget", "getClip"},
        }.get(str(pending[0]) if pending else "", set())
        current_calls = list(plan.get("calls") or [])
        current_tools = {str(call.get("tool")) for call in current_calls}
        requirement = str(pending[0]) if pending else ""
        parameter_bound_requirements = {
            "complete_track_scope",
            "hull_lookup",
            "hull_track_scope",
            "hull_keyframe_evidence",
            "hull_image_classification",
            "exact_hull_classification",
            "registry_catalog",
            "keyframe_evidence",
            "registry_image_classification",
            "deduplicated_count",
            "target_match",
        }
        if requirement in parameter_bound_requirements:
            fallback_calls = self._acceptance_fallback_calls(acceptance)
            if fallback_calls:
                repaired = dict(plan)
                repaired["calls"] = fallback_calls
                repaired["proposedState"] = "replan"
                repaired["reason"] = acceptance.get("nextAction") or plan.get("reason")
                repaired["evidenceGap"] = "、".join(
                    str(value) for value in acceptance.get("pendingRequirementLabels") or pending
                )
                repaired["planRepair"] = "已按当前验收步骤重新绑定工具参数和依赖结果"
                return repaired
        plan_covers_requirement = bool(expected_tools and current_tools.intersection(expected_tools))
        if requirement == "hull_lookup":
            plan_covers_requirement = expected_tools.issubset(current_tools)
        if expected_tools and plan_covers_requirement:
            repaired = dict(plan)
            if requirement == "exact_hull_classification" and not acceptance.get("registryLoaded") and "listRegistry" not in current_tools:
                repaired["calls"] = current_calls + [{"id": "registryCatalog", "tool": "listRegistry", "arguments": {}}]
                repaired["planRepair"] = "按验收目标补充读取完整先验库"
                return repaired
            if requirement == "hull_keyframe_evidence" and "matchImage" not in current_tools:
                frame_call = next((call for call in current_calls if call.get("tool") == "getFrames"), None)
                registry_id = next((
                    str(key) for key, value in reversed(list(self.working_scope.items()))
                    if isinstance(value, dict)
                    and normalize_hull_number(value.get("hullNumber")) == normalize_hull_number(self.meta.get("hullNumber"))
                    and value.get("registryReferences")
                ), None)
                if frame_call and registry_id:
                    repaired["calls"] = current_calls + [{
                        "id": "hullImageMatch",
                        "tool": "matchImage",
                        "arguments": {
                            "queryImages": {"$ref": f"{registry_id}.registryReferences"},
                            "galleryImages": {"$ref": f"{frame_call['id']}.keyframes"},
                            "topK": 3,
                        },
                    }]
                    repaired["planRepair"] = "按舷号验收目标补充目标库图与关键帧匹配"
                    return repaired
            if requirement == "keyframe_evidence" and "matchImage" not in current_tools:
                frame_call = next((call for call in current_calls if call.get("tool") == "getFrames"), None)
                registry_id = next((
                    str(key) for key, value in reversed(list(self.working_scope.items()))
                    if isinstance(value, dict) and "registryReferences" in value and "hullNumber" not in value
                ), None)
                if frame_call and registry_id:
                    repaired["calls"] = current_calls + [{
                        "id": "registryImageMatch",
                        "tool": "matchImage",
                        "arguments": {
                            "queryImages": {"$ref": f"{frame_call['id']}.keyframes"},
                            "galleryImages": {"$ref": f"{registry_id}.registryReferences"},
                            "topK": 3,
                        },
                    }]
                    repaired["planRepair"] = "按验收目标补充剩余轨迹的库图匹配"
                    return repaired
            return plan
        fallback_calls = self._acceptance_fallback_calls(acceptance)
        if not fallback_calls:
            return plan
        repaired = dict(plan)
        repaired["calls"] = fallback_calls
        repaired["proposedState"] = "replan"
        repaired["reason"] = acceptance.get("nextAction") or plan.get("reason")
        repaired["evidenceGap"] = "、".join(
            str(value) for value in acceptance.get("pendingRequirementLabels") or pending
        )
        repaired["planRepair"] = "原计划未落实初始验收缺口，已按 ReflectAgent 下一步调整"
        return repaired

    def _round_autonomous(self, history: list[dict[str, Any]], forced_plan: dict[str, Any] | None = None) -> dict[str, Any]:
        total_round_limit = self.max_rounds + max(0, self.acceptance_recovery_rounds)
        if len(self.rounds) >= total_round_limit:
            raise RuntimeError("达到最大问答轮次")
        round_number = len(self.rounds) + 1
        acceptance = build_acceptance_progress(self.meta, self.working_scope)
        self.working_scope["acceptance"] = acceptance
        round_meta = dict(self.meta)
        pending_text = "、".join(
            str(value)
            for value in acceptance.get("pendingRequirementLabels")
            or acceptance.get("pendingRequirements")
            or []
        )
        focus = acceptance.get("nextAction") or self.meta.get("nextAgentFocus")
        round_meta["nextAgentFocus"] = (
            f"{focus} 当前验收剩余：{pending_text}" if focus and pending_text else focus
        )
        round_meta["acceptanceProgress"] = compact_acceptance(acceptance)
        self._emit("agent_start", "PlanAgent", "自主规划本轮工具调用", round=round_number, role="planner", planMode="autonomous")
        if forced_plan is None:
            plan = self.planner.decide_tools(
                self.question,
                round_meta,
                history=history,
                memory_scope=self.working_scope,
                on_delta=lambda delta: self._emit("agent_delta", "PlanAgent", "", round=round_number, role="planner", delta=delta),
            )
            plan = self._align_plan_with_acceptance(plan, acceptance)
        else:
            plan = forced_plan
        self._apply_query_top_k(plan)
        calls = plan.get("calls") or []
        if not calls:
            fallback_calls = self._acceptance_fallback_calls(acceptance)
            if fallback_calls:
                plan = dict(plan)
                plan["calls"] = fallback_calls
                plan["proposedState"] = "replan"
                plan["reason"] = acceptance.get("nextAction") or plan.get("reason") or "按验收条件继续执行工具"
                plan["evidenceGap"] = "、".join(acceptance.get("pendingRequirementLabels") or [])
                repair = str(plan.get("planRepair") or "").strip()
                plan["planRepair"] = f"{repair}；模型计划不可执行，已按验收条件生成工具调用".strip("；")
                self._apply_query_top_k(plan)
                calls = plan["calls"]
        public_plan = self._public_plan(plan, calls)
        public_plan["planMode"] = "autonomous"
        self._emit(
            "agent_end",
            "PlanAgent",
            f"计划已修正：{plan['planRepair']}" if plan.get("planRepair") else "自主规划完成",
            round=round_number,
            role="planner",
            calls=public_plan["calls"],
            modelSummary=plan.get("modelPlan"),
            fallback=plan.get("modelFallback"),
            planRepair=plan.get("planRepair"),
            planMode="autonomous",
            planOnly=True,
            executionOwner="ObserveAgent",
            planBlueprint=self.plan_blueprint,
        )

        self._emit("agent_start", "ObserveAgent", "执行 PlanAgent 的工具计划", round=round_number, role="observer", executionOwner="ObserveAgent")
        observed = self.observer.execute(
            plan,
            context=self.working_scope,
            on_delta=lambda delta: self._emit("agent_delta", "ObserveAgent", "", round=round_number, role="observer", delta=delta),
            on_tool_event=lambda event: self._emit_observer_tool_event(round_number, event),
        )
        # 累积结果供后续 $ref，并重新计算初始验收目标的完成进度
        self.working_scope.update(observed.get("scope") or {})
        acceptance = build_acceptance_progress(self.meta, self.working_scope)
        self.working_scope["acceptance"] = acceptance
        observed["summary"]["acceptanceProgress"] = compact_acceptance(acceptance)
        self._emit(
            "agent_end",
            "ObserveAgent",
            "工具执行完成",
            round=round_number,
            role="observer",
            calls=observed["summary"].get("calls", []),
            modelSummary=observed["summary"].get("modelObservation"),
            fallback=observed["summary"].get("modelFallback"),
            executionOwner="ObserveAgent",
        )

        default_state = str(plan.get("proposedState") or "uncertain")
        reason = str(plan.get("reason") or "根据观察结果判定证据是否充分")
        evidence_gap = plan.get("evidenceGap")
        self._emit("agent_start", "ReflectAgent", "正在审计自主规划证据充分性", round=round_number, role="reflector")
        reflection = self.reflector.review(
            default_state,
            reason,
            observed["summary"],
            evidence_gap,
            lambda delta: self._emit("agent_delta", "ReflectAgent", "", round=round_number, role="reflector", delta=delta),
            autonomous=True,
            expected_outcome=self.meta.get("expectedOutcome"),
            success_criteria=self.meta.get("successCriteria"),
            next_agent_focus=acceptance.get("nextAction") or self.meta.get("nextAgentFocus"),
            previous_rounds=history,
            acceptance_context=compact_acceptance(acceptance),
        )
        # 首轮无有效观察时不能确认；后续空调用可基于历史证据正常结束。
        if reflection.get("state") == "sufficient" and not history and not self._has_successful_observation(observed):
            reflection["state"] = "uncertain"
            reflection["reason"] = "本轮未获得有效工具结果，暂不能确认"
        # 连续空计划 / 计划校验失败时停止 replan，避免多轮空转
        empty_plan = not (plan.get("calls") or [])
        if empty_plan and reflection.get("state") == "replan":
            reflection["state"] = "uncertain"
            reflection["reason"] = plan.get("planRepair") or plan.get("reason") or "未能形成可执行工具计划"
            reflection["evidenceGap"] = plan.get("evidenceGap") or "缺少可执行工具计划"
        consecutive_empty = 0
        for item in reversed(self.rounds):
            if (item.get("plan") or {}).get("calls"):
                break
            consecutive_empty += 1
        if empty_plan:
            consecutive_empty += 1
        if consecutive_empty >= 2 and reflection.get("state") == "replan":
            reflection["state"] = "uncertain"
            reflection["reason"] = "连续多轮未能形成可执行工具计划，已停止"
        self._emit(
            "agent_end",
            "ReflectAgent",
            reflection.get("reason", reason),
            round=round_number,
            role="reflector",
            state=reflection.get("state"),
            evidenceGap=reflection.get("evidenceGap"),
            modelSummary=reflection.get("modelReflection"),
            fallback=reflection.get("modelFallback"),
            nextAction=reflection.get("nextAction"),
            acceptanceGoal=acceptance.get("expectedOutcome"),
            acceptanceSatisfied=acceptance.get("acceptanceSatisfied"),
            pendingRequirements=acceptance.get("pendingRequirements") or [],
        )
        round_id = f"round-{uuid.uuid4().hex[:12]}"
        self.repository.add_round(round_id, self.session_id, public_plan, reflection)
        self._store_observations(round_id, observed, round_number)
        record = {
            "roundId": round_id,
            "plan": public_plan,
            "observed": observed["summary"],
            "reflection": reflection,
            "scope": observed.get("scope") or {},
            "acceptance": compact_acceptance(acceptance),
            "answerHint": plan.get("answerHint") or "",
        }
        self.rounds.append(record)
        return record

    def _autonomous_history(self) -> list[dict[str, Any]]:
        history = []
        for item in self.rounds:
            history.append(
                {
                    "roundId": item.get("roundId"),
                    "goal": (item.get("plan") or {}).get("goal"),
                    "calls": [
                        {"id": call.get("id"), "tool": call.get("tool"), "arguments": call.get("arguments")}
                        for call in ((item.get("plan") or {}).get("calls") or [])
                    ],
                    "observation": item.get("observed"),
                    "state": (item.get("reflection") or {}).get("state"),
                    "reason": (item.get("reflection") or {}).get("reason"),
                    "evidenceGap": (item.get("reflection") or {}).get("evidenceGap"),
                    "nextAction": (item.get("reflection") or {}).get("nextAction"),
                    "acceptance": item.get("acceptance") or {},
                }
            )
        return history

    @staticmethod
    def _has_successful_observation(observed: dict[str, Any]) -> bool:
        for item in observed.get("observations") or []:
            if item.get("skipped"):
                continue
            result = item.get("result") or {}
            if isinstance(result, dict) and result.get("ok") is False:
                continue
            return True
        return False

    def _finish_autonomous(self, last_round: dict[str, Any] | None, state: str, reason: str) -> dict[str, Any]:
        tracks = self._collect_tracks_from_scope()
        matches = self._collect_matches_from_scope()
        registry_items = self._collect_registry_from_scope()
        count_value = self._collect_count_from_scope()
        answer_hint = ""
        if last_round:
            answer_hint = str(last_round.get("answerHint") or "").strip()
            plan = last_round.get("plan") or {}
            answer_hint = answer_hint or str(plan.get("answerHint") or "").strip()

        relation = str(self.meta.get("registryRelation") or "any")
        if relation in {"in", "out"} and self.meta.get("targetScope") != "registry":
            acceptance = build_acceptance_progress(self.meta, self.working_scope)
            selected_ids = set(
                (
                    acceptance.get("inRegistryTrackIds")
                    if relation == "in"
                    else acceptance.get("outOfRegistryTrackIds")
                )
                or []
            )
            track_map = {str(item.get("trackId")): item for item in tracks if item.get("trackId") is not None}
            match_map: dict[str, dict[str, Any]] = {}
            for match in matches:
                track_id = str(match.get("matchedTrackId") or match.get("trackId") or "")
                if not track_id:
                    continue
                current = match_map.get(track_id)
                if current is None or float(match.get("embeddingScore") or -1) > float(current.get("embeddingScore") or -1):
                    match_map[track_id] = match
            selected_tracks = []
            for track_id in selected_ids:
                track = track_map.get(track_id) or {"trackId": track_id}
                match = match_map.get(track_id)
                selected_tracks.append(self._with_match(track, match) if match else track)
            unresolved_ids = acceptance.get("unresolvedTrackIds") or []
            completed = bool(acceptance.get("acceptanceSatisfied"))
            result_state = "conflict" if state == "conflict" else "sufficient" if completed else "uncertain"
            relation_name = "在库" if relation == "in" else "未在库"
            conclusion = f"{relation_name}船舶 {len(selected_tracks)} 条" if selected_tracks else f"未发现已确认的{relation_name}船舶"
            if unresolved_ids:
                conclusion = f"部分确认：{conclusion}"
            relation_reason = (
                f"验收目标：{self.meta.get('successCriteria') or '完成在库/未在库判定并得到轨迹列表'}；"
                f"范围轨迹 {acceptance.get('trackCount', 0)} 条，"
                f"已确认在库 {len(acceptance.get('inRegistryTrackIds') or [])} 条，"
                f"已确认未在库 {len(acceptance.get('outOfRegistryTrackIds') or [])} 条，"
                f"未完成分类 {len(unresolved_ids)} 条。"
            )
            return self._finish(
                conclusion,
                selected_tracks,
                answer_hint or relation_reason,
                result_state,
                extra={
                    "planMode": "autonomous",
                    "acceptanceProgress": compact_acceptance(acceptance),
                    "inRegistryTrackIds": acceptance.get("inRegistryTrackIds") or [],
                    "outOfRegistryTrackIds": acceptance.get("outOfRegistryTrackIds") or [],
                    "unresolvedTrackIds": unresolved_ids,
                    "registryItems": registry_items,
                },
                display={"tracks": selected_tracks, "includeClips": True, "includeRegistry": True},
            )

        if matches:
            supported_matches = [item for item in matches if item.get("scoreBand") != "mismatch"]
            matching_tracks = self._tracks_for_matches(supported_matches, tracks)
            extra = {
                "matches": matches,
                "matchedCount": len(supported_matches),
                "rejectedCount": len(matches) - len(supported_matches),
                "planMode": "autonomous",
            }
            if registry_items:
                extra["registryItems"] = registry_items
            if supported_matches:
                conclusion = "找到匹配目标" if any(item.get("scoreBand") == "match" for item in supported_matches) else "得到待核验候选"
                if not answer_hint:
                    answer_hint = f"共得到 {len(supported_matches)} 条匹配或待核验候选"
                return self._finish(
                    conclusion,
                    matching_tracks,
                    answer_hint or reason,
                    state,
                    extra=extra,
                    display={"tracks": matching_tracks, "includeClips": True, "includeRegistry": bool(registry_items)},
                )
            return self._finish("未找到匹配目标", [], answer_hint or reason, state, extra=extra)

        if count_value is not None:
            representatives = self._deduplication_representatives(tracks)
            conclusion = f"统计结果为 {count_value}"
            return self._finish(
                conclusion,
                representatives,
                answer_hint or reason,
                state,
                extra={"count": count_value, "sourceTrackCount": len(tracks), "planMode": "autonomous"},
                display={"tracks": representatives, "includeClips": True},
            )

        if registry_items and not tracks:
            conclusion = "已查询先验库"
            text = answer_hint or self._format_registry_hits(registry_items) or reason
            return self._finish(conclusion, [], text, state, extra={"registryItems": registry_items, "planMode": "autonomous"})

        if tracks:
            conclusion = "已定位相关轨迹" if state == "sufficient" else "仅获得部分轨迹证据"
            return self._finish(conclusion, tracks, answer_hint or reason, state, extra={"planMode": "autonomous"}, display={"tracks": tracks, "includeClips": True})

        conclusion = "未找到可靠证据" if state != "conflict" else "证据存在冲突"
        return self._finish(conclusion, [], answer_hint or reason, state, extra={"planMode": "autonomous"})

    def _collect_tracks_from_scope(self) -> list[dict[str, Any]]:
        collected: dict[str, dict[str, Any]] = {}
        for value in self.working_scope.values():
            if not isinstance(value, dict):
                continue
            for track in value.get("tracks") or []:
                if isinstance(track, dict) and track.get("trackId") is not None:
                    collected[str(track["trackId"])] = track
            # match results may embed track summaries
            for match in value.get("matches") or []:
                if not isinstance(match, dict):
                    continue
                track = match.get("track") or {}
                track_id = match.get("matchedTrackId") or match.get("trackId") or track.get("trackId")
                if track_id is not None:
                    item = dict(track) if isinstance(track, dict) else {"trackId": track_id}
                    item.setdefault("trackId", track_id)
                    if match.get("embeddingScore") is not None:
                        item["embeddingScore"] = match.get("embeddingScore")
                    if match.get("scoreBand"):
                        item["scoreBand"] = match.get("scoreBand")
                    collected[str(track_id)] = {**collected.get(str(track_id), {}), **item}
        return list(collected.values())

    def _collect_matches_from_scope(self) -> list[dict[str, Any]]:
        matches: list[dict[str, Any]] = []
        seen: set[str] = set()
        for value in self.working_scope.values():
            if not isinstance(value, dict):
                continue
            for match in value.get("matches") or []:
                if not isinstance(match, dict):
                    continue
                track_id = match.get("matchedTrackId") or match.get("trackId")
                registry_id = match.get("matchedRegistryId") or match.get("registryId")
                key = "|".join(
                    part
                    for part in (
                        f"track:{track_id}" if track_id is not None else "",
                        f"registry:{registry_id}" if registry_id is not None else "",
                        f"frame:{match.get('keyframeId')}" if match.get("keyframeId") is not None else "",
                    )
                    if part
                ) or str(len(matches))
                if key in seen:
                    continue
                seen.add(key)
                matches.append(match)
        matches.sort(key=lambda item: float(item.get("embeddingScore") or item.get("score") or 0), reverse=True)
        return matches

    def _collect_registry_from_scope(self) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        seen: set[str] = set()
        for value in self.working_scope.values():
            if not isinstance(value, dict):
                continue
            for item in value.get("registryItems") or []:
                if not isinstance(item, dict):
                    continue
                key = str(item.get("registryId") or item.get("hullNumber") or len(items))
                if key in seen:
                    continue
                seen.add(key)
                items.append(item)
        return items

    def _collect_count_from_scope(self) -> int | None:
        for value in reversed(list(self.working_scope.values())):
            if not isinstance(value, dict):
                continue
            for key in ("highThresholdShipCount", "uniqueCount", "count", "dedupCount", "finalCount"):
                if value.get(key) is not None:
                    try:
                        return int(value[key])
                    except (TypeError, ValueError):
                        pass
            groups = value.get("highGroups") or value.get("upperGroups") or value.get("groups") or value.get("mergedGroups")
            if isinstance(groups, list) and groups:
                return len(groups)
        return None

    def _deduplication_representatives(
        self,
        tracks: list[dict[str, Any]],
        deduplication: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        deduplication = deduplication or self._latest_deduplication_result()
        groups = deduplication.get("highGroups") if isinstance(deduplication, dict) else None
        if not isinstance(groups, list):
            return []

        track_map = {
            str(track.get("trackId")): track
            for track in tracks
            if isinstance(track, dict) and track.get("trackId") is not None
        }
        keyframes_by_track = self._keyframes_by_track_from_scope()
        representatives = []
        for group in groups:
            if not isinstance(group, list) or not group:
                continue
            group_ids = [str(track_id) for track_id in group]
            representative: dict[str, Any] | None = None
            best_frame: dict[str, Any] | None = None
            best_score = float("-inf")
            for track_id in group_ids:
                track = track_map.get(track_id)
                if track is None:
                    continue
                frames = keyframes_by_track.get(track_id, {}).get("keyframes", [])
                frame = max(
                    (item for item in frames if isinstance(item, dict)),
                    key=lambda item: float(item.get("retentionScore") or 0),
                    default=None,
                )
                score = float(frame.get("retentionScore") or 0) if frame else float("-inf")
                if representative is None or score > best_score:
                    representative = track
                    best_frame = frame
                    best_score = score
            if representative is None:
                continue
            item = dict(representative, deduplicatedTrackIds=group_ids)
            if best_frame and best_frame.get("keyframeId"):
                item["matchedKeyframeIds"] = [best_frame["keyframeId"]]
            representatives.append(item)
        return representatives

    def _latest_deduplication_result(self) -> dict[str, Any]:
        for value in reversed(list(self.working_scope.values())):
            if isinstance(value, dict) and isinstance(value.get("highGroups"), list):
                return value
        return {}

    def _keyframes_by_track_from_scope(self) -> dict[str, dict[str, Any]]:
        collected: dict[str, dict[str, Any]] = {}
        for value in self.working_scope.values():
            if not isinstance(value, dict):
                continue
            groups = value.get("keyframesByTrack")
            if not isinstance(groups, dict):
                continue
            for track_id, group in groups.items():
                if isinstance(group, dict):
                    collected[str(track_id)] = group
        return collected

    def _tracks_from_matches(self, matches: list[dict[str, Any]]) -> list[dict[str, Any]]:
        tracks = []
        for match in matches:
            track_id = match.get("matchedTrackId") or match.get("trackId")
            if track_id is None:
                continue
            item = {"trackId": track_id}
            if match.get("embeddingScore") is not None:
                item["embeddingScore"] = match.get("embeddingScore")
            if match.get("scoreBand"):
                item["scoreBand"] = match.get("scoreBand")
            if match.get("hullNumber"):
                item["finalHullNumber"] = match.get("hullNumber")
            tracks.append(item)
        return tracks

    def _tracks_for_matches(
        self,
        matches: list[dict[str, Any]],
        known_tracks: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """只为实际命中的轨迹补齐摘要，避免把 getTrack 的全量候选当作结果返回。"""
        track_map = {
            str(item.get("trackId")): item
            for item in known_tracks
            if isinstance(item, dict) and item.get("trackId") is not None
        }
        selected: dict[str, dict[str, Any]] = {}
        for match in matches:
            if not isinstance(match, dict):
                continue
            track_id = match.get("matchedTrackId") or match.get("trackId")
            if track_id is None:
                continue
            key = str(track_id)
            base = track_map.get(key)
            if base is None and isinstance(match.get("track"), dict):
                base = match["track"]
            if base is None:
                try:
                    base = self.repository.get_track(track_id)
                except Exception:
                    base = None
            enriched = self._with_match(base or {"trackId": track_id}, match)
            existing = selected.get(key)
            if existing is None or float(enriched.get("embeddingScore") or -1) > float(existing.get("embeddingScore") or -1):
                selected[key] = enriched
        return sorted(selected.values(), key=lambda item: float(item.get("embeddingScore") or 0), reverse=True)


    def _round(self, goal: str, calls: list[dict[str, Any]], default_state: str, reason: str, evidence_gap: str | None = None) -> dict[str, Any]:
        if len(self.rounds) >= self.max_rounds:
            raise RuntimeError("达到最大推理轮次")
        round_number = len(self.rounds) + 1
        self._emit("agent_start", "PlanAgent", "仅生成本轮工具计划，不执行工具", round=round_number, role="planner", executionOwner="ObserveAgent")
        guided_goal = goal
        if self.meta.get("expectedOutcome") and self.meta.get("expectedOutcome") not in guided_goal:
            guided_goal = f"{goal}｜验收：{self.meta.get('expectedOutcome')}"
        plan = self.planner.build(guided_goal, calls, self.meta.get("timeRange"), evidence_gap, lambda delta: self._emit("agent_delta", "PlanAgent", "", round=round_number, role="planner", delta=delta), intent=self.meta)
        self._apply_query_top_k(plan)
        calls = plan.get("calls") or []
        public_plan = self._public_plan(plan, calls)
        self._emit("agent_end", "PlanAgent", "规划完成，等待 ObserveAgent 执行", round=round_number, role="planner", calls=public_plan["calls"], modelSummary=plan.get("modelPlan"), fallback=plan.get("modelFallback"), planOnly=True, executionOwner="ObserveAgent", planBlueprint=self.plan_blueprint)

        self._emit("agent_start", "ObserveAgent", "执行 PlanAgent 的工具计划", round=round_number, role="observer", executionOwner="ObserveAgent")
        observed = self.observer.execute(
            plan,
            on_delta=lambda delta: self._emit("agent_delta", "ObserveAgent", "", round=round_number, role="observer", delta=delta),
            on_tool_event=lambda event: self._emit_observer_tool_event(round_number, event),
        )
        self._emit("agent_end", "ObserveAgent", "工具执行完成", round=round_number, role="observer", calls=observed["summary"].get("calls", []), modelSummary=observed["summary"].get("modelObservation"), fallback=observed["summary"].get("modelFallback"), executionOwner="ObserveAgent")

        self._emit("agent_start", "ReflectAgent", "正在检查证据充分性、冲突与后续动作", round=round_number, role="reflector")
        reflection = self.reflector.review(default_state, reason, observed["summary"], evidence_gap, lambda delta: self._emit("agent_delta", "ReflectAgent", "", round=round_number, role="reflector", delta=delta))
        self._emit("agent_end", "ReflectAgent", reflection.get("reason", reason), round=round_number, role="reflector", state=reflection.get("state"), evidenceGap=reflection.get("evidenceGap"), modelSummary=reflection.get("modelReflection"), fallback=reflection.get("modelFallback"), nextAction=reflection.get("nextAction"))
        round_id = f"round-{uuid.uuid4().hex[:12]}"
        self.repository.add_round(round_id, self.session_id, public_plan, reflection)
        self._store_observations(round_id, observed, round_number)
        record = {"roundId": round_id, "plan": public_plan, "observation": observed["summary"], "reflection": reflection, "scope": observed["scope"]}
        self.rounds.append(record)
        return record

    @staticmethod
    def _evidence_similarity(track: dict[str, Any]) -> float | None:
        for key in ("embeddingScore", "score", "matchScore"):
            value = track.get(key)
            if value is None:
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
        return None

    def _display_tracks(self, tracks: list[dict[str, Any]], include_clips: bool = True, include_registry: bool = False) -> None:
        if self.display_record is not None:
            return
        unique_tracks = list({str(item["trackId"]): item for item in tracks}.values())
        ranked_tracks = []
        for index, track in enumerate(unique_tracks):
            score = self._evidence_similarity(track)
            ranked_tracks.append((score is not None, score if score is not None else 0.0, -index, track))
        unique_tracks = [item[-1] for item in sorted(ranked_tracks, reverse=True)]
        if not unique_tracks:
            self.display_record = {
                "displayId": f"display-{uuid.uuid4().hex[:12]}",
                "mode": "lazy",
                "trackCount": 0,
                "registryReferenceCount": 0,
            }
            return
        missing_frame_tracks = [
            str(track["trackId"])
            for track in unique_tracks
            if not self._ids(track, "matchedKeyframeIds", "queryKeyframeIds", "keyframeIds")
        ]
        frame_groups = self.tools.getFrames(missing_frame_tracks).get("keyframesByTrack", {}) if missing_frame_tracks else {}
        track_references = {
            str(track["trackId"]): self._ids(track, "matchedRegistryReferenceIds", "queryRegistryReferenceIds", "registryReferenceIds")
            for track in unique_tracks
        }
        for track in unique_tracks:
            track_id = str(track["trackId"])
            keyframe_ids = self._ids(track, "matchedKeyframeIds", "queryKeyframeIds", "keyframeIds")
            if not keyframe_ids:
                frames = frame_groups.get(track_id, {}).get("keyframes", [])
                best = max(frames, key=lambda item: item.get("retentionScore", 0), default=None)
                keyframe_ids = [best["keyframeId"]] if best else []
            segment_ids = self._ids(track, "shipSegmentIds")[:1]
            reference_ids = self.tools._representative_registry_reference_ids(
                track_references[track_id]
            )[:1] if include_registry else []
            self.display_groups.append({
                "trackId": track_id,
                "hullNumber": track.get("finalHullNumber"),
                "embeddingScore": self._evidence_similarity(track),
                "keyframeIds": keyframe_ids[:1],
                "shipSegmentIds": segment_ids,
                "clipTrackId": track_id if include_clips else None,
                "clipTimeRange": list(self.meta["timeRange"]) if self.meta.get("timeRange") else None,
                "registryReferenceIds": reference_ids,
                "clipError": None,
            })
        self.display_record = {
            "displayId": f"display-{uuid.uuid4().hex[:12]}",
            "mode": "lazy",
            "trackCount": len(self.display_groups),
            "registryReferenceCount": sum(bool(group["registryReferenceIds"]) for group in self.display_groups),
        }

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

    def _store_observations(self, record_id: str, observed: dict[str, Any], round_number: int) -> None:
        for observation in observed["observations"]:
            if observation.get("skipped"):
                continue
            evidence_id = f"evidence-{uuid.uuid4().hex[:12]}"
            audit_result = Observer._summarize_observation(observation)
            self.repository.add_evidence(evidence_id, record_id, audit_result, {"tool": observation["tool"], "callId": observation["id"]})
            self.tool_chain.append(f"{observation['tool']}({observation['id']})")
            self.tool_records.append({"round": round_number, **audit_result})

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
            "toolRecords": result.get("toolRecords", [])[:30],
            "evidence": result.get("evidence", {}),
            "roundCount": len(result.get("rounds", [])),
        }

    def _append_tool_chain(self, observed: dict[str, Any]) -> None:
        for observation in observed["observations"]:
            if not observation.get("skipped"):
                self.tool_chain.append(f"{observation['tool']}({observation['id']})")

    def _finish(self, conclusion: str, tracks: list[dict[str, Any]], reason: str, state: str, extra: dict[str, Any] | None = None, display: dict[str, Any] | None = None) -> dict[str, Any]:
        display = display if display is not None else {"tracks": tracks}
        self._display_tracks(
            display.get("tracks") or [],
            display.get("includeClips", True) is not False,
            bool(display.get("includeRegistry")),
        )
        primary = [item["trackId"] for item in tracks[:self.display_limit]]
        result = {"sessionId": self.session_id, "question": self.question, "questionType": self.meta.get("questionType"), "conclusion": conclusion, "answerText": f"{conclusion}。{reason}", "queryScope": list(self.meta["timeRange"]) if self.meta.get("timeRange") else None, "toolChain": self.tool_chain, "toolRecords": self.tool_records, "tracks": tracks, "evidence": self._collect_evidence(), "displayGroups": self.display_groups, "display": self._public_display(), "uncertainty": state, "primaryTrackIds": primary, "remainingTrackIds": [item["trackId"] for item in tracks[self.display_limit:]], "rounds": [{key: value for key, value in item.items() if key != "scope"} for item in self.rounds]}
        if extra:
            result.update(extra)
        self._emit("synthesis", "生成最终回答", reason, conclusion=conclusion, state=state, trackCount=len(tracks))
        return result

    def _collect_evidence(self) -> dict[str, list[str]]:
        if self.display_record is not None:
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

    def _apply_query_top_k(self, plan: dict[str, Any]) -> None:
        top_k = max(1, min(20, int(getattr(self, "query_top_k", 3))))
        for call in plan.get("calls") or []:
            if call.get("tool") not in {"matchText", "matchImage"}:
                continue
            call.setdefault("arguments", {})["topK"] = top_k

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
        public["calls"] = [{"id": call["id"], "tool": call["tool"], "arguments": self._compact_arguments(call.get("arguments", {})), "condition": call.get("condition"), "planStepId": self._plan_step_id_for_tool(call.get("tool"))} for call in calls]
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
        for key in ("matchedTrackId", "matchedRegistryId", "embeddingScore", "scoreBand", "queryKeyframeIds", "queryRegistryReferenceIds", "matchedKeyframeIds", "matchedRegistryReferenceIds", "shipSegmentIds"):
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




    def _registry_item_map(self, catalog: dict[str, Any]) -> dict[str, dict[str, Any]]:
        items = catalog.get("registryItems") or []
        return {str(item.get("registryId")): item for item in items if item.get("registryId") is not None}

    def _enrich_registry_matches(self, matches: list[dict[str, Any]], catalog: dict[str, Any]) -> list[dict[str, Any]]:
        item_map = self._registry_item_map(catalog)
        enriched = []
        for match in matches or []:
            item = dict(match)
            registry_id = str(item.get("matchedRegistryId") or item.get("registryId") or "")
            record = item_map.get(registry_id) or {}
            item["registryId"] = registry_id or record.get("registryId")
            item["hullNumber"] = record.get("hullNumber") or record.get("hull_number") or item.get("hullNumber")
            item["description"] = record.get("description") or item.get("description")
            item["registryReferenceIds"] = item.get("matchedRegistryReferenceIds") or item.get("registryReferenceIds") or []
            # 兼容 references 字段
            if not item["registryReferenceIds"] and record.get("references"):
                item["registryReferenceIds"] = [ref.get("referenceId") for ref in record.get("references", []) if ref.get("referenceId")]
            enriched.append(item)
        return enriched

    def _format_registry_hits(self, matches: list[dict[str, Any]], limit: int | None = None) -> str:
        rows = []
        for item in (matches or [])[: (limit or self.display_limit)]:
            hull = item.get("hullNumber") or "未知舷号"
            registry_id = item.get("registryId") or item.get("matchedRegistryId") or "未知库项"
            score = item.get("embeddingScore")
            score_text = f"，相似度 {float(score):.3f}" if isinstance(score, (int, float)) else ""
            band = item.get("scoreBand") or item.get("verifyDecision") or ""
            band_text = f"，状态 {band}" if band else ""
            rows.append(f"{hull}（库项 {registry_id}{score_text}{band_text}）")
        return "；".join(rows)

    def _guard_meta(self, meta: dict[str, Any]) -> dict[str, Any]:
        """最小结构校验：防止无效 description 触发错误工具链。意图本身由规则表+模型决定。"""
        description = str(meta.get("description") or "").strip()
        weak = description in {"", "船", "船舶", "船只", "目标", "在库船", "未在库船", "库船", "未在", "在"}
        if meta.get("questionType") == "relation_description" and weak:
            # 无真实外观时，在库/未在库应走结构化库关系链路
            meta["description"] = None
            meta["targetKind"] = "all"
            if meta.get("registryRelation") == "out":
                meta["questionType"] = "out_of_registry"
                meta["strategy"] = "registry_relation"
            else:
                meta["questionType"] = "in_registry"
                meta["strategy"] = "registry_relation"
                meta["registryRelation"] = meta.get("registryRelation") or "in"
        if meta.get("questionType") in {"description", "description_count", "registry_description", "registry_description_count", "relation_description"} and weak:
            # 描述类问题缺少有效 targetText 时，避免 matchText 空转
            if meta.get("questionType") == "description":
                meta["questionType"] = "track_list"
                meta["strategy"] = "track_list"
                meta["targetKind"] = "all"
                meta["description"] = None
            elif meta.get("questionType") == "description_count":
                meta["questionType"] = "count"
                meta["strategy"] = "track_count"
                meta["targetKind"] = "all"
                meta["description"] = None
            elif meta.get("questionType") == "registry_description":
                meta["questionType"] = "registry_list"
                meta["strategy"] = "registry_list"
                meta["targetKind"] = "all"
                meta["description"] = None
            elif meta.get("questionType") == "registry_description_count":
                meta["questionType"] = "registry_count"
                meta["strategy"] = "registry_count"
                meta["targetKind"] = "all"
                meta["description"] = None
        return meta

    def _finalize_answer(self, result: dict[str, Any]) -> dict[str, Any]:
        """根据 operation 重组答案文本，保留原有证据与结论。"""
        operation = self.meta.get("operation") or "existence"
        tracks = result.get("tracks") or []
        description = self.meta.get("description") or result.get("description") or result.get("relationDescription")
        hull = self.meta.get("hullNumber")
        if operation == "time":
            windows = self._track_time_windows(tracks)
            if windows:
                text = "；".join(windows[: self.display_limit])
                more = f" 等共 {len(windows)} 段" if len(windows) > self.display_limit else ""
                result["conclusion"] = "时间定位完成"
                result["answerText"] = f"目标出现时间：{text}{more}。"
                result["timeWindows"] = windows
            elif result.get("uncertainty") == "sufficient":
                result["conclusion"] = "未定位到出现时间"
                result["answerText"] = "查询范围内没有可用于时间定位的轨迹。"
            else:
                result["conclusion"] = result.get("conclusion") or "无法确认出现时间"
                result["answerText"] = f"{result.get('conclusion')}。当前证据不足以完成时间定位。"
        elif operation == "explain":
            evidence = result.get("evidence") or {}
            parts = [result.get("conclusion") or "证据解释"]
            if hull:
                parts.append(f"目标舷号 {hull}")
            if description:
                parts.append(f"目标描述“{description}”")
            if tracks:
                sample = tracks[0]
                hull_value = sample.get("finalHullNumber") or sample.get("hullNumber") or "无稳定舷号"
                match_type = sample.get("finalMatchType") or sample.get("scoreBand") or "unknown"
                score = sample.get("embeddingScore")
                score_text = f"，相似度 {float(score):.3f}" if isinstance(score, (int, float)) else ""
                parts.append(f"主要依据轨迹 {sample.get('trackId')}（舷号 {hull_value}，状态 {match_type}{score_text}）")
            key_count = len(evidence.get("keyframeIds") or [])
            clip_count = len(evidence.get("shipSegmentIds") or [])
            ref_count = len(evidence.get("registryReferenceIds") or [])
            parts.append(f"已整理关键帧 {key_count} 张、视频片段 {clip_count} 段、库参考图 {ref_count} 张")
            result["conclusion"] = "证据解释完成"
            result["answerText"] = "。".join(parts) + "。"
        elif operation == "list" and tracks:
            if "查询完成" in str(result.get("conclusion", "")) or "找到" in str(result.get("conclusion", "")):
                result["answerText"] = f"共列出 {len(tracks)} 条相关结果。{result.get('answerText', '')}"
        result["operation"] = operation
        result["strategy"] = self.meta.get("strategy")
        result["targetScope"] = self.meta.get("targetScope")
        result["registryRelation"] = self.meta.get("registryRelation")
        if description and "description" not in result:
            result["description"] = description
        if hull and "hullNumber" not in result:
            result["hullNumber"] = hull
        return result

    @staticmethod
    def _format_monitor_time(value: Any) -> str:
        try:
            timestamp = float(value)
        except (TypeError, ValueError):
            return "未知时间"
        from datetime import datetime
        return datetime.fromtimestamp(timestamp).astimezone().strftime("%H:%M:%S")

    def _track_time_windows(self, tracks: list[dict[str, Any]]) -> list[str]:
        windows = []
        for track in tracks:
            start = track.get("startTime", track.get("start_time"))
            end = track.get("endTime", track.get("end_time"))
            if start is None or end is None:
                continue
            windows.append(f"{self._format_monitor_time(start)}—{self._format_monitor_time(end)}")
        return windows

    def _description_target(self) -> str:
        if self.meta.get("description"):
            return str(self.meta["description"]).strip()
        question = self.question
        for prefix in (
            "数据库中有没有出现", "数据库中是否出现", "先验库中有没有出现", "先验库中是否出现",
            "视频中有没有出现", "视频中是否出现", "监控里有没有", "监控中是否出现",
            "有没有出现", "找一下", "查找一下", "帮我找", "请查找",
        ):
            question = question.replace(prefix, "")
        return question.strip("？?。 ") or self.question
