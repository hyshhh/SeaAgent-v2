"""LangGraph 四 Agent 控制器：对外保持 answer() 与事件契约。"""
from __future__ import annotations

import uuid
from typing import Any, Callable

from config import load_config
from memory import MemoryRepository
from services import AgentLLMService, QwenMultimodalEmbedder
from tools import ToolService
from vector_store import VectorCatalog

from .graph import run_sea_agent


class AgentController:
    """前端入口：内部运行 LangGraph（Intent→Plan→Observe→Reflect）。"""

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        repository: MemoryRepository | None = None,
        tools: ToolService | None = None,
        llm: AgentLLMService | None = None,
        embedder: QwenMultimodalEmbedder | None = None,
        vectors: VectorCatalog | None = None,
        event_handler: Callable[[dict[str, Any]], None] | None = None,
    ):
        self.config = config or load_config()
        self.repository = repository or MemoryRepository(self.config)
        self.llm = llm or AgentLLMService(self.config)
        self.tools = tools or ToolService(self.config, self.repository, embedder, self.llm, vectors)
        settings = self.config.get("pipeline", {}).get("agent", {})
        self.max_rounds = int(settings.get("max_rounds", 3))
        self.display_limit = int(settings.get("display_limit", 3))
        self.event_handler = event_handler
        self.session_id = ""
        self.question = ""
        self.meta: dict[str, Any] = {}
        self.rounds: list[dict[str, Any]] = []
        self.tool_chain: list[str] = []
        self.tool_records: list[dict[str, Any]] = []
        self.display_record: dict[str, Any] | None = None
        self.display_groups: list[dict[str, Any]] = []
        self.working_scope: dict[str, Any] = {}

    def _emit(self, event_type: str, title: str, message: str, **payload: Any) -> None:
        if not self.event_handler:
            return
        try:
            self.event_handler({"type": event_type, "title": title, "message": message, **payload})
        except Exception:
            pass

    def answer(self, question: str, top_k: int | None = None) -> dict[str, Any]:
        self.session_id = f"session-{uuid.uuid4().hex[:12]}"
        self.question = str(question or "").strip()
        self.rounds, self.tool_chain, self.tool_records = [], [], []
        self.display_record, self.display_groups = None, []
        self.working_scope = {}
        agent_settings = self.config.get("pipeline", {}).get("agent", {})
        self.max_rounds = int(agent_settings.get("max_rounds", self.max_rounds))
        self.display_limit = int(agent_settings.get("display_limit", self.display_limit))
        default_top_k = int(self.config.get("pipeline", {}).get("retrieval", {}).get("top_k", 3))
        self.query_top_k = max(1, min(20, int(top_k if top_k is not None else default_top_k)))

        self._emit("status", "Controller", "LangGraph 四 Agent 协同启动", planMode="langgraph")
        try:
            state = run_sea_agent(
                self.question,
                self.llm,
                self.tools,
                max_rounds=self.max_rounds,
                query_top_k=self.query_top_k,
                event_handler=self.event_handler,
            )
        except Exception as error:
            result = self._finish(
                "执行失败",
                [],
                f"LangGraph 执行失败：{error}",
                "uncertain",
                extra={"error": str(error), "planMode": "langgraph"},
            )
            try:
                self.repository.finish_session(self.session_id, self._session_audit_result(result))
            except Exception:
                pass
            return result

        self.meta = dict(state.get("intent") or {})
        self.meta["planMode"] = "langgraph"
        self.meta["maxRounds"] = self.max_rounds
        self.meta["retrievalTopK"] = self.query_top_k
        self.working_scope = dict(state.get("working_scope") or {})
        self.rounds = list(state.get("rounds") or [])
        self.tool_chain = list(state.get("tool_chain") or [])
        self.tool_records = list(state.get("tool_records") or [])

        try:
            self.repository.add_session(self.session_id, {"question": self.question, **self.meta})
        except Exception:
            pass

        final_state = str(state.get("final_state") or "uncertain")
        final_reason = str(state.get("final_reason") or "协同结束")
        result = self._synthesize(final_state, final_reason)
        try:
            self.repository.finish_session(self.session_id, self._session_audit_result(result))
        except Exception:
            pass
        return result

    def _synthesize(self, state: str, reason: str) -> dict[str, Any]:
        tracks = self._collect_tracks()
        matches = self._collect_matches()
        registry_items = self._collect_registry()
        count_value = self._collect_count()
        answer_hint = reason

        if count_value is not None:
            return self._finish(
                f"统计结果为 {count_value}",
                tracks[: self.display_limit],
                answer_hint,
                state,
                extra={"count": count_value, "planMode": "langgraph"},
                display={"tracks": tracks, "includeClips": True},
            )
        if matches:
            supported = [m for m in matches if m.get("scoreBand") != "mismatch"]
            if supported:
                return self._finish(
                    "找到匹配目标",
                    tracks or self._tracks_from_matches(supported),
                    answer_hint or f"共 {len(supported)} 条候选",
                    state,
                    extra={"matches": matches, "planMode": "langgraph"},
                    display={"tracks": tracks or self._tracks_from_matches(supported), "includeClips": True},
                )
            return self._finish("未找到匹配目标", [], answer_hint, state, extra={"matches": matches, "planMode": "langgraph"})
        if registry_items and not tracks:
            # 标记 targetScope，前端 registry-only 证据列可直接使用
            if not self.meta.get("targetScope"):
                self.meta["targetScope"] = "registry"
            return self._finish(
                "已查询先验库",
                [],
                answer_hint,
                state,
                extra={
                    "registryItems": registry_items,
                    "planMode": "langgraph",
                    "targetScope": self.meta.get("targetScope") or "registry",
                },
                display={"tracks": [], "includeClips": False, "includeRegistry": True},
            )
        if tracks:
            conclusion = "已定位相关轨迹" if state == "sufficient" else "仅获得部分轨迹证据"
            return self._finish(
                conclusion,
                tracks,
                answer_hint,
                state,
                extra={"planMode": "langgraph"},
                display={"tracks": tracks, "includeClips": True},
            )
        conclusion = "证据存在冲突" if state == "conflict" else "未找到可靠证据"
        return self._finish(conclusion, [], answer_hint, state, extra={"planMode": "langgraph"})

    def _collect_tracks(self) -> list[dict[str, Any]]:
        collected: dict[str, dict[str, Any]] = {}
        for value in self.working_scope.values():
            if not isinstance(value, dict):
                continue
            for track in value.get("tracks") or []:
                if isinstance(track, dict) and track.get("trackId") is not None:
                    collected[str(track["trackId"])] = track
            for match in value.get("matches") or []:
                if not isinstance(match, dict):
                    continue
                track = match.get("track") if isinstance(match.get("track"), dict) else {}
                track_id = match.get("matchedTrackId") or match.get("trackId") or track.get("trackId")
                if track_id is None:
                    continue
                item = dict(track) if track else {"trackId": track_id}
                item.setdefault("trackId", track_id)
                for key in ("embeddingScore", "scoreBand", "hullNumber"):
                    if match.get(key) is not None:
                        item[key] = match.get(key)
                collected[str(track_id)] = {**collected.get(str(track_id), {}), **item}
        return list(collected.values())

    def _collect_matches(self) -> list[dict[str, Any]]:
        matches: list[dict[str, Any]] = []
        for value in self.working_scope.values():
            if isinstance(value, dict):
                for match in value.get("matches") or []:
                    if isinstance(match, dict):
                        matches.append(match)
        matches.sort(key=lambda item: float(item.get("embeddingScore") or item.get("score") or 0), reverse=True)
        return matches

    def _collect_registry(self) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        seen: set[str] = set()

        def _add(item: dict[str, Any]) -> None:
            if not isinstance(item, dict):
                return
            key = str(item.get("registryId") or item.get("hullNumber") or len(items))
            if key in seen:
                return
            seen.add(key)
            items.append(item)

        for value in self.working_scope.values():
            if not isinstance(value, dict):
                continue
            for item in value.get("registryItems") or []:
                _add(item)
            # matchHull 返回 exactMatches: {hull: [registryItem, ...]}
            exact = value.get("exactMatches")
            if isinstance(exact, dict):
                for bucket in exact.values():
                    if isinstance(bucket, list):
                        for item in bucket:
                            _add(item)
                    elif isinstance(bucket, dict):
                        _add(bucket)
        return items

    def _collect_count(self) -> int | None:
        for value in reversed(list(self.working_scope.values())):
            if not isinstance(value, dict):
                continue
            for key in ("highThresholdShipCount", "uniqueCount", "count", "dedupCount", "finalCount"):
                if value.get(key) is not None:
                    try:
                        return int(value[key])
                    except (TypeError, ValueError):
                        pass
            groups = value.get("highGroups") or value.get("groups")
            if isinstance(groups, list) and groups:
                return len(groups)
        return None

    @staticmethod
    def _tracks_from_matches(matches: list[dict[str, Any]]) -> list[dict[str, Any]]:
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
            tracks.append(item)
        return tracks

    def _finish(
        self,
        conclusion: str,
        tracks: list[dict[str, Any]],
        reason: str,
        state: str,
        extra: dict[str, Any] | None = None,
        display: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        display = display if display is not None else {"tracks": tracks}
        self._display_tracks(
            display.get("tracks") or [],
            display.get("includeClips", True) is not False,
            bool(display.get("includeRegistry")),
        )
        primary = [item["trackId"] for item in tracks[: self.display_limit] if item.get("trackId") is not None]
        result = {
            "sessionId": self.session_id,
            "question": self.question,
            "questionType": self.meta.get("questionType"),
            "targetScope": self.meta.get("targetScope"),
            "targetKind": self.meta.get("targetKind"),
            "operation": self.meta.get("operation"),
            "registryRelation": self.meta.get("registryRelation"),
            "conclusion": conclusion,
            "answerText": f"{conclusion}。{reason}",
            "queryScope": list(self.meta["timeRange"]) if self.meta.get("timeRange") else None,
            "toolChain": self.tool_chain,
            "toolRecords": self.tool_records,
            "tracks": tracks,
            "evidence": self._collect_evidence(),
            "displayGroups": self.display_groups,
            "display": self._public_display(),
            "uncertainty": state,
            "primaryTrackIds": primary,
            "remainingTrackIds": [
                item["trackId"] for item in tracks[self.display_limit :] if item.get("trackId") is not None
            ],
            "rounds": [{k: v for k, v in item.items() if k != "scope"} for item in self.rounds],
            "intent": self.meta,
            "planMode": "langgraph",
        }
        if extra:
            result.update(extra)
        self._emit("synthesis", "生成最终回答", reason, conclusion=conclusion, state=state, trackCount=len(tracks))
        return result

    def _display_tracks(self, tracks: list[dict[str, Any]], include_clips: bool = True, include_registry: bool = False) -> None:
        if self.display_record is not None:
            return
        unique_tracks = list({str(item["trackId"]): item for item in tracks if item.get("trackId") is not None}.values())
        if not unique_tracks:
            # 仅先验库结果：从 working_scope 抽参考图 ID，供前端 registry-only 展示
            reference_ids: list[str] = []
            if include_registry:
                for value in self.working_scope.values():
                    if not isinstance(value, dict):
                        continue
                    for key in ("registryReferenceIds", "shownRegistryReferenceIds"):
                        for item in value.get(key) or []:
                            if item is not None:
                                reference_ids.append(str(item))
                    for item in value.get("registryItems") or []:
                        if not isinstance(item, dict):
                            continue
                        for ref in item.get("references") or []:
                            if isinstance(ref, dict) and ref.get("referenceId"):
                                reference_ids.append(str(ref["referenceId"]))
                    exact = value.get("exactMatches")
                    if isinstance(exact, dict):
                        for bucket in exact.values():
                            rows = bucket if isinstance(bucket, list) else [bucket]
                            for item in rows:
                                if not isinstance(item, dict):
                                    continue
                                for ref in item.get("references") or []:
                                    if isinstance(ref, dict) and ref.get("referenceId"):
                                        reference_ids.append(str(ref["referenceId"]))
                reference_ids = list(dict.fromkeys(reference_ids))
                if hasattr(self.tools, "_representative_registry_reference_ids"):
                    try:
                        reference_ids = list(self.tools._representative_registry_reference_ids(reference_ids))
                    except Exception:
                        pass
            self.display_record = {
                "displayId": f"display-{uuid.uuid4().hex[:12]}",
                "mode": "lazy",
                "trackCount": 0,
                "registryReferenceCount": len(reference_ids),
            }
            if reference_ids:
                # evidence 汇总依赖 display_groups 的 id 列表
                self.display_groups.append({
                    "trackId": None,
                    "keyframeIds": [],
                    "shipSegmentIds": [],
                    "registryReferenceIds": reference_ids[:12],
                })
            return
        # 按相似度粗排（有分数的靠前）
        def _score(track: dict[str, Any]) -> float:
            for key in ("embeddingScore", "score", "matchScore"):
                if track.get(key) is not None:
                    try:
                        return float(track[key])
                    except (TypeError, ValueError):
                        pass
            return float("-inf")

        unique_tracks = sorted(unique_tracks, key=_score, reverse=True)
        missing = [
            str(t["trackId"])
            for t in unique_tracks
            if not self._ids(t, "matchedKeyframeIds", "queryKeyframeIds", "keyframeIds")
        ]
        frame_groups = {}
        if missing:
            try:
                frame_groups = self.tools.getFrames(missing).get("keyframesByTrack", {}) or {}
            except Exception:
                frame_groups = {}
        for track in unique_tracks:
            track_id = str(track["trackId"])
            keyframe_ids = self._ids(track, "matchedKeyframeIds", "queryKeyframeIds", "keyframeIds")
            if not keyframe_ids:
                frames = frame_groups.get(track_id, {}).get("keyframes", [])
                best = max(frames, key=lambda item: item.get("retentionScore", 0), default=None)
                keyframe_ids = [best["keyframeId"]] if best and best.get("keyframeId") else []
            segment_ids = self._ids(track, "shipSegmentIds")[:1]
            if include_clips and not segment_ids and hasattr(self.tools, "getClip"):
                try:
                    clip = self.tools.getClip(track_id)
                    if isinstance(clip, dict) and clip.get("shipSegmentId"):
                        segment_ids = [str(clip["shipSegmentId"])]
                except Exception:
                    pass
            reference_ids: list[str] = []
            if include_registry:
                raw_refs = self._ids(track, "matchedRegistryReferenceIds", "registryReferenceIds")
                if hasattr(self.tools, "_representative_registry_reference_ids"):
                    try:
                        raw_refs = self.tools._representative_registry_reference_ids(raw_refs)
                    except Exception:
                        pass
                reference_ids = list(raw_refs)[:1]
            group: dict[str, Any] = {
                "trackId": track_id,
                "clipTrackId": track_id,
                "keyframeIds": keyframe_ids[:3],
                "shipSegmentIds": segment_ids,
                "registryReferenceIds": reference_ids,
            }
            for key in ("embeddingScore", "score", "matchScore", "scoreBand", "hullNumber"):
                if track.get(key) is not None:
                    group[key] = track[key]
            self.display_groups.append(group)
        self.display_record = {
            "displayId": f"display-{uuid.uuid4().hex[:12]}",
            "mode": "lazy",
            "trackCount": len(unique_tracks),
            "registryReferenceCount": sum(len(g.get("registryReferenceIds") or []) for g in self.display_groups),
            "groups": self.display_groups,
        }

    def _collect_evidence(self) -> dict[str, list[str]]:
        if self.display_groups:
            return {
                key: list(dict.fromkeys(value for group in self.display_groups for value in group.get(key) or []))
                for key in ("keyframeIds", "shipSegmentIds", "registryReferenceIds")
            }
        return {"keyframeIds": [], "shipSegmentIds": [], "registryReferenceIds": []}

    def _public_display(self) -> dict[str, Any] | None:
        if not self.display_record:
            return None
        return {key: value for key, value in self.display_record.items() if key != "scope"}

    @staticmethod
    def _ids(item: dict[str, Any], *keys: str) -> list[str]:
        values = []
        for key in keys:
            value = item.get(key, [])
            values.extend(value if isinstance(value, list) else [value] if value else [])
        return list(dict.fromkeys(str(value) for value in values if value))

    @staticmethod
    def _session_audit_result(result: dict[str, Any]) -> dict[str, Any]:
        return {
            "conclusion": result.get("conclusion"),
            "uncertainty": result.get("uncertainty"),
            "trackCount": len(result.get("tracks") or []),
            "toolChain": result.get("toolChain") or [],
            "planMode": result.get("planMode") or "langgraph",
        }
