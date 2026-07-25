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
        self._pending_registry_items: list[dict[str, Any]] = []

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
        self._pending_registry_items: list[dict[str, Any]] = []
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
        operation = str(self.meta.get("operation") or "")
        target_kind = str(self.meta.get("targetKind") or "")
        target_scope = str(self.meta.get("targetScope") or "")
        description = str(self.meta.get("description") or "").strip()
        hull = str(self.meta.get("hullNumber") or "").strip()

        if count_value is not None:
            return self._finish(
                f"统计结果为 {count_value}",
                tracks[: self.display_limit],
                answer_hint,
                state,
                extra={"count": count_value, "planMode": "langgraph"},
                display={"tracks": tracks, "includeClips": True},
            )
        registry_relation = str(self.meta.get("registryRelation") or "")
        question_type = str(self.meta.get("questionType") or "")
        is_registry_in_list = (
            registry_relation == "in"
            and operation == "list"
            and not hull
            and (target_scope in {"both", "registry"} or question_type == "registry_in_list")
        )
        # 伪描述：用户整句当 matchText 时，命中不可信
        bogus_description = bool(
            description
            and any(token in description for token in ("哪些", "有哪些", "在库", "未在库", "先验库", "库船"))
        )

        if matches:
            # mismatch 丢弃；uncertain 仅作灰区候选，match 才算确认命中
            ranked = sorted(
                [m for m in matches if isinstance(m, dict)],
                key=lambda m: float(m.get("embeddingScore") or m.get("score") or 0),
                reverse=True,
            )
            confirmed = [m for m in ranked if str(m.get("scoreBand") or "") == "match"]
            uncertain = [m for m in ranked if str(m.get("scoreBand") or "") == "uncertain"]
            supported = confirmed or uncertain  # 展示优先确认，否则灰区
            # 展示轨迹必须按匹配分排序，禁止用 getTrack 全量列表的 1,2,3 顺序盖住
            hit_tracks = self._tracks_ranked_by_matches(supported, tracks)
            if supported and not (is_registry_in_list and bogus_description):
                # 先验库描述匹配：展示命中库项，不把整库当结果
                if target_scope == "registry" and not hit_tracks:
                    matched_items = self._registry_items_from_matches(supported)
                    return self._finish(
                        f"找到 {len(matched_items) or len(supported)} 个匹配库项" if matched_items or supported else "找到匹配目标",
                        [],
                        answer_hint or f"描述「{description}」命中 {len(supported)} 条候选",
                        state if state in {"sufficient", "uncertain", "conflict"} else "sufficient",
                        extra={
                            "matches": ranked,
                            "registryItems": matched_items or registry_items,
                            "planMode": "langgraph",
                            "targetScope": "registry",
                            "matchCount": len(supported),
                            "confirmedMatchCount": len(confirmed),
                        },
                        display={"tracks": [], "includeClips": False, "includeRegistry": True},
                    )
                # 在库列表 + matchImage 命中：按匹配轨迹/库项列名单
                if is_registry_in_list:
                    hit_items = self._registry_items_from_matches(supported)
                    label = (
                        f"视频中确认 {len(confirmed)} 个在库匹配"
                        if confirmed
                        else f"视频中疑似 {len(uncertain)} 个在库匹配（相似度未达确认阈值）"
                    )
                    return self._finish(
                        label + (f"（{len(hit_tracks)} 条轨迹）" if hit_tracks else ""),
                        hit_tracks[: self.display_limit],
                        answer_hint or "listRegistry + matchImage 对照完成",
                        state if state in {"sufficient", "uncertain", "conflict"} else "sufficient",
                        extra={
                            "matches": ranked,
                            "registryItems": hit_items or registry_items,
                            "planMode": "langgraph",
                            "targetScope": "both",
                            "matchCount": len(supported),
                            "confirmedMatchCount": len(confirmed),
                            "found": bool(confirmed),
                        },
                        display={
                            "tracks": hit_tracks[: self.display_limit],
                            "includeClips": True,
                            "includeRegistry": True,
                        },
                    )
                if confirmed:
                    conclusion = "找到匹配目标"
                    finish_state = state if state in {"sufficient", "uncertain", "conflict"} else "sufficient"
                    hint = answer_hint or f"共 {len(confirmed)} 条确认匹配（scoreBand=match）"
                else:
                    conclusion = "仅有灰区匹配，未达确认阈值"
                    finish_state = "uncertain" if operation == "existence" else state
                    hint = answer_hint or f"共 {len(uncertain)} 条 uncertain，最高分未达 match 阈值"
                return self._finish(
                    conclusion,
                    hit_tracks[: self.display_limit],
                    hint,
                    finish_state,
                    extra={
                        "matches": ranked,
                        "planMode": "langgraph",
                        "matchCount": len(supported),
                        "confirmedMatchCount": len(confirmed),
                        "found": bool(confirmed),
                    },
                    display={"tracks": hit_tracks[: self.display_limit], "includeClips": True},
                )
            # 有打分结果但无一达到 match/uncertain：仍按分数展示 top 候选作证据，禁止证据区空白
            candidate_tracks = self._tracks_ranked_by_matches(ranked, tracks)[: self.display_limit]
            top_score = float(ranked[0].get("embeddingScore") or ranked[0].get("score") or 0) if ranked else 0.0
            if is_registry_in_list:
                return self._finish(
                    "未在视频中确认在库船舶匹配",
                    candidate_tracks,
                    answer_hint or (
                        f"共 {len(ranked)} 条打分结果均为 mismatch（最高分 {top_score:.3f}）；"
                        "已附 top 候选供对照，勿视为确认命中"
                    ),
                    "sufficient" if state in {"sufficient", "uncertain"} else state,
                    extra={
                        "matches": ranked,
                        "registryItems": registry_items,
                        "planMode": "langgraph",
                        "targetScope": "both",
                        "matchCount": 0,
                        "candidateCount": len(ranked),
                        "found": False,
                    },
                    display={
                        "tracks": candidate_tracks,
                        "includeClips": bool(candidate_tracks),
                        "includeRegistry": bool(registry_items),
                    },
                )
            no_match_conclusion = "未找到确认匹配目标"
            if operation == "existence":
                label = hull or description or "目标"
                if registry_items:
                    no_match_conclusion = f"未在视频中确认「{label}」（先验库有记录，视觉匹配未达阈值）"
                else:
                    no_match_conclusion = f"未确认符合条件的目标（{label}）"
            return self._finish(
                no_match_conclusion,
                candidate_tracks,
                answer_hint or (
                    f"匹配结果均为 mismatch 或空（共 {len(ranked)} 条打分，最高分 {top_score:.3f}）；"
                    "已展示 top 候选证据，非确认命中"
                ),
                "sufficient" if operation == "existence" else state,
                extra={
                    "matches": ranked,
                    "registryItems": registry_items,
                    "planMode": "langgraph",
                    "matchCount": 0,
                    "candidateCount": len(ranked),
                    "found": False,
                },
                display={
                    "tracks": candidate_tracks,
                    "includeClips": bool(candidate_tracks),
                    "includeRegistry": bool(registry_items),
                },
            )
        if registry_items and not tracks:
            if not self.meta.get("targetScope"):
                self.meta["targetScope"] = "registry"
            # 有描述却没有 match 结果时：说明只 list 了库，未完成筛选
            if description and target_kind == "description":
                return self._finish(
                    "先验库已列出但未完成描述筛选",
                    [],
                    answer_hint or f"已读取 {len(registry_items)} 个库项，缺少 matchText 筛选结果",
                    "uncertain" if state == "sufficient" else state,
                    extra={
                        "registryItems": registry_items,
                        "planMode": "langgraph",
                        "targetScope": self.meta.get("targetScope") or "registry",
                    },
                    display={"tracks": [], "includeClips": False, "includeRegistry": True},
                )
            # 舷号：库中有记录，但视频轨迹/视觉匹配未命中
            if operation == "existence" and hull:
                labels = "、".join(
                    str(item.get("hullNumber") or item.get("registryId") or "")
                    for item in registry_items[:3]
                    if item.get("hullNumber") or item.get("registryId")
                )
                return self._finish(
                    f"未在视频中发现「{hull}」" + (f"（先验库有：{labels}）" if labels else "（先验库有记录）"),
                    [],
                    answer_hint or f"getTrack 无舷号命中；先验库命中 {len(registry_items)} 项，视觉匹配未给出视频轨迹",
                    "sufficient" if state in {"sufficient", "uncertain"} else state,
                    extra={
                        "registryItems": registry_items,
                        "planMode": "langgraph",
                        "targetScope": "both",
                    },
                    display={"tracks": [], "includeClips": False, "includeRegistry": True},
                )
            return self._finish(
                f"先验库共 {len(registry_items)} 个库项" if not description else "已查询先验库",
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
            # 存在判断 + 舷号：全量扫轨得到的轨迹 ≠ 该舷号已确认出现
            # 仅当有非 mismatch 的视觉/文本匹配，或轨迹自身带该舷号时，才「确认出现」
            if operation == "existence" and hull:
                hull_on_track = any(
                    str(t.get("hullNumber") or "").upper() == hull.upper()
                    or str(t.get("finalHullNumber") or "").upper() == hull.upper()
                    for t in tracks
                    if isinstance(t, dict)
                )
                if not hull_on_track:
                    labels = "、".join(
                        str(item.get("hullNumber") or item.get("registryId") or "")
                        for item in registry_items[:3]
                        if item.get("hullNumber") or item.get("registryId")
                    ) if registry_items else ""
                    return self._finish(
                        f"未在视频中确认「{hull}」"
                        + (f"（先验库有：{labels}；全量检索 {len(tracks)} 条轨迹均未标此舷号/未视觉命中）" if labels
                           else f"（全量检索 {len(tracks)} 条轨迹，均未标此舷号）"),
                        tracks[: self.display_limit],
                        answer_hint or "放开舷号过滤后的轨迹不能直接当作目标命中",
                        "sufficient" if state in {"sufficient", "uncertain"} else state,
                        extra={
                            "registryItems": registry_items,
                            "planMode": "langgraph",
                            "found": False,
                            "targetScope": "both" if registry_items else target_scope,
                        },
                        display={"tracks": tracks[: self.display_limit], "includeClips": True, "includeRegistry": bool(registry_items)},
                    )
            if operation == "existence":
                conclusion = "确认出现" if state == "sufficient" else "疑似出现（证据不完整）"
            else:
                conclusion = "已定位相关轨迹" if state == "sufficient" else "仅获得部分轨迹证据"
            return self._finish(
                conclusion,
                tracks,
                answer_hint,
                state,
                extra={"planMode": "langgraph"},
                display={"tracks": tracks, "includeClips": True},
            )
        # 0 轨迹/0 匹配：存在判断应给明确否定，而不是「未找到可靠证据」+ 误导标签
        if state == "conflict":
            return self._finish("证据存在冲突", [], answer_hint, state, extra={"planMode": "langgraph"})
        if operation == "existence":
            label = hull or description or "目标"
            return self._finish(
                f"未发现「{label}」",
                [],
                answer_hint or "检索范围内无对应轨迹或匹配",
                "sufficient",
                extra={"planMode": "langgraph", "found": False},
            )
        return self._finish("未找到可靠证据", [], answer_hint, state, extra={"planMode": "langgraph"})

    def _collect_tracks(self) -> list[dict[str, Any]]:
        collected: dict[str, dict[str, Any]] = {}
        for value in self.working_scope.values():
            if not isinstance(value, dict):
                continue
            for track in value.get("tracks") or []:
                if isinstance(track, dict) and track.get("trackId") is not None:
                    collected[str(track["trackId"])] = dict(track)
            for match in value.get("matches") or []:
                if not isinstance(match, dict):
                    continue
                track = match.get("track") if isinstance(match.get("track"), dict) else {}
                track_id = match.get("matchedTrackId") or match.get("trackId") or track.get("trackId")
                if track_id is None:
                    continue
                item = dict(track) if track else {"trackId": track_id}
                item.setdefault("trackId", track_id)
                for key in ("embeddingScore", "scoreBand", "hullNumber", "matchedRegistryId"):
                    if match.get(key) is not None:
                        item[key] = match.get(key)
                prev = collected.get(str(track_id), {})
                # 同轨迹多匹配时保留更高分
                if prev.get("embeddingScore") is not None and item.get("embeddingScore") is not None:
                    if float(item["embeddingScore"]) < float(prev["embeddingScore"]):
                        item = {**item, **{k: prev[k] for k in ("embeddingScore", "scoreBand") if k in prev}}
                collected[str(track_id)] = {**prev, **item}
        items = list(collected.values())
        # 有相似度的排前面，避免展示固定成轨迹编号顺序
        items.sort(
            key=lambda t: (
                0 if t.get("embeddingScore") is not None else 1,
                -float(t.get("embeddingScore") or 0),
                str(t.get("trackId") or ""),
            )
        )
        return items

    def _collect_matches(self) -> list[dict[str, Any]]:
        matches: list[dict[str, Any]] = []
        for value in self.working_scope.values():
            if isinstance(value, dict):
                for match in value.get("matches") or []:
                    if isinstance(match, dict):
                        matches.append(match)
        matches.sort(key=lambda item: float(item.get("embeddingScore") or item.get("score") or 0), reverse=True)
        return matches

    def _tracks_ranked_by_matches(
        self,
        matches: list[dict[str, Any]],
        all_tracks: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """只返回匹配命中的轨迹，按 embeddingScore 降序；用全量轨迹补全元数据。"""
        by_id = {
            str(t.get("trackId")): dict(t)
            for t in (all_tracks or [])
            if isinstance(t, dict) and t.get("trackId") is not None
        }
        # scope 里可能还有更完整的 track 记录
        for value in self.working_scope.values():
            if not isinstance(value, dict):
                continue
            for track in value.get("tracks") or []:
                if isinstance(track, dict) and track.get("trackId") is not None:
                    tid = str(track["trackId"])
                    by_id[tid] = {**by_id.get(tid, {}), **track}
        ranked: list[dict[str, Any]] = []
        seen: set[str] = set()
        ordered = sorted(
            [m for m in matches if isinstance(m, dict)],
            key=lambda m: float(m.get("embeddingScore") or m.get("score") or 0),
            reverse=True,
        )
        for match in ordered:
            track_id = match.get("matchedTrackId") or match.get("trackId")
            if track_id is None:
                continue
            key = str(track_id)
            if key in seen:
                continue
            seen.add(key)
            base = dict(by_id.get(key) or {"trackId": track_id})
            base["trackId"] = track_id
            if match.get("embeddingScore") is not None:
                base["embeddingScore"] = match.get("embeddingScore")
            if match.get("scoreBand") is not None:
                base["scoreBand"] = match.get("scoreBand")
            if match.get("matchedRegistryId") is not None:
                base["matchedRegistryId"] = match.get("matchedRegistryId")
            if match.get("hullNumber") is not None:
                base.setdefault("hullNumber", match.get("hullNumber"))
            # 把匹配关键帧挂到轨迹上，供 _display_tracks 直接用
            kids = match.get("matchedKeyframeIds") or match.get("queryKeyframeIds") or []
            if kids:
                base["matchedKeyframeIds"] = [str(x) for x in kids if x is not None]
                base["keyframeIds"] = list(base["matchedKeyframeIds"])
            refs = match.get("matchedRegistryReferenceIds") or match.get("queryRegistryReferenceIds") or []
            if refs:
                base["matchedRegistryReferenceIds"] = [str(x) for x in refs if x is not None]
            ranked.append(base)
        return ranked

    def _registry_items_from_matches(self, matches: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """从 matchText/matchImage 的库侧命中还原 registryItems，供前端只展示命中项。"""
        by_id: dict[str, dict[str, Any]] = {}
        all_items = {str(item.get("registryId")): item for item in self._collect_registry() if item.get("registryId")}
        for match in matches:
            if not isinstance(match, dict):
                continue
            rid = str(match.get("matchedRegistryId") or match.get("registryId") or "")
            if not rid:
                continue
            base = dict(all_items.get(rid) or {})
            base.setdefault("registryId", rid)
            if match.get("hullNumber"):
                base.setdefault("hullNumber", match.get("hullNumber"))
            if match.get("embeddingScore") is not None:
                base["embeddingScore"] = match.get("embeddingScore")
            if match.get("scoreBand"):
                base["scoreBand"] = match.get("scoreBand")
            by_id[rid] = base
        return list(by_id.values())

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
        # 供 _display_tracks 优先使用的命中库项（避免展示整库）
        if extra and isinstance(extra.get("registryItems"), list):
            self._pending_registry_items = list(extra["registryItems"])
        else:
            self._pending_registry_items = []
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
            # 仅先验库结果：优先用本轮合成得到的 registry 命中项，再回退 working_scope
            reference_ids: list[str] = []
            preferred_items: list[dict[str, Any]] = []
            if include_registry:
                preferred_items = list(self._pending_registry_items or [])
                sources = [preferred_items] if preferred_items else []
                if not sources:
                    for value in self.working_scope.values():
                        if isinstance(value, dict) and value.get("registryItems"):
                            sources.append(value.get("registryItems") or [])
                for value in self.working_scope.values():
                    if not isinstance(value, dict):
                        continue
                    for key in ("registryReferenceIds", "shownRegistryReferenceIds"):
                        for item in value.get(key) or []:
                            if item is not None:
                                reference_ids.append(str(item))
                    for item in (preferred_items or value.get("registryItems") or []):
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
                # 有命中库项但无参考图 ID 时，仍按库项生成展示组
                if preferred_items and not reference_ids:
                    for item in preferred_items:
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
                "registryItemCount": len(preferred_items) if preferred_items else 0,
            }
            if reference_ids:
                self.display_groups.append({
                    "trackId": None,
                    "keyframeIds": [],
                    "shipSegmentIds": [],
                    "registryReferenceIds": reference_ids[:12],
                })
            elif preferred_items:
                # 无向量参考图时仍把库项 ID 放进 evidence，前端可按 registryItems 渲染
                self.display_groups.append({
                    "trackId": None,
                    "keyframeIds": [],
                    "shipSegmentIds": [],
                    "registryReferenceIds": [],
                    "registryItems": preferred_items[:12],
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
            # 匹配结果常只有 trackId+分数，从 working_scope.matches 补关键帧
            if not keyframe_ids:
                for value in self.working_scope.values():
                    if not isinstance(value, dict):
                        continue
                    for match in value.get("matches") or []:
                        if not isinstance(match, dict):
                            continue
                        mid = str(match.get("matchedTrackId") or match.get("trackId") or "")
                        if mid != track_id:
                            continue
                        for kid in match.get("matchedKeyframeIds") or match.get("queryKeyframeIds") or []:
                            if kid is not None:
                                keyframe_ids.append(str(kid))
                        if match.get("embeddingScore") is not None and track.get("embeddingScore") is None:
                            track["embeddingScore"] = match.get("embeddingScore")
                        if match.get("scoreBand") is not None and track.get("scoreBand") is None:
                            track["scoreBand"] = match.get("scoreBand")
                keyframe_ids = list(dict.fromkeys(keyframe_ids))
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
                if not raw_refs:
                    for value in self.working_scope.values():
                        if not isinstance(value, dict):
                            continue
                        for match in value.get("matches") or []:
                            if not isinstance(match, dict):
                                continue
                            mid = str(match.get("matchedTrackId") or match.get("trackId") or "")
                            if mid != track_id:
                                continue
                            for rid in match.get("matchedRegistryReferenceIds") or match.get("queryRegistryReferenceIds") or []:
                                if rid is not None:
                                    raw_refs.append(str(rid))
                if hasattr(self.tools, "_representative_registry_reference_ids"):
                    try:
                        raw_refs = self.tools._representative_registry_reference_ids(raw_refs)
                    except Exception:
                        pass
                reference_ids = list(dict.fromkeys(str(x) for x in raw_refs if x is not None))[:3]
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
