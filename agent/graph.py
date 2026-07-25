"""LangGraph 四 Agent 编排：角色工具集 → handoff → Graph。

流程（对齐 old 自主规划）：
  IntentAgent ⇄ tools → handoff_to_plan
  PlanAgent ⇄ tools → handoff_to_observe(calls+$ref) | handoff_to_reflect
  ObserveAgent = 确定性执行 calls（完整结果进 working_scope，模型只看摘要）→ reflect
  ReflectAgent ⇄ tools → handoff_to_plan_replan | handoff_finish
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Annotated, Any, Callable, Literal, TypedDict

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command
from pydantic import BaseModel, Field

from services import AgentLLMService
from tools import ToolService
from tools.target_parser import infer_intent_fields, normalize_target_items

from .lc_tools import build_intent_tools, build_load_skill_tool
from .llm_adapter import build_chat_model
from .plan_executor import PlanExecutor
from .roles import (
    INTENT_RESPONSIBILITY,
    OBSERVE_RESPONSIBILITY,
    PLAN_RESPONSIBILITY,
    REFLECT_RESPONSIBILITY,
    role_system_prompt,
)
from .skill_loader import load_skill_body


def _merge_dict(left: dict[str, Any] | None, right: dict[str, Any] | None) -> dict[str, Any]:
    result = dict(left or {})
    if right:
        result.update(right)
    return result


def _merge_list(left: list[Any] | None, right: list[Any] | None) -> list[Any]:
    return list(left or []) + list(right or [])


class AgentState(TypedDict, total=False):
    # 跨节点只共享结构化字段；各角色内层 ReAct 的 messages 不写入图状态
    question: str
    intent: dict[str, Any]
    working_scope: Annotated[dict[str, Any], _merge_dict]
    reflection: dict[str, Any]
    rounds: Annotated[list[dict[str, Any]], _merge_list]
    tool_chain: Annotated[list[str], _merge_list]
    tool_records: Annotated[list[dict[str, Any]], _merge_list]
    plan_hint: str
    plan_calls: list[dict[str, Any]]
    observation_summary: str
    active_agent: str
    final_state: str
    final_reason: str
    loop_count: int
    max_rounds: int
    query_top_k: int
    broad_match_top_k: int
    error: str


class HandoffToPlanArgs(BaseModel):
    intent: dict[str, Any] = Field(default_factory=dict, description="结构化意图规格")
    note: str = Field(default="", description="给 PlanAgent 的备注")


class HandoffToObserveArgs(BaseModel):
    goal: str = Field(description="本轮观察目标")
    calls: list[dict[str, Any]] = Field(
        default_factory=list,
        description="本轮可执行工具调用列表；每项含 id/tool/arguments，跨步骤用 $ref",
    )
    planHint: str = Field(default="", description="自然语言计划说明，供前端展示")
    reason: str = Field(default="")


class HandoffToReflectArgs(BaseModel):
    summary: str = Field(description="观察/规划摘要")
    evidenceGap: str = Field(default="")
    proposedState: str = Field(default="replan")


class HandoffFinishArgs(BaseModel):
    state: Literal["sufficient", "conflict", "uncertain"] = Field(description="最终状态")
    reason: str = Field(description="结束依据")
    answerHint: str = Field(default="")


class HandoffReplanArgs(BaseModel):
    reason: str = Field(description="为何 replan")
    nextAction: str = Field(default="", description="给 Plan 的下一步")
    evidenceGap: str = Field(default="")


def _safe_json(text: str) -> dict[str, Any]:
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else {"value": value}
    except Exception:
        return {"raw": text}


def _content_parts(content: Any) -> tuple[str, str]:
    """从消息 content 拆出 (可见正文, 思考/推理)。"""
    if content is None:
        return "", ""
    if isinstance(content, str):
        text = content.strip()
        # 兼容 <think>…</think> 与纯正文
        think_bits = re.findall(r"<think>(.*?)</think>", text, flags=re.S | re.I)
        if think_bits:
            cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.S | re.I).strip()
            return cleaned, "\n".join(bit.strip() for bit in think_bits if bit.strip())
        return text, ""
    if isinstance(content, list):
        texts: list[str] = []
        thinks: list[str] = []
        for item in content:
            if isinstance(item, dict):
                kind = str(item.get("type") or "")
                if kind in {"thinking", "reasoning", "reason"}:
                    thinks.append(str(item.get("thinking") or item.get("text") or item.get("content") or ""))
                elif kind == "text":
                    texts.append(str(item.get("text") or ""))
                else:
                    texts.append(str(item.get("text") or item.get("content") or ""))
            else:
                texts.append(str(item))
        body, nested = _content_parts("\n".join(texts))
        thinking = "\n".join([*(t for t in thinks if t.strip()), nested]).strip()
        return body, thinking
    return str(content).strip(), ""


def _last_ai_text(messages: list[Any]) -> str:
    for message in reversed(messages or []):
        if isinstance(message, AIMessage):
            body, _ = _content_parts(message.content)
            if body:
                return body
    return ""


def _last_ai_thinking(messages: list[Any]) -> str:
    chunks: list[str] = []
    for message in messages or []:
        if isinstance(message, AIMessage):
            _, thinking = _content_parts(message.content)
            if thinking:
                chunks.append(thinking)
            # 部分适配器把推理放在 additional_kwargs
            extra = getattr(message, "additional_kwargs", None) or {}
            for key in ("reasoning_content", "thinking", "reasoning"):
                value = extra.get(key)
                if value:
                    chunks.append(str(value))
    return "\n".join(chunk.strip() for chunk in chunks if chunk and str(chunk).strip()).strip()


def _emit(handler: Callable[[dict[str, Any]], None] | None, event: dict[str, Any]) -> None:
    if not handler:
        return
    try:
        handler(event)
    except Exception:
        pass


def _tool_summary(name: str, payload: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key in (
        "trackCount",
        "returnedTrackCount",
        "totalTrackCount",
        "keyframeCount",
        "matchCount",
        "registryItemCount",
        "registryReferenceCount",
        "exactMatchHullCount",
        "highThresholdShipCount",
        "lowThresholdShipCount",
        "shipSegmentId",
        "found",
        "searchable",
        "decision",
        "hasMore",
        "error",
    ):
        if payload.get(key) is not None:
            summary[key] = payload.get(key)
    # registryCount 仅在下方按 registryItems 优先计算，避免被参考图数覆盖
    if payload.get("tracks") is not None and summary.get("trackCount") is None:
        try:
            summary["trackCount"] = len(payload.get("tracks") or [])
        except Exception:
            pass
    if payload.get("matches") is not None and summary.get("matchCount") is None:
        try:
            summary["matchCount"] = len(payload.get("matches") or [])
        except Exception:
            pass
    if payload.get("keyframes") is not None and summary.get("keyframeCount") is None:
        try:
            summary["keyframeCount"] = len(payload.get("keyframes") or [])
        except Exception:
            pass
    if payload.get("registryItems") is not None:
        try:
            summary["registryCount"] = len(payload.get("registryItems") or [])
            summary["registryItemCount"] = summary["registryCount"]
        except Exception:
            pass
    if payload.get("registryReferences") is not None:
        try:
            summary["registryReferenceCount"] = len(payload.get("registryReferences") or [])
            if summary.get("registryCount") is None:
                summary["registryCount"] = summary["registryReferenceCount"]
        except Exception:
            pass
    if payload.get("matchedHullNumbers") is not None and summary.get("exactMatchHullCount") is None:
        try:
            summary["exactMatchHullCount"] = len(payload.get("matchedHullNumbers") or [])
        except Exception:
            pass
    if not summary:
        summary["tool"] = name
        summary["ok"] = payload.get("ok") is not False
    return summary


def _normalize_broad_match_top_k(value: int | None) -> int:
    """规范化广泛库图匹配上限；0 表示不截断。"""
    try:
        return max(0, int(value if value is not None else 0))
    except (TypeError, ValueError):
        return 0


def _default_plan_calls(
    intent: dict[str, Any],
    top_k: int,
    broad_match_top_k: int = 0,
    *,
    replan_hint: str = "",
) -> list[dict[str, Any]]:
    """模型未给出 calls 时的最小可执行链（结构化 $ref，不是业务硬编码分支表）。"""
    hull = str(intent.get("hullNumber") or "").strip()
    description = str(intent.get("description") or "").strip()
    time_range = intent.get("timeRange")
    operation = str(intent.get("operation") or "list")
    target_scope = str(intent.get("targetScope") or "track_memory")
    top = max(1, min(20, int(top_k or 3)))
    broad_top = _normalize_broad_match_top_k(broad_match_top_k)
    hint_raw = str(replan_hint or intent.get("nextAgentFocus") or "")
    hint = hint_raw.lower()
    wants_registry = (
        target_scope in {"registry", "both"}
        or str(intent.get("registryRelation") or "any") in {"in", "out"}
        or any(token in hint for token in ("先验库", "在库", "未在库", "getregistry", "listregistry", "matchhull", "registry"))
    )
    # 视觉匹配：库参考图 ↔ 视频关键帧（舷号 OCR 未命中 / 在库列表）
    wants_visual_match = any(
        token in hint
        for token in (
            "matchimage", "match_image", "视觉匹配", "图像匹配", "图匹配",
            "registryreferences", "关键帧匹配", "库图", "对照视频",
        )
    ) or ("match" in hint and "image" in hint)
    registry_relation = str(intent.get("registryRelation") or "any")
    # 「有哪些在库船出现」：list + in + both/all，禁止当描述 matchText
    wants_registry_in_list = (
        registry_relation == "in"
        and operation == "list"
        and not hull
        and (
            target_scope in {"both", "registry"}
            or str(intent.get("questionType") or "") == "registry_in_list"
            or any(token in hint for token in ("listregistry", "在库", "哪些", "matchimage", "matchhull"))
        )
    )
    # 伪描述：整句问法残留，不当 matchText description
    bogus_description = bool(
        description
        and (
            any(token in description for token in ("哪些", "在库", "先验库", "有哪些", "未在库"))
            or description in {"船", "船舶", "船只", "目标"}
        )
    )
    if bogus_description:
        description = ""

    def _track_args(*, with_hull: bool, all_tracks: bool = False) -> dict[str, Any]:
        args: dict[str, Any] = {"offset": 0, "limit": 0 if all_tracks else 60}
        if time_range:
            args["timeRange"] = time_range
        if with_hull and hull:
            args["hullNumber"] = hull
        return args

    def _match_image_from_list_registry() -> list[dict[str, Any]]:
        return [
            {"id": "registry", "tool": "listRegistry", "arguments": {}},
            {"id": "tracks", "tool": "getTrack", "arguments": _track_args(with_hull=False, all_tracks=True)},
            {"id": "frames", "tool": "getFrames", "arguments": {"trackIds": {"$ref": "tracks.trackIds"}}},
            {
                "id": "match",
                "tool": "matchImage",
                "arguments": {
                    "queryImages": {"$ref": "registry.registryReferences"},
                    "galleryImages": {"$ref": "frames.keyframes"},
                    "topK": broad_top,
                },
            },
        ]

    # 在库船列表（视频中出现的库船）：listRegistry → getTrack → getFrames → matchImage
    if wants_registry_in_list or (wants_visual_match and not hull and wants_registry and not description):
        return _match_image_from_list_registry()

    # 已有/待查先验库后做视觉匹配：不带舷号扫视频轨迹 → 关键帧 → matchImage
    if wants_visual_match and hull:
        return [
            {"id": "registry", "tool": "getRegistry", "arguments": {"hullNumber": hull}},
            {"id": "tracks", "tool": "getTrack", "arguments": _track_args(with_hull=False)},
            {"id": "frames", "tool": "getFrames", "arguments": {"trackIds": {"$ref": "tracks.trackIds"}}},
            {
                "id": "match",
                "tool": "matchImage",
                "arguments": {
                    # 只引用参考图列表；执行器会再从 registryItems.references 展开补齐
                    "queryImages": {"$ref": "registry.registryReferences"},
                    "galleryImages": {"$ref": "frames.keyframes"},
                    "topK": top,
                },
            },
        ]
    if wants_visual_match and description:
        return [
            {"id": "tracks", "tool": "getTrack", "arguments": _track_args(with_hull=False)},
            {"id": "frames", "tool": "getFrames", "arguments": {"trackIds": {"$ref": "tracks.trackIds"}}},
            {
                "id": "match",
                "tool": "matchText",
                "arguments": {
                    "description": description,
                    "galleryImages": {"$ref": "frames.keyframes"},
                    "topK": top,
                },
            },
        ]

    if target_scope == "registry" or (wants_registry and "gettrack" not in hint and operation != "count" and not wants_visual_match and not wants_registry_in_list):
        if hull:
            # 查库后仍可能需要视觉匹配；若 hint 只写查库则先 getRegistry
            return [{"id": "registry", "tool": "getRegistry", "arguments": {"hullNumber": hull}}]
        if description:
            return [
                {"id": "registry", "tool": "listRegistry", "arguments": {}},
                {
                    "id": "match",
                    "tool": "matchText",
                    "arguments": {
                        "description": description,
                        "galleryImages": {
                            "$ref": "registry.registryReferences",
                            "$default": {"$ref": "registry.registryItems"},
                        },
                        "topK": top,
                    },
                },
            ]
        # 无描述的在库关系：给完整对照链，避免只 list 库
        if registry_relation in {"in", "out"}:
            return _match_image_from_list_registry()
        return [{"id": "registry", "tool": "listRegistry", "arguments": {}}]

    # replan 明确要求查库（尚未要求视觉匹配）
    if wants_registry and any(token in hint for token in ("先验库", "getregistry", "listregistry", "matchhull", "在库")):
        if hull:
            return [
                {"id": "registry", "tool": "getRegistry", "arguments": {"hullNumber": hull}},
                {"id": "matchHull", "tool": "matchHull", "arguments": {"hullNumberArray": [hull]}},
            ]
        if registry_relation in {"in", "out"} or wants_registry_in_list:
            return _match_image_from_list_registry()
        return [{"id": "registry", "tool": "listRegistry", "arguments": {}}]

    track_args = _track_args(with_hull=bool(hull))
    calls: list[dict[str, Any]] = [
        {"id": "tracks", "tool": "getTrack", "arguments": track_args},
    ]
    if operation == "count":
        calls.append({"id": "frames", "tool": "getFrames", "arguments": {"trackIds": {"$ref": "tracks.trackIds"}}})
        calls.append({
            "id": "dedup",
            "tool": "dedupTracks",
            "arguments": {
                "tracks": {"$ref": "tracks.tracks"},
                "keyframesByTrack": {"$ref": "frames.keyframesByTrack"},
            },
        })
        return calls
    if description:
        calls.append({"id": "frames", "tool": "getFrames", "arguments": {"trackIds": {"$ref": "tracks.trackIds"}}})
        calls.append({
            "id": "match",
            "tool": "matchText",
            "arguments": {
                "description": description,
                "galleryImages": {"$ref": "frames.keyframes"},
                "topK": top,
            },
        })
        return calls
    if hull:
        calls.append({"id": "frames", "tool": "getFrames", "arguments": {"trackIds": {"$ref": "tracks.trackIds"}}})
        return calls
    # both + in 且无描述：默认在库对照视觉链
    if wants_registry and registry_relation == "in":
        return _match_image_from_list_registry()
    calls.append({"id": "frames", "tool": "getFrames", "arguments": {"trackIds": {"$ref": "tracks.trackIds"}}})
    return calls


def _apply_retrieval_limits(
    calls: list[dict[str, Any]],
    *,
    broad_match_top_k: int,
) -> list[dict[str, Any]]:
    """修正模型计划中的广泛库图匹配参数，避免退回普通检索上限。"""
    has_list_registry = False
    has_match_image = False
    has_unfiltered_track = False
    for call in calls:
        if not isinstance(call, dict):
            continue
        tool_name = str(call.get("tool") or "")
        arguments = call.get("arguments") if isinstance(call.get("arguments"), dict) else {}
        if tool_name == "listRegistry":
            has_list_registry = True
        elif tool_name == "matchImage":
            has_match_image = True
        elif tool_name == "getTrack" and not str(arguments.get("hullNumber") or "").strip():
            has_unfiltered_track = True

    # 只有“全库 + 全轨迹 + 图像匹配”才属于广泛匹配；指定舷号的单库匹配仍沿用普通 topK。
    if not (has_list_registry and has_unfiltered_track and has_match_image):
        return calls

    broad_top = _normalize_broad_match_top_k(broad_match_top_k)
    normalized: list[dict[str, Any]] = []
    for call in calls:
        if not isinstance(call, dict):
            continue
        item = dict(call)
        arguments = dict(item.get("arguments") or {})
        tool_name = str(item.get("tool") or "")
        if tool_name == "getTrack" and not str(arguments.get("hullNumber") or "").strip():
            arguments["limit"] = 0
        elif tool_name == "matchImage":
            arguments["topK"] = broad_top
        item["arguments"] = arguments
        normalized.append(item)
    return normalized


def build_sea_agent_graph(
    llm: AgentLLMService,
    tools: ToolService,
    *,
    max_rounds: int = 3,
    query_top_k: int | None = None,
    broad_match_top_k: int | None = None,
    event_handler: Callable[[dict[str, Any]], None] | None = None,
):
    """编译四 Agent LangGraph。"""
    model = build_chat_model(llm)
    reference_time = datetime.now().astimezone()
    default_top_k = int(query_top_k or 3)
    default_broad_match_top_k = _normalize_broad_match_top_k(broad_match_top_k)

    @tool("handoff_to_plan", args_schema=HandoffToPlanArgs)
    def handoff_to_plan(intent: dict[str, Any] | None = None, note: str = "") -> str:
        """意图完成后移交给 PlanAgent。"""
        return json.dumps({"ok": True, "handoff": "plan", "intent": intent or {}, "note": note}, ensure_ascii=False)

    @tool("handoff_to_observe", args_schema=HandoffToObserveArgs)
    def handoff_to_observe(
        goal: str,
        calls: list[dict[str, Any]] | None = None,
        planHint: str = "",
        reason: str = "",
    ) -> str:
        """规划完成后移交给 ObserveAgent 按 calls 确定性执行。"""
        return json.dumps(
            {
                "ok": True,
                "handoff": "observe",
                "goal": goal,
                "calls": calls or [],
                "planHint": planHint,
                "reason": reason,
            },
            ensure_ascii=False,
        )

    @tool("handoff_to_reflect", args_schema=HandoffToReflectArgs)
    def handoff_to_reflect(summary: str, evidenceGap: str = "", proposedState: str = "replan") -> str:
        """观察或规划后移交给 ReflectAgent。"""
        return json.dumps(
            {
                "ok": True,
                "handoff": "reflect",
                "summary": summary,
                "evidenceGap": evidenceGap,
                "proposedState": proposedState,
            },
            ensure_ascii=False,
        )

    @tool("handoff_finish", args_schema=HandoffFinishArgs)
    def handoff_finish(state: str, reason: str, answerHint: str = "") -> str:
        """证据充分或应结束时退出循环。"""
        return json.dumps(
            {"ok": True, "handoff": "finish", "state": state, "reason": reason, "answerHint": answerHint},
            ensure_ascii=False,
        )

    @tool("handoff_to_plan_replan", args_schema=HandoffReplanArgs)
    def handoff_to_plan_replan(reason: str, nextAction: str = "", evidenceGap: str = "") -> str:
        """Reflect 判定 replan，交回 PlanAgent。"""
        return json.dumps(
            {
                "ok": True,
                "handoff": "plan",
                "replan": True,
                "reason": reason,
                "nextAction": nextAction,
                "evidenceGap": evidenceGap,
            },
            ensure_ascii=False,
        )

    def _skill_loader(agent_key: str):
        def _load(skill_id: str) -> dict[str, Any]:
            body = load_skill_body(agent_key, skill_id)
            if not body:
                return {"ok": False, "error": f"unknown_skill:{skill_id}"}
            return {"ok": True, "skillId": skill_id, "content": body}

        return _load

    def _tool_event(name: str, arguments: dict[str, Any], result: dict[str, Any], *, round_number: int) -> None:
        call_id = f"{name}-{round_number}-{abs(hash(json.dumps(arguments, ensure_ascii=False, sort_keys=True, default=str))) % 10_000}"
        summary = _tool_summary(name, result if isinstance(result, dict) else {})
        ok = result.get("ok") is not False if isinstance(result, dict) else False
        base = {
            "type": "agent_tool",
            "title": "ObserveAgent",
            "message": name,
            "role": "observer",
            "round": round_number,
            "id": call_id,
            "tool": name,
            "arguments": arguments,
            "ok": ok,
            "status": "completed" if ok else "failed",
            "phase": "completed" if ok else "failed",
            "summary": summary,
            "error": None if ok else (result.get("error") if isinstance(result, dict) else "tool_failed"),
            **summary,
        }
        _emit(event_handler, {**base, "phase": "running", "status": "running", "ok": True, "error": None})
        _emit(event_handler, base)

    intent_tools = build_intent_tools(reference_time) + [
        build_load_skill_tool("intent_agent", _skill_loader("intent_agent")),
        handoff_to_plan,
    ]
    # Plan：core+tool_chains 已 always；保留 loadSkill 供可选 recovery/acceptance 自主补载（限 1 次）
    plan_tools = [
        build_load_skill_tool("plan_agent", _skill_loader("plan_agent")),
        handoff_to_observe,
        handoff_to_reflect,
    ]

    def _run_agent(
        name: str,
        agent_key: str,
        title: str,
        responsibility: str,
        agent_tools: list[Any],
        state: AgentState,
        user_content: str,
        *,
        role: str | None = None,
        round_number: int = 0,
        recursion_limit: int = 12,
    ) -> dict[str, Any]:
        prompt, skill_ids = role_system_prompt(
            agent_key,
            title,
            responsibility,
            context={
                "question": state.get("question"),
                "intent": state.get("intent"),
                "plan_hint": state.get("plan_hint"),
                "observation_summary": state.get("observation_summary"),
                "evidenceGap": (state.get("reflection") or {}).get("evidenceGap"),
            },
        )
        event_round = 0 if role == "intent" else max(1, round_number or 1)
        _emit(
            event_handler,
            {
                "type": "status",
                "title": title,
                "message": f"{title} 开始（skills: {', '.join(skill_ids) or 'core'}）",
                "enabledSkills": skill_ids,
                "role": role,
                "round": event_round,
            },
        )
        if role:
            _emit(
                event_handler,
                {
                    "type": "agent_start",
                    "title": title,
                    "message": f"{title} 开始",
                    "role": role,
                    "round": event_round,
                },
            )

        agent = create_agent(
            model,
            agent_tools,
            system_prompt=prompt,
            name=name,
        )
        invoke_error = ""
        messages: list[Any] = []
        streamed_thinking = ""
        streamed_text = ""

        def _emit_delta(delta: str, *, kind: str = "thinking") -> None:
            if not delta or not role:
                return
            _emit(
                event_handler,
                {
                    "type": "agent_delta",
                    "title": title,
                    "message": "",
                    "role": role,
                    "round": event_round,
                    "kind": kind,
                    "delta": delta,
                },
            )

        try:
            # messages：边生成边推思考；values：拿最终 messages，避免二次 invoke 拖垮超时
            final_values: dict[str, Any] | None = None
            try:
                stream = agent.stream(
                    {"messages": [HumanMessage(content=user_content)]},
                    config={"recursion_limit": recursion_limit},
                    stream_mode=["messages", "values"],
                )
            except TypeError:
                # 旧版 langgraph 不支持 list stream_mode
                stream = agent.stream(
                    {"messages": [HumanMessage(content=user_content)]},
                    config={"recursion_limit": recursion_limit},
                    stream_mode="messages",
                )
            for item in stream:
                mode = None
                data: Any = item
                if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str):
                    mode, data = item[0], item[1]
                if mode == "values" or (
                    mode is None
                    and isinstance(data, dict)
                    and "messages" in data
                    and not isinstance(data.get("messages"), tuple)
                ):
                    if isinstance(data, dict):
                        final_values = data
                    continue
                # messages 模式：data 多为 (message, metadata)
                message = data[0] if isinstance(data, tuple) and data else data
                if not isinstance(message, (AIMessageChunk, AIMessage, ToolMessage, HumanMessage)):
                    continue
                # 完整消息（非 chunk）直接入列，保证 handoff ToolMessage 可解析
                if isinstance(message, (AIMessage, ToolMessage, HumanMessage)) and not isinstance(message, AIMessageChunk):
                    messages.append(message)
                if isinstance(message, (AIMessageChunk, AIMessage)):
                    body, thinking = _content_parts(getattr(message, "content", None))
                    extra = getattr(message, "additional_kwargs", None) or {}
                    for key in ("reasoning_content", "thinking", "reasoning"):
                        if extra.get(key):
                            thinking = f"{thinking}\n{extra.get(key)}".strip() if thinking else str(extra.get(key))
                    response_meta = getattr(message, "response_metadata", None) or {}
                    for key in ("reasoning_content", "thinking", "reasoning"):
                        if response_meta.get(key):
                            thinking = f"{thinking}\n{response_meta.get(key)}".strip() if thinking else str(response_meta.get(key))
                    if thinking:
                        piece = thinking
                        if streamed_thinking and thinking.startswith(streamed_thinking):
                            piece = thinking[len(streamed_thinking):]
                        elif streamed_thinking and streamed_thinking.endswith(thinking):
                            piece = ""
                        if piece:
                            streamed_thinking += piece
                            _emit_delta(piece, kind="thinking")
                    if body and isinstance(message, AIMessageChunk):
                        piece = body
                        if streamed_text and body.startswith(streamed_text):
                            piece = body[len(streamed_text):]
                        if piece:
                            streamed_text += piece
                            _emit_delta(piece, kind="token")
            if final_values and isinstance(final_values.get("messages"), list):
                # values 含完整对话，优先于边收边攒的 messages
                messages = list(final_values.get("messages") or [])
            elif not messages:
                # 仅当流式完全没拿到状态时才 invoke 一次（不再双重重试）
                result = agent.invoke(
                    {"messages": [HumanMessage(content=user_content)]},
                    config={"recursion_limit": recursion_limit},
                )
                messages = result.get("messages") or []
        except Exception as error:
            invoke_error = str(error)
            if not messages:
                # 流式失败时只做一次非流式兜底
                try:
                    result = agent.invoke(
                        {"messages": [HumanMessage(content=user_content)]},
                        config={"recursion_limit": recursion_limit},
                    )
                    messages = result.get("messages") or []
                    invoke_error = ""
                except Exception as error2:
                    invoke_error = str(error2)
                    messages = []
            if invoke_error:
                _emit(
                    event_handler,
                    {
                        "type": "status",
                        "title": title,
                        "message": f"{title} 提前结束：{invoke_error[:160]}",
                        "role": role,
                        "round": event_round,
                    },
                )

        tool_chain: list[str] = []
        tool_records: list[dict[str, Any]] = []
        handoff: dict[str, Any] | None = None
        scope_updates: dict[str, Any] = {}
        pending_calls: dict[str, dict[str, Any]] = {}
        plan_calls: list[dict[str, Any]] = []

        for message in messages:
            if isinstance(message, AIMessage) and getattr(message, "tool_calls", None):
                for call in message.tool_calls or []:
                    tname = str(call.get("name") or "")
                    args = call.get("args") if isinstance(call.get("args"), dict) else {}
                    call_id = str(call.get("id") or f"{tname}-{len(pending_calls)+1}")
                    pending_calls[call_id] = {"name": tname, "arguments": args}
                    if tname and not tname.startswith("handoff") and tname != "loadSkill":
                        tool_chain.append(tname)
                        if role == "planner":
                            plan_calls.append({"id": call_id, "tool": tname, "arguments": args})
            if isinstance(message, ToolMessage):
                payload = _safe_json(str(message.content or ""))
                tool_call_id = str(getattr(message, "tool_call_id", "") or "")
                pending = pending_calls.get(tool_call_id) or {}
                tname = str(getattr(message, "name", "") or pending.get("name") or "")
                arguments = pending.get("arguments") if isinstance(pending.get("arguments"), dict) else {}
                if payload.get("handoff"):
                    handoff = payload
                    continue
                if not tname or tname.startswith("handoff") or tname == "loadSkill":
                    # loadSkill 结果可当作思考补充展示
                    if tname == "loadSkill" and role:
                        skill_id = arguments.get("skillId") or payload.get("skillId") or ""
                        _emit_delta(f"\n[已加载 skill: {skill_id}]\n", kind="thinking")
                    continue
                call_id = tool_call_id or f"{tname}-{len(tool_records)+1}"
                scope_updates[call_id] = payload
                ok = payload.get("ok") is not False
                summary = _tool_summary(tname, payload)
                tool_records.append({
                    "id": call_id,
                    "tool": tname,
                    "arguments": arguments,
                    "result": payload,
                    "summary": summary,
                    "ok": ok,
                    "round": max(1, round_number),
                    "phase": "completed" if ok else "failed",
                    "error": None if ok else payload.get("error"),
                    **summary,
                })

        text = _last_ai_text(messages) or streamed_text.strip()
        thinking = _last_ai_thinking(messages) or streamed_thinking.strip()
        if invoke_error and not text:
            text = f"{title} 未正常结束：{invoke_error[:200]}"
        if role:
            end_event: dict[str, Any] = {
                "type": "agent_end",
                "title": title,
                "message": text[:300] if text else f"{title} 完成",
                "role": role,
                "round": event_round,
                "thinking": thinking[:2000] if thinking else "",
                "modelSummary": {
                    "summary": text[:500] if text else "",
                    "thinking": thinking[:800] if thinking else "",
                    "goal": (handoff or {}).get("goal") or (handoff or {}).get("planHint") or "",
                    "reason": (handoff or {}).get("reason") or invoke_error or "",
                    "answerHint": (handoff or {}).get("answerHint") or "",
                },
                "calls": plan_calls if role == "planner" else [
                    {
                        "id": item["id"],
                        "tool": item["tool"],
                        "arguments": item.get("arguments") or {},
                        "ok": item.get("ok") is not False,
                        "skipped": False,
                        "error": item.get("error"),
                        "summary": item.get("summary") or {},
                        **(item.get("summary") or {}),
                    }
                    for item in tool_records
                ],
            }
            if role == "planner":
                handoff_calls = PlanExecutor.sanitize_calls((handoff or {}).get("calls"))
                if handoff_calls:
                    end_event["calls"] = [
                        {"id": c["id"], "tool": c["tool"], "arguments": c.get("arguments") or {}}
                        for c in handoff_calls
                    ]
                end_event["planBlueprint"] = [
                    {
                        "stepId": str(call.get("id") or f"step-{i+1}"),
                        "title": f"执行 {call.get('tool')}",
                        "tools": [call.get("tool")],
                        "optional": False,
                    }
                    for i, call in enumerate(end_event.get("calls") or [])
                ]
                if (invoke_error or not handoff_calls) and not handoff:
                    end_event["fallback"] = "规划未完成 handoff，将使用默认检索计划"
                elif invoke_error and handoff:
                    end_event["fallback"] = ""
                # Plan 的 agent_end 一律延后到 plan_node，确保含最终 calls
                end_event["_defer_emit"] = True
            if role == "reflector":
                end_event["state"] = (handoff or {}).get("state") or (
                    "replan" if (handoff or {}).get("handoff") == "plan" or (handoff or {}).get("replan") else "uncertain"
                )
                end_event["evidenceGap"] = (handoff or {}).get("evidenceGap")
                end_event["nextAction"] = (handoff or {}).get("nextAction")
                end_event["acceptanceGoal"] = (
                    (state.get("intent") or {}).get("successCriteria")
                    or (state.get("intent") or {}).get("expectedOutcome")
                )
            if end_event.get("_defer_emit"):
                end_event.pop("_defer_emit", None)
            else:
                _emit(event_handler, end_event)

        return {
            "handoff": handoff,
            "text": text,
            "tool_chain": tool_chain,
            "tool_records": tool_records,
            "scope_updates": scope_updates,
            "skill_ids": skill_ids,
            "plan_calls": plan_calls,
            "invoke_error": invoke_error,
            "thinking": thinking,
            "deferred_end_event": end_event if role == "planner" else None,
        }

    def intent_node(state: AgentState) -> Command:
        question = state.get("question") or ""
        user = json.dumps(
            {
                "task": "识别意图并必须调用 handoff_to_plan",
                "question": question,
                "referenceTime": reference_time.isoformat(timespec="seconds"),
                "queryTopK": state.get("query_top_k") or default_top_k,
                "broadMatchTopK": _normalize_broad_match_top_k(state.get("broad_match_top_k", default_broad_match_top_k)),
                "intentSchema": {
                    "targetScope": "track_memory|registry|both",
                    "targetKind": "hull|description|all",
                    "operation": "existence|list|time|count|explain",
                    "registryRelation": "any|in|out",
                    "hullNumber": "str|null",
                    "description": "str|null",
                    "timeRange": "[start,end]|null",
                    "timeExpression": "str|null",
                    "targetItems": [],
                    "expectedOutcome": "str",
                    "successCriteria": "str",
                    "nextAgentFocus": "str",
                    "questionType": "str",
                },
            },
            ensure_ascii=False,
        )
        out = _run_agent(
            "intent",
            "intent_agent",
            "意图识别智能体（IntentAgent）",
            INTENT_RESPONSIBILITY,
            intent_tools,
            state,
            user,
            role="intent",
            round_number=0,
        )
        handoff = out.get("handoff") or {}
        intent = handoff.get("intent") if isinstance(handoff.get("intent"), dict) else {}
        inferred = infer_intent_fields(question)
        used_fallback = not intent

        if not intent:
            intent = {
                **inferred,
                "question": question,
                "intentSource": "langgraph_fallback",
            }
        else:
            intent = dict(intent)
            intent["intentSource"] = intent.get("intentSource") or "langgraph_react"

        # 工具结果优先写入时间/多目标/舷号
        for record in out.get("tool_records") or []:
            result = record.get("result") or {}
            if record.get("tool") == "parseTime" and result.get("timeRange"):
                intent["timeRange"] = result.get("timeRange")
                intent["timeExpression"] = result.get("expression")
                intent["timeSource"] = "tool"
            if record.get("tool") == "parseTargets" and result.get("targetItems"):
                intent["targetItems"] = result.get("targetItems")
            if record.get("tool") == "extractHull" and result.get("hullNumber"):
                intent["hullNumber"] = result.get("hullNumber")
                intent["targetKind"] = "hull"

        # 模型 handoff 残缺时，用规则补全 description / operation 等
        # 规则舷号更完整时覆盖（避免 extractHull 旧结果只剩数字）
        inferred_hull = str(inferred.get("hullNumber") or "").strip()
        current_hull = str(intent.get("hullNumber") or "").strip()
        if inferred_hull and (
            not current_hull
            or (current_hull.isdigit() and inferred_hull != current_hull and current_hull in inferred_hull)
            or (len(inferred_hull) > len(current_hull) and current_hull and current_hull in inferred_hull)
        ):
            intent["hullNumber"] = inferred_hull
        if not intent.get("description") and inferred.get("description"):
            intent["description"] = inferred["description"]
        # 伪描述：把「哪些在库船」当 description → 清空，避免 matchText(问句)
        desc_now = str(intent.get("description") or "").strip()
        if desc_now and (
            any(token in desc_now for token in ("哪些", "有哪些", "在库", "未在库", "先验库", "库船"))
            or desc_now in {"船", "船舶", "船只", "目标", "对象"}
            or str(inferred.get("questionType") or "") == "registry_in_list"
        ):
            if not inferred.get("description") or str(inferred.get("questionType") or "") == "registry_in_list":
                intent["description"] = None
        if not intent.get("targetItems") and inferred.get("targetItems"):
            intent["targetItems"] = inferred["targetItems"]
        if intent.get("targetItems"):
            intent["targetItems"] = normalize_target_items(intent.get("targetItems"))
        if not intent.get("operation") or intent.get("operation") not in {
            "existence", "list", "time", "count", "explain",
        }:
            intent["operation"] = inferred.get("operation") or "list"
        # 在库列表问法：强制 list + in + both（覆盖模型误判 existence/OCR）
        if str(inferred.get("questionType") or "") == "registry_in_list":
            intent["operation"] = "list"
            intent["registryRelation"] = "in"
            intent["targetScope"] = inferred.get("targetScope") or "both"
            intent["targetKind"] = "all"
            intent["description"] = None
            intent["questionType"] = "registry_in_list"
        if not intent.get("targetKind") or intent.get("targetKind") == "all":
            if intent.get("hullNumber"):
                intent["targetKind"] = "hull"
            elif intent.get("description"):
                intent["targetKind"] = "description"
            else:
                intent["targetKind"] = inferred.get("targetKind") or "all"
        if not intent.get("targetScope"):
            intent["targetScope"] = inferred.get("targetScope") or "track_memory"
        if not intent.get("registryRelation"):
            intent["registryRelation"] = inferred.get("registryRelation") or "any"
        # 验收/焦点：残缺或过窄（仅 getTrack、把 0 轨迹当否定、OCR-only 在库列表）时用规则覆盖
        def _acceptance_too_narrow(value: Any) -> bool:
            text = str(value or "").strip()
            if not text:
                return True
            low = text.lower()
            bad_tokens = (
                "0 轨迹即可否定", "0轨迹即可否定", "未检测到=未出现", "未检测到即未出现",
                "仅 gettrack", "仅gettrack", "只 gettrack", "只gettrack",
            )
            if any(t in low for t in bad_tokens):
                return True
            # 舷号存在判断却只提 getTrack、不提库/视觉
            if intent.get("hullNumber") and str(intent.get("operation") or "") == "existence":
                mentions_track = "gettrack" in low or "轨迹" in text
                mentions_followup = any(
                    t in low for t in ("getregistry", "matchimage", "先验库", "视觉", "库图", "关键帧匹配")
                )
                if mentions_track and not mentions_followup and len(text) < 80:
                    return True
            # 在库列表却写成 OCR 舷号比对 / matchText 问句
            if str(intent.get("registryRelation") or "") == "in" and str(intent.get("operation") or "") == "list":
                ocr_only = ("ocr" in low or "识别舷号" in text) and "matchimage" not in low and "库图" not in text
                matchtext_query = "matchtext" in low and any(t in text for t in ("哪些", "在库", "用户问题"))
                no_visual = "listregistry" not in low and "matchimage" not in low and "库图" not in text
                if ocr_only or matchtext_query or (no_visual and ("ocr" in low or "舷号" in text)):
                    return True
            return False

        if _acceptance_too_narrow(intent.get("expectedOutcome")) and inferred.get("expectedOutcome"):
            intent["expectedOutcome"] = inferred["expectedOutcome"]
        if _acceptance_too_narrow(intent.get("successCriteria")) and inferred.get("successCriteria"):
            intent["successCriteria"] = inferred["successCriteria"]
        if _acceptance_too_narrow(intent.get("nextAgentFocus")) and inferred.get("nextAgentFocus"):
            intent["nextAgentFocus"] = inferred["nextAgentFocus"]
        # registry_in_list 始终用规则验收覆盖模型 OCR 写法
        if str(inferred.get("questionType") or "") == "registry_in_list":
            for key in ("expectedOutcome", "successCriteria", "nextAgentFocus"):
                if inferred.get(key):
                    intent[key] = inferred[key]
        if not intent.get("questionType"):
            intent["questionType"] = inferred.get("questionType")
        if intent.get("intentConfidence") is None:
            intent["intentConfidence"] = inferred.get("intentConfidence")
        if used_fallback and intent.get("description"):
            # 描述类兜底比纯 all 更可信
            intent["intentConfidence"] = max(float(intent.get("intentConfidence") or 0), 0.72)

        intent.setdefault("question", question)
        intent["selectedSkills"] = out.get("skill_ids") or []
        if intent.get("timeRange") and not intent.get("queryScope"):
            intent["queryScope"] = intent.get("timeRange")
        _emit(
            event_handler,
            {
                "type": "classification",
                "title": "IntentAgent 意图识别完成",
                "message": "LangGraph IntentAgent 完成",
                "queryScope": intent.get("timeRange") or intent.get("queryScope"),
                "intentSource": intent.get("intentSource"),
                **{k: intent.get(k) for k in (
                    "questionType", "strategy", "operation", "targetScope", "targetKind",
                    "registryRelation", "description", "hullNumber", "targetItems",
                    "timeExpression", "timeRange", "expectedOutcome", "successCriteria", "nextAgentFocus",
                    "timeParseError", "timeSource", "intentConfidence", "selectedRules",
                )},
            },
        )
        return Command(
            goto="plan",
            update={
                "intent": intent,
                "active_agent": "plan",
                "tool_chain": out.get("tool_chain") or [],
                "tool_records": out.get("tool_records") or [],
            },
        )

    def plan_node(state: AgentState) -> Command:
        loop_count = int(state.get("loop_count") or 0)
        round_number = loop_count + 1
        intent = state.get("intent") or {}
        reflection = state.get("reflection") or {}
        replan_hint = str(
            reflection.get("nextAction")
            or reflection.get("evidenceGap")
            or reflection.get("reason")
            or ""
        )
        # 硬兜底 replan：Reflect 已写清补洞链时走确定性 calls（安全网）
        # 其它轮次尽量让 Plan 模型自主规划；模型失败/空 calls 再兜底
        hard_replan = bool(reflection.get("hardReplan")) or any(
            token in replan_hint.lower()
            for token in ("matchimage", "视觉匹配", "不带hull", "registryreferences")
        )
        use_deterministic_replan = bool(loop_count > 0 and replan_hint and hard_replan)
        used_default_plan = False
        out: dict[str, Any] = {
            "handoff": None,
            "text": "",
            "tool_chain": [],
            "tool_records": [],
            "invoke_error": "",
            "thinking": "",
            "deferred_end_event": None,
        }
        if use_deterministic_replan:
            plan_calls = _default_plan_calls(
                intent,
                state.get("query_top_k") or default_top_k,
                broad_match_top_k=_normalize_broad_match_top_k(
                    state.get("broad_match_top_k", default_broad_match_top_k)
                ),
                replan_hint=replan_hint,
            )
            used_default_plan = True
            plan_hint = f"[补洞计划] {' → '.join(c['tool'] for c in plan_calls)}"
            handoff = {
                "handoff": "observe",
                "goal": replan_hint,
                "calls": plan_calls,
                "planHint": plan_hint,
                "reason": "按 Reflect 硬兜底 nextAction 确定性规划",
            }
            target = "observe"
            _emit(
                event_handler,
                {
                    "type": "agent_start",
                    "title": "规划智能体（PlanAgent）",
                    "message": "按验收缺口生成补洞计划",
                    "role": "planner",
                    "round": round_number,
                },
            )
            out["handoff"] = handoff
            out["text"] = plan_hint
            out["deferred_end_event"] = {
                "type": "agent_end",
                "title": "规划智能体（PlanAgent）",
                "role": "planner",
                "round": round_number,
                "message": plan_hint,
                "modelSummary": {
                    "summary": plan_hint,
                    "goal": replan_hint,
                    "reason": "硬兜底补洞链，确定性规划",
                },
                "thinking": "",
            }
        else:
            compact_intent = {
                k: intent.get(k)
                for k in (
                    "operation", "targetScope", "targetKind", "registryRelation",
                    "hullNumber", "description", "timeRange", "timeExpression",
                    "targetItems", "expectedOutcome", "successCriteria", "nextAgentFocus",
                    "questionType",
                )
                if intent.get(k) not in (None, "", [], {})
            }
            user = json.dumps(
                {
                    "task": "立刻调用 handoff_to_observe(goal, calls, planHint)。禁止只输出正文，禁止执行业务工具。",
                    "question": state.get("question"),
                    "intent": compact_intent,
                    "loop": loop_count,
                    "round": round_number,
                    "maxRounds": state.get("max_rounds") or max_rounds,
                    "queryTopK": state.get("query_top_k") or default_top_k,
                    "broadMatchTopK": _normalize_broad_match_top_k(state.get("broad_match_top_k", default_broad_match_top_k)),
                    "replanHint": replan_hint or None,
                    "workingScopeKeys": list((state.get("working_scope") or {}).keys())[:24],
                    "availableTools": [
                        "getTrack", "getFrames", "getClip", "getRegistry", "listRegistry",
                        "matchHull", "matchText", "matchImage", "verifyTarget", "showEvidence", "dedupTracks",
                    ],
                    "rules": [
                        "calls 至少 1 步；arguments 跨步骤用 {\"$ref\":\"{callId}.{field}\"}",
                        "视频舷号：getTrack(hullNumber)；0 轨迹后由 Reflect 引导查库/视觉，勿一次塞满",
                        "描述：getTrack → getFrames → matchText(galleryImages=$ref frames.keyframes)",
                        "先验库舷号：getRegistry(hullNumber)",
                        "视觉补洞：getRegistry → getTrack(不带hull) → getFrames → matchImage(query=registryReferences, gallery=keyframes)",
                        "广泛多库多轨迹：listRegistry → getTrack(limit=0) → getFrames → matchImage，使用 broadMatchTopK；0 表示不截断，不要复用 queryTopK",
                        "有 replanHint 时优先落实其中点名的工具链",
                        "无法规划时才 handoff_to_reflect",
                    ],
                },
                ensure_ascii=False,
            )
            out = _run_agent(
                "plan",
                "plan_agent",
                "规划智能体（PlanAgent）",
                PLAN_RESPONSIBILITY,
                plan_tools,
                state,
                user,
                role="planner",
                round_number=round_number,
                recursion_limit=8,
            )
            handoff = out.get("handoff") or {}
            target = str(handoff.get("handoff") or "observe")
            plan_hint = str(handoff.get("planHint") or handoff.get("goal") or "")
            plan_calls = _apply_retrieval_limits(
                PlanExecutor.sanitize_calls(handoff.get("calls")),
                broad_match_top_k=_normalize_broad_match_top_k(
                    state.get("broad_match_top_k", default_broad_match_top_k)
                ),
            )

            # 模型未给出 calls 时，按意图 + replanHint 生成最小可执行链
            if target != "reflect" and not plan_calls:
                plan_calls = _default_plan_calls(
                    intent,
                    state.get("query_top_k") or default_top_k,
                    broad_match_top_k=_normalize_broad_match_top_k(
                        state.get("broad_match_top_k", default_broad_match_top_k)
                    ),
                    replan_hint=replan_hint,
                )
                used_default_plan = True
                if not plan_hint:
                    plan_hint = " → ".join(c["tool"] for c in plan_calls) or "无步骤"
                if out.get("invoke_error"):
                    plan_hint = f"[规划兜底] {plan_hint}"

        # Plan agent_end 统一在此发出（含最终 calls）
        deferred = out.get("deferred_end_event") or {
            "type": "agent_end",
            "title": "规划智能体（PlanAgent）",
            "role": "planner",
            "round": round_number,
        }
        end_event = dict(deferred)
        end_event.update(
            {
                "type": "agent_end",
                "title": "规划智能体（PlanAgent）",
                "role": "planner",
                "round": round_number,
                "message": (plan_hint or end_event.get("message") or "规划完成")[:300],
                "fallback": (
                    (
                        "按 Reflect 补洞指令生成计划"
                        if use_deterministic_replan
                        else "规划未完成 handoff，已使用默认检索计划"
                    )
                    if used_default_plan
                    else (end_event.get("fallback") or "")
                ),
                "calls": [
                    {"id": c["id"], "tool": c["tool"], "arguments": c.get("arguments") or {}}
                    for c in plan_calls
                ],
                "planBlueprint": [
                    {
                        "stepId": str(c.get("id") or f"step-{i+1}"),
                        "title": f"执行 {c.get('tool')}",
                        "tools": [c.get("tool")],
                        "optional": False,
                    }
                    for i, c in enumerate(plan_calls)
                ],
                "modelSummary": {
                    **(end_event.get("modelSummary") or {}),
                    "summary": (plan_hint or "")[:500],
                    "goal": plan_hint or (end_event.get("modelSummary") or {}).get("goal") or "",
                    "reason": (
                        out.get("invoke_error")
                        or (end_event.get("modelSummary") or {}).get("reason")
                        or ""
                    )[:200],
                },
            }
        )
        if out.get("thinking") and not end_event.get("thinking"):
            end_event["thinking"] = str(out.get("thinking") or "")[:2000]
        _emit(event_handler, end_event)

        update: dict[str, Any] = {
            "plan_hint": plan_hint,
            "plan_calls": plan_calls,
            "active_agent": "observe" if target != "reflect" else "reflect",
            "tool_chain": out.get("tool_chain") or [],
            "tool_records": out.get("tool_records") or [],
        }
        if target == "reflect":
            update["observation_summary"] = str(handoff.get("summary") or plan_hint or "规划未给出可执行步骤")
            return Command(goto="reflect", update=update)
        return Command(goto="observe", update=update)

    def observe_node(state: AgentState) -> Command:
        """确定性执行 Plan 的 calls（对齐 old Observer），不把完整工具结果塞进 ReAct 对话。"""
        loop_count = int(state.get("loop_count") or 0)
        round_number = loop_count + 1
        plan_calls = _apply_retrieval_limits(
            PlanExecutor.sanitize_calls(state.get("plan_calls") or []),
            broad_match_top_k=_normalize_broad_match_top_k(
                state.get("broad_match_top_k", default_broad_match_top_k)
            ),
        )
        if not plan_calls:
            # 无 calls 时再兜底一次
            plan_calls = _default_plan_calls(
                state.get("intent") or {},
                state.get("query_top_k") or default_top_k,
                broad_match_top_k=_normalize_broad_match_top_k(
                    state.get("broad_match_top_k", default_broad_match_top_k)
                ),
                replan_hint=str((state.get("reflection") or {}).get("nextAction") or state.get("plan_hint") or ""),
            )

        _emit(
            event_handler,
            {
                "type": "agent_start",
                "title": "观察执行智能体（ObserveAgent）",
                "message": f"按计划确定性执行 {len(plan_calls)} 个工具步骤",
                "role": "observer",
                "round": round_number,
            },
        )

        def on_tool_event(event: dict[str, Any]) -> None:
            tool_name = str(event.get("tool") or "")
            phase = str(event.get("phase") or "running")
            summary = event.get("summary") if isinstance(event.get("summary"), dict) else {}
            _emit(
                event_handler,
                {
                    "type": "agent_tool",
                    "title": "ObserveAgent",
                    "message": tool_name,
                    "role": "observer",
                    "round": round_number,
                    "id": event.get("id") or tool_name,
                    "tool": tool_name,
                    "arguments": event.get("arguments") or {},
                    "ok": event.get("ok", True) is not False,
                    "status": phase,
                    "phase": phase,
                    "summary": summary,
                    "error": event.get("error"),
                    "skipped": bool(event.get("skipped")),
                    **{k: v for k, v in summary.items() if k not in {"id", "tool"}},
                },
            )

        executor = PlanExecutor(tools)
        executed = executor.execute(
            plan_calls,
            scope=state.get("working_scope") or {},
            on_tool_event=on_tool_event,
        )
        summary_obj = executed.get("summary") or {}
        call_summaries = summary_obj.get("calls") or []
        lines = []
        for item in call_summaries:
            if item.get("skipped"):
                lines.append(f"{item.get('tool')}: 跳过 ({item.get('skipReason') or item.get('error') or '条件不满足'})")
            elif item.get("ok") is False:
                lines.append(f"{item.get('tool')}: 失败 ({item.get('error') or 'unknown'})")
            else:
                bits = [str(item.get("tool") or "tool")]
                if item.get("trackCount") is not None:
                    bits.append(f"{item['trackCount']} 条轨迹")
                if item.get("keyframeCount") is not None:
                    bits.append(f"{item['keyframeCount']} 张关键帧")
                if item.get("matchCount") is not None:
                    bits.append(f"{item['matchCount']} 条匹配")
                if item.get("registryCount") is not None:
                    bits.append(f"{item['registryCount']} 个库项")
                if item.get("exactMatchHullCount") is not None:
                    bits.append(f"{item['exactMatchHullCount']} 组舷号命中")
                lines.append(" · ".join(bits) if len(bits) > 1 else f"{bits[0]}: 完成")
        observation_summary = "；".join(lines) if lines else "本轮无工具执行"
        if state.get("plan_hint"):
            observation_summary = f"计划：{state.get('plan_hint')}\n结果：{observation_summary}"

        _emit(
            event_handler,
            {
                "type": "agent_end",
                "title": "观察执行智能体（ObserveAgent）",
                "message": observation_summary[:300],
                "role": "observer",
                "round": round_number,
                "modelSummary": {"summary": observation_summary[:500]},
                "calls": [
                    {
                        "id": item.get("id"),
                        "tool": item.get("tool"),
                        "arguments": {},
                        # skip 不算失败；仅真实执行失败才 ok=false
                        "ok": False if (not item.get("skipped") and item.get("ok") is False) else True,
                        "skipped": bool(item.get("skipped")),
                        "error": item.get("error") or item.get("skipReason"),
                        "summary": item,
                        **{k: v for k, v in item.items() if k not in {"id", "tool"}},
                    }
                    for item in call_summaries
                ],
            },
        )

        # working_scope：保留本轮完整工具结果（按 call id 索引），供合成与后续 $ref
        scope_updates = {
            key: value
            for key, value in (executed.get("scope") or {}).items()
            if key not in (state.get("working_scope") or {}) or key in {c["id"] for c in plan_calls}
        }
        # 若 merge 需要整表，直接传 executed.scope 中本轮 id
        for call in plan_calls:
            cid = call["id"]
            if cid in (executed.get("scope") or {}):
                scope_updates[cid] = executed["scope"][cid]

        return Command(
            goto="reflect",
            update={
                "working_scope": scope_updates,
                "observation_summary": observation_summary,
                "active_agent": "reflect",
                "tool_chain": executed.get("tool_chain") or [],
                "tool_records": executed.get("tool_records") or [],
            },
        )

    def reflect_node(state: AgentState) -> Command:
        loop_count = int(state.get("loop_count") or 0) + 1
        limit = int(state.get("max_rounds") or max_rounds)
        intent = state.get("intent") or {}
        scope = state.get("working_scope") or {}
        question = str(state.get("question") or "")
        # 成功证据：working_scope 中存在 ok 且含 tracks/matches/keyframes/registry 等
        has_tool_evidence = False
        evidence_bits: list[str] = []
        track_counts: list[int] = []
        registry_checked = False
        for key, value in scope.items():
            if not isinstance(value, dict) or value.get("ok") is False:
                continue
            # 空 tracks=[] 也算已执行 getTrack（否定证据）
            if isinstance(value.get("tracks"), list):
                n = len(value.get("tracks") or [])
                track_counts.append(n)
                evidence_bits.append(f"{key}:tracks={n}")
                has_tool_evidence = True
            if isinstance(value.get("trackIds"), list):
                has_tool_evidence = True
                evidence_bits.append(f"{key}:trackIds={len(value.get('trackIds') or [])}")
            if any(value.get(field) not in (None, [], {}) for field in (
                "keyframes", "keyframeIds", "matches",
                "registryItems", "registryReferences", "exactMatches",
                "highThresholdShipCount", "uniqueCount", "count",
            )):
                has_tool_evidence = True
                if value.get("keyframes") is not None:
                    evidence_bits.append(f"{key}:keyframes={len(value.get('keyframes') or [])}")
                if value.get("matches") is not None:
                    evidence_bits.append(f"{key}:matches={len(value.get('matches') or [])}")
                if value.get("registryItems") is not None or value.get("registryReferences") is not None:
                    registry_checked = True
                    evidence_bits.append(f"{key}:registry")
                if value.get("exactMatches") is not None:
                    registry_checked = True
                    evidence_bits.append(f"{key}:exactHull")
            # getRegistry 命中单项 / 明确 found
            if value.get("registryItem") is not None or value.get("found") in (True, False):
                registry_checked = True
                has_tool_evidence = True
                evidence_bits.append(f"{key}:registryLookup")
        # 工具链非空也算有执行痕迹
        if not has_tool_evidence and (state.get("tool_chain") or state.get("tool_records")):
            has_tool_evidence = any(
                isinstance(r, dict) and r.get("ok") is not False and not r.get("skipped")
                for r in (state.get("tool_records") or [])
            )
        tool_names = {
            str(r.get("tool") or "")
            for r in (state.get("tool_records") or [])
            if isinstance(r, dict)
        }
        tool_names.update(str(t) for t in (state.get("tool_chain") or []))
        if tool_names & {"getRegistry", "listRegistry", "matchHull"}:
            registry_checked = True
        visual_matched = bool(tool_names & {"matchImage", "matchText"})
        match_count_total = 0
        confirmed_match_count = 0
        uncertain_match_count = 0
        registry_searchable = False
        registry_found = False
        registry_has_items = False
        for key, value in scope.items():
            if not isinstance(value, dict) or value.get("ok") is False:
                continue
            if isinstance(value.get("matches"), list):
                visual_matched = True
                for m in value.get("matches") or []:
                    if not isinstance(m, dict):
                        continue
                    match_count_total += 1
                    band = str(m.get("scoreBand") or "")
                    if band == "match":
                        confirmed_match_count += 1
                    elif band == "uncertain":
                        uncertain_match_count += 1
            # 优先使用工具返回的 confirmedMatches
            if isinstance(value.get("confirmedMatches"), list):
                confirmed_match_count = max(confirmed_match_count, len(value.get("confirmedMatches") or []))
            if isinstance(value.get("uncertainMatches"), list):
                uncertain_match_count = max(uncertain_match_count, len(value.get("uncertainMatches") or []))
            refs = value.get("registryReferences")
            items = value.get("registryItems")
            item_one = value.get("registryItem")
            if value.get("searchable") is True or (isinstance(refs, list) and refs):
                registry_searchable = True
                registry_checked = True
            if value.get("found") is True or item_one is not None or (isinstance(items, list) and items):
                registry_found = True
                registry_has_items = True
                registry_checked = True
            # 有库项但 searchable 未标/参考图嵌在 items.references 时仍视为可尝试视觉
            if isinstance(items, list):
                for it in items:
                    if not isinstance(it, dict):
                        continue
                    nested = it.get("references") or it.get("registryReferences") or []
                    if nested:
                        registry_searchable = True
            if isinstance(item_one, dict):
                nested = item_one.get("references") or item_one.get("registryReferences") or []
                if nested:
                    registry_searchable = True
        # 历史 tool_records 摘要：found/searchable（working_scope 字段名不一致时的兜底）
        for r in state.get("tool_records") or []:
            if not isinstance(r, dict) or r.get("tool") not in {"getRegistry", "listRegistry"}:
                continue
            res = r.get("result") if isinstance(r.get("result"), dict) else {}
            summary = r.get("summary") if isinstance(r.get("summary"), dict) else {}
            if res.get("found") is True or summary.get("found") is True:
                registry_found = True
                registry_checked = True
            if res.get("searchable") is True or summary.get("searchable") is True:
                registry_searchable = True
            if (res.get("registryReferenceCount") or summary.get("registryReferenceCount") or 0) > 0:
                registry_searchable = True
            if (res.get("registryItemCount") or summary.get("registryItemCount") or 0) > 0:
                registry_has_items = True
                registry_found = True
        zero_tracks = bool(track_counts) and max(track_counts) == 0
        # 带舷号过滤轨迹为 0，但可能仍有未标舷号的视频目标 → 需要放开 hull 再扫 + matchImage
        hull_filtered_zero = zero_tracks and any(
            isinstance(r, dict)
            and r.get("tool") == "getTrack"
            and not r.get("skipped")
            and (r.get("arguments") or {}).get("hullNumber")
            for r in (state.get("tool_records") or [])
        )
        hull = str(intent.get("hullNumber") or "").strip()
        target_scope = str(intent.get("targetScope") or "track_memory")
        registry_relation = str(intent.get("registryRelation") or "any")
        focus_blob = " ".join(
            str(intent.get(k) or "")
            for k in ("nextAgentFocus", "expectedOutcome", "successCriteria")
        )
        op = str(intent.get("operation") or "")
        # 1) 未查库 → 先查库（舷号存在/列表类，视频 0 轨迹）
        should_replan_registry = (
            loop_count < limit
            and bool(hull)
            and zero_tracks
            and not registry_checked
            and op in {"existence", "list", "explain", "time", ""}
        )
        # 2) 已查库且有库参考图/可搜向量，但还没 matchImage → 视觉补洞
        #    found 库项 + references 也算 can_try（不单靠 searchable 标志）
        visual_attempted = visual_matched or any(
            isinstance(r, dict)
            and r.get("tool") in {"matchImage", "matchText"}
            for r in (state.get("tool_records") or [])
        )
        for r in state.get("tool_records") or []:
            if not isinstance(r, dict):
                continue
            if r.get("tool") in {"matchImage", "matchText"}:
                visual_attempted = True
                res = r.get("result") if isinstance(r.get("result"), dict) else {}
                if isinstance(res.get("matches"), list) or res.get("visualAttempted"):
                    visual_matched = True
        can_try_visual = bool(registry_searchable or registry_found or registry_has_items)
        should_replan_visual = (
            loop_count < limit
            and bool(hull)
            and registry_checked
            and can_try_visual
            and not visual_attempted
            and op in {"existence", "list", "explain", "time", ""}
            and (zero_tracks or hull_filtered_zero or target_scope in {"track_memory", "both", ""})
        )
        # 3) 「有哪些在库船出现」：已 list 库但未做 matchImage，应 replan 视觉对照
        is_registry_in_list = (
            registry_relation == "in"
            and op == "list"
            and not hull
            and (
                target_scope in {"both", "registry"}
                or str(intent.get("questionType") or "") == "registry_in_list"
            )
        )
        should_replan_registry_list_visual = (
            loop_count < limit
            and is_registry_in_list
            and registry_checked
            and can_try_visual
            and not visual_attempted
        )
        # 4) 在库列表完全没查库
        should_replan_registry_list = (
            loop_count < limit
            and is_registry_in_list
            and not registry_checked
        )
        pure_video_sufficient = (
            has_tool_evidence
            and not should_replan_registry
            and not should_replan_visual
            and not should_replan_registry_list
            and not should_replan_registry_list_visual
            and op == "existence"
            and registry_checked
            and (visual_attempted or not can_try_visual)
        )
        user = json.dumps(
            {
                "task": "判定是否退出。replan→handoff_to_plan_replan；否则必须 handoff_finish",
                "question": question,
                "expectedOutcome": intent.get("expectedOutcome"),
                "successCriteria": intent.get("successCriteria"),
                "nextAgentFocus": intent.get("nextAgentFocus"),
                "targetScope": target_scope,
                "registryRelation": registry_relation,
                "hullNumber": hull or None,
                "observation": state.get("observation_summary"),
                "workingScopeKeys": list(scope.keys()),
                "evidenceHints": evidence_bits[:20],
                "hasToolEvidence": has_tool_evidence,
                "zeroTracks": zero_tracks,
                "registryChecked": registry_checked,
                "registrySearchable": registry_searchable,
                "registryFound": registry_found,
                "registryHasItems": registry_has_items,
                "canTryVisual": can_try_visual,
                "visualMatched": visual_matched,
                "visualAttempted": visual_attempted,
                "matchCount": match_count_total,
                "confirmedMatchCount": confirmed_match_count,
                "uncertainMatchCount": uncertain_match_count,
                "shouldReplanRegistry": should_replan_registry,
                "shouldReplanVisual": should_replan_visual or should_replan_registry_list_visual,
                "shouldReplanRegistryList": should_replan_registry_list,
                "isRegistryInList": is_registry_in_list,
                "zeroTracks": zero_tracks,
                "hullFilteredZero": hull_filtered_zero,
                "toolChain": list(tool_names)[:20],
                "loop": loop_count,
                "maxRounds": limit,
                "notes": [
                    "hasToolEvidence=true 表示已有工具成功结果，勿说「没有任何成功工具结果」",
                    "shouldReplanRegistry=true → 必须 handoff_to_plan_replan，nextAction 写 getRegistry",
                    "shouldReplanVisual=true → 必须 handoff_to_plan_replan，nextAction 写完整视觉链（含 matchImage）",
                    "isRegistryInList=true：问题是「哪些在库船出现」，禁止用 matchText(用户问句) 当证据",
                    "shouldReplanRegistryList=true → replan：listRegistry→getTrack→getFrames→matchImage",
                    "禁止在 shouldReplan*=true 时 sufficient",
                    "visualAttempted=true 或 matchCount 已给出 → 勿再要求 matchImage",
                    "canTryVisual=false（无可搜库图）且已查库 → 可 sufficient「库有记录但无法视觉匹配/视频未发现」",
                    "全量 getTrack 有轨迹但 matchCount=0 且 hull 过滤为 0 → 结论仍是视频未确认该舷号，不是「确认出现」",
                    "matchText 的 description 若是用户整句/含「哪些在库」→ 无效，须 replan 走 matchImage",
                    "仅 confirmedMatchCount>0 才可说「确认出现」；uncertainMatchCount 只能说疑似/灰区",
                    "展示候选必须按 embeddingScore 排序，禁止固定轨迹 1/2/3",
                    "无任何工具执行痕迹时禁止 sufficient",
                    "接近 maxRounds 仍不足 → uncertain",
                ],
            },
            ensure_ascii=False,
        )
        out = _run_agent(
            "reflect",
            "reflect_agent",
            "反思判定智能体（ReflectAgent）",
            REFLECT_RESPONSIBILITY,
            [
                build_load_skill_tool("reflect_agent", _skill_loader("reflect_agent")),
                handoff_to_plan_replan,
                handoff_finish,
            ],
            state,
            user,
            role="reflector",
            round_number=loop_count,
            recursion_limit=6,
        )
        handoff = out.get("handoff") or {}
        # 硬兜底：模型误判 sufficient / 漏写 nextAction 时强制 replan（始终覆盖）
        if should_replan_registry:
            handoff = {
                "handoff": "plan",
                "replan": True,
                "hardReplan": True,
                "state": "replan",
                "reason": "视频轨迹为 0，尚需对照先验库确认身份/在库情况",
                "nextAction": f"使用 getRegistry(hullNumber={hull}) 查先验库，勿重复相同 getTrack",
                "evidenceGap": "未查询先验库",
            }
        elif should_replan_visual:
            handoff = {
                "handoff": "plan",
                "replan": True,
                "hardReplan": True,
                "state": "replan",
                "reason": "先验库已命中且有可搜参考图，需 matchImage 对照视频关键帧",
                "nextAction": (
                    f"getRegistry(hullNumber={hull}) → getTrack(不带hullNumber, 全时域) → "
                    "getFrames($ref trackIds) → matchImage(queryImages=$ref registry.registryReferences, "
                    "galleryImages=$ref frames.keyframes)"
                ),
                "evidenceGap": "已有可搜库图但未做库图↔视频关键帧匹配",
            }
        elif should_replan_registry_list or should_replan_registry_list_visual:
            handoff = {
                "handoff": "plan",
                "replan": True,
                "hardReplan": True,
                "state": "replan",
                "reason": "在库船列表需 listRegistry + 库图↔视频关键帧 matchImage，禁止 matchText(用户问句)",
                "nextAction": (
                    "listRegistry → getTrack(不带hullNumber) → getFrames($ref trackIds) → "
                    "matchImage(queryImages=$ref registry.registryReferences, galleryImages=$ref frames.keyframes)"
                ),
                "evidenceGap": (
                    "未列出先验库" if should_replan_registry_list
                    else "已有库图但未做库图↔视频关键帧匹配（在库列表）"
                ),
            }
        elif not handoff:
            if loop_count < limit and not has_tool_evidence:
                handoff = {
                    "handoff": "plan",
                    "replan": True,
                    "hardReplan": False,
                    "reason": "本轮未获得可用工具证据",
                    "nextAction": "补充 getTrack/getFrames 或匹配工具",
                    "evidenceGap": "working_scope 为空",
                }
            elif pure_video_sufficient or has_tool_evidence:
                if visual_attempted and match_count_total == 0 and (zero_tracks or hull_filtered_zero):
                    reason = f"先验库有记录，但库图与视频关键帧无匹配，视频中未发现{hull or '目标'}"
                elif registry_checked and not can_try_visual and (zero_tracks or hull_filtered_zero):
                    reason = f"先验库有记录但无可搜参考图，无法视觉匹配；视频 OCR 未检出{hull or '目标'}"
                elif pure_video_sufficient and (zero_tracks or hull_filtered_zero):
                    reason = f"getTrack 返回 0 条舷号命中，视频中未发现{hull or '目标'}"
                else:
                    reason = "已有工具证据，模型未显式 handoff，按充分结束"
                handoff = {
                    "handoff": "finish",
                    "state": "sufficient",
                    "reason": reason,
                    "answerHint": state.get("observation_summary") or "",
                }
            else:
                handoff = {
                    "handoff": "finish",
                    "state": "uncertain",
                    "reason": "无工具证据且无法继续",
                    "answerHint": state.get("observation_summary") or "",
                }
        elif (
            str(handoff.get("state") or "") == "sufficient"
            and bool(hull)
            and not visual_attempted
            and can_try_visual
            and loop_count < limit
            and (zero_tracks or hull_filtered_zero)
        ):
            # 模型在未视觉匹配时宣称充分 → 纠偏
            handoff = {
                "handoff": "plan",
                "replan": True,
                "hardReplan": True,
                "state": "replan",
                "reason": "库有可搜参考图但尚未 matchImage，不能直接结束",
                "nextAction": (
                    f"getRegistry(hullNumber={hull}) → getTrack(不带hullNumber) → getFrames → matchImage"
                ),
                "evidenceGap": "未做库图↔视频关键帧匹配",
            }
        elif (
            str(handoff.get("state") or "") == "sufficient"
            and is_registry_in_list
            and not visual_attempted
            and loop_count < limit
        ):
            handoff = {
                "handoff": "plan",
                "replan": True,
                "hardReplan": True,
                "state": "replan",
                "reason": "在库船列表尚未完成库图↔视频匹配，matchText(问句) 不算充分",
                "nextAction": (
                    "listRegistry → getTrack(不带hullNumber) → getFrames → "
                    "matchImage(queryImages=$ref registry.registryReferences, galleryImages=$ref frames.keyframes)"
                ),
                "evidenceGap": "在库列表缺少 matchImage",
            }
        elif (
            str(handoff.get("state") or "") == "sufficient"
            and zero_tracks
            and bool(hull)
            and not registry_checked
            and loop_count < limit
        ):
            handoff = {
                "handoff": "plan",
                "replan": True,
                "hardReplan": True,
                "state": "replan",
                "reason": "视频 0 轨迹且未查库，不能直接判定未出现",
                "nextAction": f"使用 getRegistry(hullNumber={hull}) 查先验库",
                "evidenceGap": "未查询先验库",
            }

        round_item = {
            "round": loop_count,
            "planHint": state.get("plan_hint"),
            "observation": state.get("observation_summary"),
            "reflection": handoff,
            "toolChain": out.get("tool_chain") or [],
        }
        _emit(
            event_handler,
            {
                "type": "reflection",
                "title": "ReflectAgent",
                "message": str(handoff.get("reason") or handoff.get("summary") or ""),
                "state": handoff.get("state") or ("replan" if handoff.get("handoff") == "plan" else "uncertain"),
                "round": loop_count,
                "role": "reflector",
                "evidenceGap": handoff.get("evidenceGap"),
                "nextAction": handoff.get("nextAction"),
            },
        )

        if handoff.get("handoff") == "plan" or handoff.get("replan") or str(handoff.get("state") or "") == "replan":
            if loop_count >= limit:
                return Command(
                    goto=END,
                    update={
                        "loop_count": loop_count,
                        "rounds": [round_item],
                        "reflection": {
                            "state": "uncertain",
                            "reason": f"已达最大轮次 {limit}，仍要求 replan",
                            "nextAction": handoff.get("nextAction"),
                            "evidenceGap": handoff.get("evidenceGap"),
                        },
                        "final_state": "uncertain",
                        "final_reason": f"已达最大轮次 {limit}，仍要求 replan",
                        "tool_chain": out.get("tool_chain") or [],
                        "tool_records": out.get("tool_records") or [],
                    },
                )
            return Command(
                goto="plan",
                update={
                    "loop_count": loop_count,
                    "rounds": [round_item],
                    "reflection": {
                        "state": "replan",
                        "reason": handoff.get("reason") or "需要补充证据",
                        "nextAction": handoff.get("nextAction"),
                        "evidenceGap": handoff.get("evidenceGap"),
                    },
                    "plan_hint": str(handoff.get("nextAction") or handoff.get("reason") or ""),
                    "active_agent": "plan",
                    "tool_chain": out.get("tool_chain") or [],
                    "tool_records": out.get("tool_records") or [],
                },
            )

        final_state = str(handoff.get("state") or "uncertain")
        if final_state not in {"sufficient", "conflict", "uncertain"}:
            final_state = "uncertain"
        final_reason = str(handoff.get("reason") or handoff.get("answerHint") or out.get("text") or "反思结束")
        return Command(
            goto=END,
            update={
                "loop_count": loop_count,
                "rounds": [round_item],
                "reflection": {
                    "state": final_state,
                    "reason": final_reason,
                    "answerHint": handoff.get("answerHint"),
                    "evidenceGap": handoff.get("evidenceGap"),
                },
                "final_state": final_state,
                "final_reason": final_reason,
                "tool_chain": out.get("tool_chain") or [],
                "tool_records": out.get("tool_records") or [],
            },
        )

    graph = StateGraph(AgentState)
    graph.add_node("intent", intent_node)
    graph.add_node("plan", plan_node)
    graph.add_node("observe", observe_node)
    graph.add_node("reflect", reflect_node)
    graph.add_edge(START, "intent")
    return graph.compile()


def run_sea_agent(
    question: str,
    llm: AgentLLMService,
    tools: ToolService,
    *,
    max_rounds: int = 3,
    query_top_k: int | None = None,
    broad_match_top_k: int | None = None,
    event_handler: Callable[[dict[str, Any]], None] | None = None,
) -> AgentState:
    top_k = max(1, min(20, int(query_top_k or 3)))
    broad_top_k = _normalize_broad_match_top_k(broad_match_top_k)
    app = build_sea_agent_graph(
        llm,
        tools,
        max_rounds=max_rounds,
        query_top_k=top_k,
        broad_match_top_k=broad_top_k,
        event_handler=event_handler,
    )
    initial: AgentState = {
        "question": question.strip(),
        "intent": {},
        "working_scope": {},
        "reflection": {},
        "rounds": [],
        "tool_chain": [],
        "tool_records": [],
        "plan_hint": "",
        "observation_summary": "",
        "active_agent": "intent",
        "final_state": "uncertain",
        "final_reason": "",
        "loop_count": 0,
        "max_rounds": max_rounds,
        "query_top_k": top_k,
        "broad_match_top_k": broad_top_k,
    }
    return app.invoke(initial, config={"recursion_limit": max(32, max_rounds * 20)})
