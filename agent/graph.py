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
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
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


def _last_ai_text(messages: list[Any]) -> str:
    for message in reversed(messages or []):
        if isinstance(message, AIMessage):
            content = message.content
            if isinstance(content, list):
                parts = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        parts.append(str(item.get("text") or ""))
                    else:
                        parts.append(str(item))
                return "\n".join(parts).strip()
            return str(content or "").strip()
    return ""


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


def build_sea_agent_graph(
    llm: AgentLLMService,
    tools: ToolService,
    *,
    max_rounds: int = 3,
    query_top_k: int | None = None,
    event_handler: Callable[[dict[str, Any]], None] | None = None,
):
    """编译四 Agent LangGraph。"""
    model = build_chat_model(llm)
    reference_time = datetime.now().astimezone()
    default_top_k = int(query_top_k or 3)

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

    def _default_plan_calls(intent: dict[str, Any], top_k: int) -> list[dict[str, Any]]:
        """模型未给出 calls 时的最小可执行链（结构化 $ref，不是业务硬编码分支表）。"""
        hull = str(intent.get("hullNumber") or "").strip()
        description = str(intent.get("description") or "").strip()
        time_range = intent.get("timeRange")
        operation = str(intent.get("operation") or "list")
        target_scope = str(intent.get("targetScope") or "track_memory")
        top = max(1, min(20, int(top_k or 3)))

        if target_scope == "registry":
            if hull:
                return [{"id": "registry", "tool": "getRegistry", "arguments": {"hullNumber": hull}}]
            if description:
                # 对齐 old：listRegistry → matchText；优先参考图，缺图时用 registryItems 关键字弱匹配
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
            return [{"id": "registry", "tool": "listRegistry", "arguments": {}}]

        track_args: dict[str, Any] = {"offset": 0, "limit": 60}
        if time_range:
            track_args["timeRange"] = time_range
        if hull:
            track_args["hullNumber"] = hull
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
            calls.append({"id": "matchHull", "tool": "matchHull", "arguments": {"hullNumberArray": [hull]}})
            return calls
        # 列表/存在：先取轨迹，再取关键帧供展示
        calls.append({"id": "frames", "tool": "getFrames", "arguments": {"trackIds": {"$ref": "tracks.trackIds"}}})
        return calls

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
        _emit(
            event_handler,
            {
                "type": "status",
                "title": title,
                "message": f"{title} 开始（skills: {', '.join(skill_ids) or 'core'}）",
                "enabledSkills": skill_ids,
                "role": role,
                "round": round_number,
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
                    # Intent 全局一轮；Plan/Observe/Reflect 从 1 计
                    "round": 0 if role == "intent" else max(1, round_number),
                },
            )

        agent = create_agent(
            model,
            agent_tools,
            system_prompt=prompt,
            name=name,
        )
        invoke_error = ""
        try:
            result = agent.invoke(
                {"messages": [HumanMessage(content=user_content)]},
                config={"recursion_limit": recursion_limit},
            )
            messages = result.get("messages") or []
        except Exception as error:
            # 内层 ReAct 触顶或模型异常时不炸穿外层图，交给节点级 handoff 兜底
            invoke_error = str(error)
            messages = []
            _emit(
                event_handler,
                {
                    "type": "status",
                    "title": title,
                    "message": f"{title} 提前结束：{invoke_error[:160]}",
                    "role": role,
                    "round": round_number,
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

        text = _last_ai_text(messages)
        if invoke_error and not text:
            text = f"{title} 未正常结束：{invoke_error[:200]}"
        event_round = 0 if role == "intent" else max(1, round_number or 1)
        if text:
            _emit(
                event_handler,
                {
                    "type": "agent_delta",
                    "title": title,
                    "message": "",
                    "role": role,
                    "round": event_round,
                    "delta": text[:500] + ("…" if len(text) > 500 else ""),
                },
            )
        if role:
            end_event: dict[str, Any] = {
                "type": "agent_end",
                "title": title,
                "message": text[:300] if text else f"{title} 完成",
                "role": role,
                "round": event_round,
                "modelSummary": {
                    "summary": text[:500] if text else "",
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
                # 优先使用 handoff.calls；否则从 planHint 文本抽工具名作展示
                handoff_calls = PlanExecutor.sanitize_calls((handoff or {}).get("calls"))
                if handoff_calls:
                    end_event["calls"] = [
                        {"id": c["id"], "tool": c["tool"], "arguments": c.get("arguments") or {}}
                        for c in handoff_calls
                    ]
                else:
                    tools_in_hint = []
                    for name_candidate in (
                        "getTrack", "getFrames", "getClip", "getRegistry", "listRegistry",
                        "matchHull", "matchText", "matchImage", "verifyTarget", "showEvidence", "dedupTracks",
                    ):
                        blob = f"{(handoff or {}).get('planHint') or ''} {(handoff or {}).get('goal') or ''} {text}"
                        if name_candidate in blob:
                            tools_in_hint.append(name_candidate)
                    if tools_in_hint and not end_event["calls"]:
                        end_event["calls"] = [{"id": f"plan-{i+1}", "tool": tool_name} for i, tool_name in enumerate(tools_in_hint)]
                end_event["planBlueprint"] = [
                    {
                        "stepId": str(call.get("id") or f"step-{i+1}"),
                        "title": f"执行 {call.get('tool')}",
                        "tools": [call.get("tool")],
                        "optional": False,
                    }
                    for i, call in enumerate(end_event["calls"])
                ]
                if invoke_error and not handoff:
                    end_event["fallback"] = "规划超时，已使用默认检索计划"
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
        }

    def intent_node(state: AgentState) -> Command:
        question = state.get("question") or ""
        user = json.dumps(
            {
                "task": "识别意图并必须调用 handoff_to_plan",
                "question": question,
                "referenceTime": reference_time.isoformat(timespec="seconds"),
                "queryTopK": state.get("query_top_k") or default_top_k,
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
        if not intent.get("targetItems") and inferred.get("targetItems"):
            intent["targetItems"] = inferred["targetItems"]
        if intent.get("targetItems"):
            intent["targetItems"] = normalize_target_items(intent.get("targetItems"))
        if not intent.get("operation") or intent.get("operation") not in {
            "existence", "list", "time", "count", "explain",
        }:
            intent["operation"] = inferred.get("operation") or "list"
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
        if not intent.get("expectedOutcome"):
            intent["expectedOutcome"] = inferred.get("expectedOutcome")
        if not intent.get("successCriteria"):
            intent["successCriteria"] = inferred.get("successCriteria") or "工具结果足以回答用户问题"
        if not intent.get("nextAgentFocus"):
            intent["nextAgentFocus"] = inferred.get("nextAgentFocus")
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
        scope_keys = list((state.get("working_scope") or {}).keys())
        user = json.dumps(
            {
                "task": "规划本轮检索；必须调用 handoff_to_observe(goal, calls, planHint) 或 handoff_to_reflect",
                "question": state.get("question"),
                "intent": intent,
                "loop": loop_count,
                "round": round_number,
                "maxRounds": state.get("max_rounds") or max_rounds,
                "queryTopK": state.get("query_top_k") or default_top_k,
                "previousReflection": reflection,
                "workingScopeKeys": scope_keys,
                "availableTools": [
                    "getTrack", "getFrames", "getClip", "getRegistry", "listRegistry",
                    "matchHull", "matchText", "matchImage", "verifyTarget", "showEvidence", "dedupTracks",
                ],
                "callSchema": {
                    "id": "本轮唯一调用 id，如 tracks/frames/match",
                    "tool": "工具名",
                    "arguments": "参数对象；跨步骤用 {\"$ref\": \"{callId}.{field}\"}",
                },
                "examples": [
                    {
                        "desc": "描述存在判断",
                        "calls": [
                            {"id": "tracks", "tool": "getTrack", "arguments": {"timeRange": intent.get("timeRange"), "offset": 0, "limit": 60}},
                            {"id": "frames", "tool": "getFrames", "arguments": {"trackIds": {"$ref": "tracks.trackIds"}}},
                            {"id": "match", "tool": "matchText", "arguments": {
                                "description": intent.get("description") or "目标描述",
                                "galleryImages": {"$ref": "frames.keyframes"},
                                "topK": state.get("query_top_k") or default_top_k,
                            }},
                        ],
                    },
                    {
                        "desc": "舷号查询",
                        "calls": [
                            {"id": "tracks", "tool": "getTrack", "arguments": {"hullNumber": intent.get("hullNumber"), "timeRange": intent.get("timeRange")}},
                            {"id": "frames", "tool": "getFrames", "arguments": {"trackIds": {"$ref": "tracks.trackIds"}}},
                        ],
                    },
                    {
                        "desc": "先验库描述筛选",
                        "calls": [
                            {"id": "registry", "tool": "listRegistry", "arguments": {}},
                            {"id": "match", "tool": "matchText", "arguments": {
                                "description": intent.get("description") or "目标描述",
                                "galleryImages": {"$ref": "registry.registryReferences"},
                                "topK": state.get("query_top_k") or default_top_k,
                            }},
                        ],
                    },
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
        plan_calls = PlanExecutor.sanitize_calls(handoff.get("calls"))

        # 模型未给出 calls 时，按意图生成最小可执行链（仍是结构化 calls+$ref，非硬编码业务流程分支）
        if target != "reflect" and not plan_calls:
            plan_calls = _default_plan_calls(intent, state.get("query_top_k") or default_top_k)
            if not plan_hint:
                plan_hint = " → ".join(c["tool"] for c in plan_calls) or "无步骤"
            if out.get("invoke_error"):
                plan_hint = f"[规划兜底] {plan_hint}"

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
        plan_calls = PlanExecutor.sanitize_calls(state.get("plan_calls") or [])
        if not plan_calls:
            # 无 calls 时再兜底一次
            plan_calls = _default_plan_calls(state.get("intent") or {}, state.get("query_top_k") or default_top_k)

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
                        "ok": item.get("ok") is not False and not item.get("skipped"),
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
        # 成功证据：working_scope 中存在 ok 且含 tracks/matches/keyframes/registry 等
        has_tool_evidence = False
        evidence_bits: list[str] = []
        for key, value in scope.items():
            if not isinstance(value, dict) or value.get("ok") is False:
                continue
            if any(value.get(field) not in (None, [], {}) for field in (
                "tracks", "trackIds", "keyframes", "keyframeIds", "matches",
                "registryItems", "registryReferences", "exactMatches",
                "highThresholdShipCount", "uniqueCount", "count",
            )):
                has_tool_evidence = True
                if value.get("tracks") is not None:
                    evidence_bits.append(f"{key}:tracks={len(value.get('tracks') or [])}")
                if value.get("keyframes") is not None:
                    evidence_bits.append(f"{key}:keyframes={len(value.get('keyframes') or [])}")
                if value.get("matches") is not None:
                    evidence_bits.append(f"{key}:matches={len(value.get('matches') or [])}")
                if value.get("registryItems") is not None or value.get("registryReferences") is not None:
                    evidence_bits.append(f"{key}:registry")
                if value.get("exactMatches") is not None:
                    evidence_bits.append(f"{key}:exactHull")
        # 工具链非空也算有执行痕迹
        if not has_tool_evidence and (state.get("tool_chain") or state.get("tool_records")):
            has_tool_evidence = any(
                isinstance(r, dict) and r.get("ok") is not False and not r.get("skipped")
                for r in (state.get("tool_records") or [])
            )
        user = json.dumps(
            {
                "task": "判定是否退出。replan→handoff_to_plan_replan；否则必须 handoff_finish",
                "question": state.get("question"),
                "expectedOutcome": intent.get("expectedOutcome"),
                "successCriteria": intent.get("successCriteria"),
                "observation": state.get("observation_summary"),
                "workingScopeKeys": list(scope.keys()),
                "evidenceHints": evidence_bits[:20],
                "hasToolEvidence": has_tool_evidence,
                "toolChain": state.get("tool_chain") or [],
                "loop": loop_count,
                "maxRounds": limit,
                "notes": [
                    "hasToolEvidence=true 表示已有工具成功结果，勿说「没有任何成功工具结果」",
                    "存在判断：getTrack 返回 0 条轨迹本身就是充分否定证据 → sufficient，结论写「未出现/未找到」",
                    "先验库描述：listRegistry 有库项后应看 matchText；仅 list 且无匹配步骤时若模型只列了库，勿谎称已筛选",
                    "无任何工具执行痕迹时禁止 sufficient",
                    "接近 maxRounds 仍不足 → uncertain",
                    "证据矛盾 → conflict",
                    "描述匹配已有 matches 且非空 → 倾向 sufficient 或按分数解释",
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
        # 未调用 handoff 时的硬兜底，保证图一定结束或 replan
        if not handoff:
            if loop_count < limit and not has_tool_evidence:
                handoff = {
                    "handoff": "plan",
                    "replan": True,
                    "reason": "本轮未获得可用工具证据",
                    "nextAction": "补充 getTrack/getFrames 或匹配工具",
                    "evidenceGap": "working_scope 为空",
                }
            elif has_tool_evidence:
                handoff = {
                    "handoff": "finish",
                    "state": "sufficient",
                    "reason": "已有工具证据，模型未显式 handoff，按充分结束",
                    "answerHint": state.get("observation_summary") or "",
                }
            else:
                handoff = {
                    "handoff": "finish",
                    "state": "uncertain",
                    "reason": "无工具证据且无法继续",
                    "answerHint": state.get("observation_summary") or "",
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
    event_handler: Callable[[dict[str, Any]], None] | None = None,
) -> AgentState:
    top_k = max(1, min(20, int(query_top_k or 3)))
    app = build_sea_agent_graph(
        llm,
        tools,
        max_rounds=max_rounds,
        query_top_k=top_k,
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
    }
    return app.invoke(initial, config={"recursion_limit": max(32, max_rounds * 20)})
