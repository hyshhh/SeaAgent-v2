"""将现有 ToolService / 解析函数封装为 LangChain tools。"""
from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Callable

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from tools.target_parser import extract_hull_number, extract_target_items
from tools.time_normalizer import normalize_time_range


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return str(value)


def _dump(result: Any) -> str:
    return json.dumps(_jsonable(result), ensure_ascii=False)


class ParseTimeArgs(BaseModel):
    expression: str = Field(description="自然语言时间表达，如昨天下午")


class ParseTargetsArgs(BaseModel):
    question: str = Field(description="用户原问题，用于多目标切分")


class ExtractHullArgs(BaseModel):
    question: str = Field(description="用户原问题，用于抽取舷号")


class GetTrackArgs(BaseModel):
    timeRange: list[float] | None = Field(default=None, description="Unix 秒 [start, end]")
    hullNumber: str | None = Field(default=None, description="舷号")
    finalMatchType: str | None = Field(default=None)
    offset: int = Field(default=0)
    limit: int = Field(default=60)


class GetFramesArgs(BaseModel):
    trackIds: list[str | int] = Field(description="轨迹 ID 列表")


class GetClipArgs(BaseModel):
    trackId: str | int
    timeRange: list[float] | None = None
    scale: float | None = None


class GetRegistryArgs(BaseModel):
    hullNumber: str


class MatchHullArgs(BaseModel):
    hullNumberArray: list[str | None]


class MatchTextArgs(BaseModel):
    description: str = Field(description="外观/类别描述，如黄色无人艇")
    galleryImages: list[dict[str, Any]] | dict[str, Any] | None = Field(
        default=None,
        description="可选。关键帧列表或 getFrames 结果；省略时自动使用本轮最近 getFrames/listRegistry 结果",
    )
    topK: int | None = None


class MatchImageArgs(BaseModel):
    queryImages: list[dict[str, Any]] | dict[str, Any] | None = Field(
        default=None,
        description="可选。查询侧图像；省略时自动用最近 getFrames/listRegistry",
    )
    galleryImages: list[dict[str, Any]] | dict[str, Any] | None = Field(
        default=None,
        description="可选。图库侧图像；省略时自动用最近另一侧结果",
    )
    topK: int | None = None


class VerifyTargetArgs(BaseModel):
    description: str | None = None
    registryReferenceIds: list[str] | None = None
    keyframeIds: list[str] | None = None
    shipSegmentIds: list[str] | None = None


class ShowEvidenceArgs(BaseModel):
    keyframeIds: list[str] | None = None
    shipSegmentIds: list[str] | None = None
    registryReferenceIds: list[str] | None = None


class DedupTracksArgs(BaseModel):
    tracks: list[dict[str, Any]]
    keyframesByTrack: dict[str, Any]


class ListRegistryArgs(BaseModel):
    dummy: str | None = Field(default=None, description="无需参数，可忽略")


class LoadSkillArgs(BaseModel):
    skillId: str = Field(description="catalog 中的 skill id")


def build_intent_tools(reference_time: datetime | None = None) -> list[StructuredTool]:
    now = reference_time or datetime.now().astimezone()

    def parse_time(expression: str) -> str:
        rng = normalize_time_range(expression, now=now)
        return _dump({
            "ok": True,
            "timeRange": list(rng) if rng else None,
            "expression": expression,
            "hint": "确认后写入意图 timeRange",
        })

    def parse_targets(question: str) -> str:
        return _dump({
            "ok": True,
            "targetItems": extract_target_items(question),
            "hint": "确认后写入意图 targetItems",
        })

    def extract_hull(question: str) -> str:
        return _dump({"ok": True, "hullNumber": extract_hull_number(question)})

    return [
        StructuredTool.from_function(
            name="parseTime",
            description="将自然语言时间归一化为 Unix 秒区间 [start, end]",
            func=parse_time,
            args_schema=ParseTimeArgs,
        ),
        StructuredTool.from_function(
            name="parseTargets",
            description="从问题中切分多个船舶目标",
            func=parse_targets,
            args_schema=ParseTargetsArgs,
        ),
        StructuredTool.from_function(
            name="extractHull",
            description="从问题中抽取疑似舷号",
            func=extract_hull,
            args_schema=ExtractHullArgs,
        ),
    ]


def build_observe_tools(
    tools_service: Any,
    on_tool: Callable[[str, dict[str, Any], dict[str, Any]], None] | None = None,
) -> list[StructuredTool]:
    """业务检索工具：绑定 ToolService.execute。

    兼容旧版 $ref 行为：本轮 getFrames / listRegistry / getRegistry 的结果会缓存，
    matchText/matchImage 未传 galleryImages/queryImages 时自动注入。
    """
    # 本轮观察会话缓存（单次 ObserveAgent 内共享）
    session: dict[str, Any] = {
        "keyframes": None,
        "registryReferences": None,
        "tracks": None,
        "keyframesByTrack": None,
    }

    def _remember(name: str, result: dict[str, Any]) -> None:
        if not isinstance(result, dict) or result.get("ok") is False:
            return
        if name == "getFrames":
            if result.get("keyframes") is not None:
                session["keyframes"] = result.get("keyframes")
            if result.get("keyframesByTrack") is not None:
                session["keyframesByTrack"] = result.get("keyframesByTrack")
        elif name in {"listRegistry", "getRegistry"}:
            if result.get("registryReferences") is not None:
                session["registryReferences"] = result.get("registryReferences")
        elif name == "getTrack":
            if result.get("tracks") is not None:
                session["tracks"] = result.get("tracks")

    def _auto_fill(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        args = dict(arguments)
        if name == "matchText" and args.get("galleryImages") is None:
            # 描述匹配轨迹关键帧优先；否则用先验库参考图
            if session.get("keyframes") is not None:
                args["galleryImages"] = session["keyframes"]
            elif session.get("registryReferences") is not None:
                args["galleryImages"] = session["registryReferences"]
        if name == "matchImage":
            if args.get("queryImages") is None and session.get("keyframes") is not None:
                args["queryImages"] = session["keyframes"]
            if args.get("galleryImages") is None and session.get("registryReferences") is not None:
                args["galleryImages"] = session["registryReferences"]
            # 反过来：查询库图、匹配轨迹
            if args.get("queryImages") is None and session.get("registryReferences") is not None:
                args["queryImages"] = session["registryReferences"]
            if args.get("galleryImages") is None and session.get("keyframes") is not None:
                args["galleryImages"] = session["keyframes"]
        if name == "dedupTracks":
            if args.get("tracks") is None and session.get("tracks") is not None:
                args["tracks"] = session["tracks"]
            if args.get("keyframesByTrack") is None and session.get("keyframesByTrack") is not None:
                args["keyframesByTrack"] = session["keyframesByTrack"]
        if name == "showEvidence":
            # 未指定证据 ID 时，从最近关键帧取一批展示
            if not args.get("keyframeIds") and isinstance(session.get("keyframes"), list):
                ids = [
                    str(item.get("keyframeId"))
                    for item in session["keyframes"][:12]
                    if isinstance(item, dict) and item.get("keyframeId") is not None
                ]
                if ids:
                    args["keyframeIds"] = ids
        return args

    def _wrap(name: str, schema: type[BaseModel], description: str) -> StructuredTool:
        def _run(**kwargs: Any) -> str:
            arguments = {k: v for k, v in kwargs.items() if v is not None}
            # timeRange list -> tuple for ToolService
            if "timeRange" in arguments and isinstance(arguments["timeRange"], list):
                tr = arguments["timeRange"]
                if len(tr) == 2:
                    arguments["timeRange"] = (float(tr[0]), float(tr[1]))
            arguments = _auto_fill(name, arguments)
            result = tools_service.execute(name, arguments)
            if not isinstance(result, dict):
                result = {"ok": False, "error": "tool_result_invalid", "tool": name}
            _remember(name, result)
            if on_tool:
                try:
                    # 事件里不塞完整关键帧大对象，只报模型实际传入的关键参数
                    event_args = {k: v for k, v in arguments.items() if k not in {"galleryImages", "queryImages", "keyframesByTrack"} or not isinstance(v, (list, dict))}
                    if "galleryImages" in arguments and "galleryImages" not in event_args:
                        event_args["galleryImages"] = f"<auto:{len(arguments.get('galleryImages') or []) if isinstance(arguments.get('galleryImages'), list) else 'obj'}>"
                    if "queryImages" in arguments and "queryImages" not in event_args:
                        event_args["queryImages"] = f"<auto:{len(arguments.get('queryImages') or []) if isinstance(arguments.get('queryImages'), list) else 'obj'}>"
                    on_tool(name, event_args, result)
                except Exception:
                    pass
            return _dump(result)

        return StructuredTool.from_function(
            name=name,
            description=description,
            func=_run,
            args_schema=schema,
        )

    def _list_registry(dummy: str | None = None) -> str:
        result = tools_service.execute("listRegistry", {})
        if isinstance(result, dict):
            _remember("listRegistry", result)
        if on_tool:
            try:
                on_tool("listRegistry", {}, result if isinstance(result, dict) else {})
            except Exception:
                pass
        return _dump(result)

    return [
        _wrap("getTrack", GetTrackArgs, "按时间/舷号分页检索轨迹"),
        _wrap("getFrames", GetFramesArgs, "按轨迹取关键帧；结果会自动供给后续 matchText"),
        _wrap("getClip", GetClipArgs, "生成轨迹片段"),
        _wrap("getRegistry", GetRegistryArgs, "按舷号查先验库"),
        StructuredTool.from_function(
            name="listRegistry",
            description="列出先验库全部条目；结果可自动供给 matchText/matchImage",
            func=_list_registry,
            args_schema=ListRegistryArgs,
        ),
        _wrap("matchHull", MatchHullArgs, "舷号精确匹配先验库"),
        _wrap(
            "matchText",
            MatchTextArgs,
            "文本描述匹配关键帧或库图。galleryImages 可省略：默认用本轮 getFrames 的 keyframes",
        ),
        _wrap(
            "matchImage",
            MatchImageArgs,
            "图像相似度匹配。queryImages/galleryImages 可省略，自动用本轮关键帧与库参考图",
        ),
        _wrap("verifyTarget", VerifyTargetArgs, "VLM 核验目标"),
        _wrap("showEvidence", ShowEvidenceArgs, "汇总展示证据 ID"),
        _wrap("dedupTracks", DedupTracksArgs, "跨轨迹去重计数；tracks/keyframes 可省略用本轮缓存"),
    ]


def build_load_skill_tool(agent_key: str, load_fn: Callable[[str], dict[str, Any]]) -> StructuredTool:
    def _run(skillId: str) -> str:
        return _dump(load_fn(skillId))

    return StructuredTool.from_function(
        name="loadSkill",
        description="按需加载可选 skill 全文",
        func=_run,
        args_schema=LoadSkillArgs,
    )
