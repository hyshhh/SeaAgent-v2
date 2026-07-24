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
    description: str
    galleryImages: list[dict[str, Any]] | dict[str, Any] | None = None
    topK: int | None = None


class MatchImageArgs(BaseModel):
    queryImages: list[dict[str, Any]] | dict[str, Any] | None = None
    galleryImages: list[dict[str, Any]] | dict[str, Any] | None = None
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
    """业务检索工具：绑定 ToolService.execute。"""

    def _wrap(name: str, schema: type[BaseModel], description: str) -> StructuredTool:
        def _run(**kwargs: Any) -> str:
            arguments = {k: v for k, v in kwargs.items() if v is not None}
            # timeRange list -> tuple for ToolService
            if "timeRange" in arguments and isinstance(arguments["timeRange"], list):
                tr = arguments["timeRange"]
                if len(tr) == 2:
                    arguments["timeRange"] = (float(tr[0]), float(tr[1]))
            result = tools_service.execute(name, arguments)
            if not isinstance(result, dict):
                result = {"ok": False, "error": "tool_result_invalid", "tool": name}
            if on_tool:
                try:
                    on_tool(name, arguments, result)
                except Exception:
                    pass
            return _dump(result)

        return StructuredTool.from_function(
            name=name,
            description=description,
            func=_run,
            args_schema=schema,
        )

    return [
        _wrap("getTrack", GetTrackArgs, "按时间/舷号分页检索轨迹"),
        _wrap("getFrames", GetFramesArgs, "按轨迹取关键帧"),
        _wrap("getClip", GetClipArgs, "生成轨迹片段"),
        _wrap("getRegistry", GetRegistryArgs, "按舷号查先验库"),
        StructuredTool.from_function(
            name="listRegistry",
            description="列出先验库全部条目",
            func=lambda dummy: _dump(tools_service.execute("listRegistry", {})),
            args_schema=ListRegistryArgs,
        ),
        _wrap("matchHull", MatchHullArgs, "舷号精确匹配先验库"),
        _wrap("matchText", MatchTextArgs, "文本描述匹配关键帧"),
        _wrap("matchImage", MatchImageArgs, "图像相似度匹配"),
        _wrap("verifyTarget", VerifyTargetArgs, "VLM 核验目标"),
        _wrap("showEvidence", ShowEvidenceArgs, "汇总展示证据 ID"),
        _wrap("dedupTracks", DedupTracksArgs, "跨轨迹去重计数"),
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
