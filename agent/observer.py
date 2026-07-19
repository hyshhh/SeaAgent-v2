"""按计划执行白名单工具并记录观察。"""
from __future__ import annotations
from typing import Any, Callable
from services import AgentLLMService
from tools import ToolService

class Observer:
    def __init__(self, llm: AgentLLMService, tools: ToolService):
        self.llm = llm
        self.tools = tools

    def execute(self, plan: dict[str, Any], context: dict[str, Any] | None = None, on_delta: Callable[[str], None] | None = None) -> dict[str, Any]:
        scope = dict(context or {})
        observations = []
        for call in plan["calls"]:
            if not self._condition(call.get("condition"), scope):
                observations.append({"id": call["id"], "tool": call["tool"], "skipped": True})
                continue
            arguments = self._resolve(call.get("arguments", {}), scope)
            result = self.tools.execute(call["tool"], arguments)
            scope[call["id"]] = result
            observations.append({"id": call["id"], "tool": call["tool"], "arguments": self._compact(arguments), "result": result})
        summary = {"calls": [self._summarize_observation(item) for item in observations]}
        try:
            summary["modelObservation"] = self.llm.role("observer", summary, on_delta)
        except Exception as error:
            summary["modelFallback"] = str(error)
        return {"observations": observations, "scope": scope, "summary": summary}

    def _resolve(self, value: Any, scope: dict[str, Any]) -> Any:
        if isinstance(value, list):
            return [self._resolve(item, scope) for item in value]
        if not isinstance(value, dict):
            return value
        if "$ref" not in value:
            return {key: self._resolve(item, scope) for key, item in value.items()}
        resolved = self._read(scope, value["$ref"])
        if value.get("$map") and isinstance(resolved, list):
            resolved = [self._read(item, value["$map"]) for item in resolved]
        if value.get("$compact") and isinstance(resolved, list):
            resolved = [item for item in resolved if item not in (None, "", [])]
        if value.get("$list"):
            resolved = [] if resolved is None else resolved if isinstance(resolved, list) else [resolved]
        return value.get("$default") if resolved is None and "$default" in value else resolved

    def _condition(self, condition: dict[str, Any] | None, scope: dict[str, Any]) -> bool:
        if not condition:
            return True
        current = self._read(scope, condition["ref"])
        if "equals" in condition:
            return current == condition["equals"]
        if "in" in condition:
            return current in condition["in"]
        return bool(current)

    @staticmethod
    def _read(value: Any, path: str) -> Any:
        current = value
        for part in path.split("."):
            if isinstance(current, dict):
                current = current.get(part)
            else:
                return None
        return current

    @staticmethod
    def _compact(arguments: dict[str, Any]) -> dict[str, Any]:
        compact = {}
        for key, value in arguments.items():
            if isinstance(value, list) and value and isinstance(value[0], dict):
                compact[key] = [item.get("keyframeId") or item.get("referenceId") or item.get("trackId") for item in value]
            elif isinstance(value, dict) and len(str(value)) > 500:
                compact[key] = list(value)
            else:
                compact[key] = value
        return compact

    @classmethod
    def _summarize_observation(cls, observation: dict[str, Any]) -> dict[str, Any]:
        summary = {"id": observation["id"], "tool": observation["tool"], "skipped": observation.get("skipped", False)}
        if summary["skipped"]:
            return summary
        result = observation.get("result", {})
        summary["ok"] = result.get("ok")
        if result.get("error"):
            summary["error"] = result["error"]
        tracks = result.get("tracks")
        if "tracks" in result and isinstance(tracks, list):
            summary["trackCount"] = len(tracks)
            summary["tracks"] = [{key: cls._short(item.get(key)) for key in ("trackId", "startTime", "endTime", "finalHullNumber", "finalDescription", "finalMatchType") if item.get(key) not in (None, "")} for item in tracks[:5]]
        keyframes = result.get("keyframes")
        if "keyframes" in result and isinstance(keyframes, list):
            summary["keyframeCount"] = len(keyframes)
            summary["keyframes"] = [{key: cls._short(item.get(key)) for key in ("keyframeId", "trackId", "timestamp", "description", "isEmbedded") if item.get(key) not in (None, "")} for item in keyframes[:5]]
        matches = result.get("matches")
        if "matches" in result and isinstance(matches, list):
            summary["matchCount"] = len(matches)
            summary["matches"] = [{key: cls._short(item.get(key)) for key in ("matchedTrackId", "matchedRegistryId", "embeddingScore", "scoreBand", "matchedKeyframeIds", "matchedRegistryReferenceIds") if item.get(key) not in (None, "", [])} for item in matches[:5]]
        for key in ("queryScope", "searchable", "found", "decision", "matchMode", "trackIds", "keyframeIds", "unsearchableTrackIds", "discardedKeyframeIds", "missingKeyframeIds", "totalTrackCount", "returnedTrackCount", "offset", "limit", "hasMore", "nextOffset"):
            value = result.get(key)
            if value not in (None, "", []):
                summary[key] = cls._short(value)
        return summary

    @staticmethod
    def _short(value: Any) -> Any:
        if isinstance(value, str):
            return value if len(value) <= 160 else value[:157] + "..."
        if isinstance(value, list):
            return value[:10]
        return value
