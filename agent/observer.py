"""按计划执行白名单工具并记录观察。"""
from __future__ import annotations
from typing import Any, Callable
from services import AgentLLMService
from tools import ToolService

class Observer:
    _REQUIRED_ARGUMENTS = {
        "getFrames": ("trackIds",),
        "getClip": ("trackId",),
        "getRegistry": ("hullNumber",),
        "matchHull": ("hullNumberArray",),
        "matchText": ("description", "galleryImages"),
        "matchImage": ("queryImages", "galleryImages"),
        "dedupTracks": ("tracks", "keyframesByTrack"),
    }

    def __init__(self, llm: AgentLLMService, tools: ToolService):
        self.llm = llm
        self.tools = tools

    def execute(
        self,
        plan: dict[str, Any],
        context: dict[str, Any] | None = None,
        on_delta: Callable[[str], None] | None = None,
        on_tool_event: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        scope = dict(context or {})
        observations = []
        for call in plan.get("calls") or []:
            call_id = call["id"]
            tool = call["tool"]
            if not self._condition(call.get("condition"), scope):
                result = {"ok": False, "error": "condition_not_met", "tool": tool}
                observation = {"id": call_id, "tool": tool, "skipped": True, "skipReason": "condition_not_met", "result": result}
                scope[call_id] = result
                observations.append(observation)
                self._emit_tool_event(on_tool_event, "skipped", observation)
                continue
            dependency_issue = self._dependency_issue(call.get("arguments", {}), scope)
            if dependency_issue:
                result = {"ok": False, "error": dependency_issue, "tool": tool}
                observation = {"id": call_id, "tool": tool, "skipped": True, "skipReason": dependency_issue, "result": result}
                scope[call_id] = result
                observations.append(observation)
                self._emit_tool_event(on_tool_event, "skipped", observation)
                continue
            arguments = self._resolve(call.get("arguments", {}), scope)
            argument_issue = self._required_argument_issue(tool, arguments)
            if argument_issue:
                result = {"ok": False, "error": argument_issue, "tool": tool}
                observation = {"id": call_id, "tool": tool, "skipped": True, "skipReason": argument_issue, "result": result}
                scope[call_id] = result
                observations.append(observation)
                self._emit_tool_event(on_tool_event, "skipped", observation)
                continue
            self._emit_tool_event(on_tool_event, "running", {"id": call_id, "tool": tool})
            try:
                result = self.tools.execute(tool, arguments)
                if not isinstance(result, dict):
                    result = {"ok": False, "error": "tool_result_invalid", "tool": tool}
            except Exception as error:
                result = {"ok": False, "error": f"tool_execution_failed:{error}", "tool": tool}
            observation = {"id": call_id, "tool": tool, "arguments": self._compact(arguments), "result": result}
            scope[call_id] = result
            observations.append(observation)
            self._emit_tool_event(on_tool_event, "completed" if result.get("ok") is not False else "failed", observation)
        intent = plan.get("intent") if isinstance(plan.get("intent"), dict) else {}
        summary = {
            "task": {
                "goal": plan.get("goal"),
                "expectedOutcome": intent.get("expectedOutcome"),
                "successCriteria": intent.get("successCriteria"),
                "evidenceGap": plan.get("evidenceGap"),
                "proposedState": plan.get("proposedState"),
            },
            "calls": [self._summarize_observation(item) for item in observations],
            "executedCount": sum(1 for item in observations if not item.get("skipped")),
            "failedCount": sum(1 for item in observations if (item.get("result") or {}).get("ok") is False),
            "skippedCount": sum(1 for item in observations if item.get("skipped")),
        }
        try:
            summary["modelObservation"] = self.llm.role("observer", summary, on_delta)
        except Exception as error:
            summary["modelFallback"] = str(error)
        return {"observations": observations, "scope": scope, "summary": summary}

    @classmethod
    def _dependency_issue(cls, value: Any, scope: dict[str, Any]) -> str | None:
        if isinstance(value, dict):
            if "$ref" in value:
                reference = str(value.get("$ref") or "").strip()
                if not reference:
                    return "dependency_reference_empty"
                root = reference.split(".", 1)[0]
                if root not in scope:
                    return f"dependency_missing:{reference}"
                source = scope.get(root)
                if isinstance(source, dict) and source.get("ok") is False:
                    return f"dependency_failed:{root}"
                present, resolved = cls._read_with_presence(scope, reference)
                if not present:
                    if "$default" in value:
                        resolved = value["$default"]
                    else:
                        return f"dependency_field_missing:{reference}"
                if cls._empty_dependency(resolved):
                    return f"dependency_empty:{reference}"
            for item in value.values():
                issue = cls._dependency_issue(item, scope)
                if issue:
                    return issue
        elif isinstance(value, list):
            for item in value:
                issue = cls._dependency_issue(item, scope)
                if issue:
                    return issue
        return None

    @staticmethod
    def _empty_dependency(value: Any) -> bool:
        return value is None or value == "" or (isinstance(value, (list, tuple, set, dict)) and not value)

    @classmethod
    def _required_argument_issue(cls, tool: str, arguments: dict[str, Any]) -> str | None:
        for field in cls._REQUIRED_ARGUMENTS.get(tool, ()):
            if cls._empty_dependency(arguments.get(field)):
                return f"argument_missing:{field}"
        if tool == "showEvidence" and not any(arguments.get(field) for field in ("keyframeIds", "shipSegmentIds", "registryReferenceIds")):
            return "argument_missing:evidence"
        return None

    @staticmethod
    def _emit_tool_event(callback: Callable[[dict[str, Any]], None] | None, phase: str, observation: dict[str, Any]) -> None:
        if not callback:
            return
        try:
            payload = {"phase": phase, "id": observation.get("id"), "tool": observation.get("tool")}
            if observation.get("skipped"):
                payload["skipped"] = True
                payload["error"] = observation.get("skipReason")
            elif phase in {"completed", "failed"}:
                payload["summary"] = Observer._summarize_observation(observation)
                result = observation.get("result") or {}
                payload["ok"] = result.get("ok") is not False
                if result.get("error"):
                    payload["error"] = result["error"]
            callback(payload)
        except Exception:
            pass

    @staticmethod
    def _read_with_presence(value: Any, path: str) -> tuple[bool, Any]:
        current = value
        for part in path.split("."):
            if isinstance(current, dict) and part in current:
                current = current[part]
                continue
            if isinstance(current, list) and part.isdigit():
                index = int(part)
                if 0 <= index < len(current):
                    current = current[index]
                    continue
            return False, None
        return True, current

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
        present, current = Observer._read_with_presence(value, path)
        return current if present else None

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
            summary["skipReason"] = observation.get("skipReason")
            result = observation.get("result") or {}
            if result.get("error"):
                summary["error"] = result["error"]
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
        registry_items = result.get("registryItems")
        if "registryItems" in result and isinstance(registry_items, list):
            summary["registryCount"] = len(registry_items)
        matches = result.get("matches")
        if "matches" in result and isinstance(matches, list):
            summary["matchCount"] = len(matches)
            summary["matches"] = [{key: cls._short(item.get(key)) for key in ("matchedTrackId", "matchedRegistryId", "embeddingScore", "scoreBand", "matchedKeyframeIds", "matchedRegistryReferenceIds") if item.get(key) not in (None, "", [])} for item in matches[:5]]
        exact_matches = result.get("exactMatches")
        if isinstance(exact_matches, dict):
            summary["exactMatchHullCount"] = len(exact_matches)
        for key in ("queryScope", "searchable", "found", "decision", "matchMode", "trackIds", "keyframeIds", "shipSegmentId", "registryReferenceIds", "matchedHullNumbers", "unmatchedHullNumbers", "unsearchableRegistryIds", "unsearchableTrackIds", "discardedKeyframeIds", "missingKeyframeIds", "missingRegistryReferenceIds", "totalTrackCount", "returnedTrackCount", "highThresholdShipCount", "lowThresholdShipCount", "countStability", "offset", "limit", "hasMore", "nextOffset"):
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
