"""按 PlanAgent 产出的 calls 确定性执行工具（对齐 old Observer）。

完整结果写入 working_scope，供后续 $ref 与最终合成；
回传给模型 / Reflect 的仅是压缩摘要，不把关键帧大 JSON 塞进对话。
"""
from __future__ import annotations

from typing import Any, Callable


class PlanExecutor:
    """确定性执行器：解析 $ref、校验依赖、调用 ToolService。"""

    _REQUIRED_ARGUMENTS = {
        "getFrames": ("trackIds",),
        "getClip": ("trackId",),
        "getRegistry": ("hullNumber",),
        "matchHull": ("hullNumberArray",),
        "matchText": ("description", "galleryImages"),
        "matchImage": ("queryImages", "galleryImages"),
        "dedupTracks": ("tracks", "keyframesByTrack"),
    }

    def __init__(self, tools: Any):
        self.tools = tools

    def execute(
        self,
        calls: list[dict[str, Any]],
        scope: dict[str, Any] | None = None,
        on_tool_event: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        working = dict(scope or {})
        observations: list[dict[str, Any]] = []
        tool_chain: list[str] = []
        tool_records: list[dict[str, Any]] = []

        for index, call in enumerate(calls or []):
            if not isinstance(call, dict):
                continue
            call_id = str(call.get("id") or f"step-{index + 1}")
            tool = str(call.get("tool") or "").strip()
            if not tool:
                continue

            if not self._condition(call.get("condition"), working):
                result = {"ok": False, "error": "condition_not_met", "tool": tool}
                observation = {
                    "id": call_id,
                    "tool": tool,
                    "skipped": True,
                    "skipReason": "condition_not_met",
                    "result": result,
                }
                working[call_id] = result
                observations.append(observation)
                self._emit(on_tool_event, "skipped", observation)
                continue

            dependency_issue = self._dependency_issue(call.get("arguments", {}), working)
            if dependency_issue:
                result = {"ok": False, "error": dependency_issue, "tool": tool}
                observation = {
                    "id": call_id,
                    "tool": tool,
                    "skipped": True,
                    "skipReason": dependency_issue,
                    "result": result,
                }
                working[call_id] = result
                observations.append(observation)
                self._emit(on_tool_event, "skipped", observation)
                continue

            arguments = self._resolve(call.get("arguments", {}), working)
            argument_issue = self._required_argument_issue(tool, arguments)
            if argument_issue:
                result = {"ok": False, "error": argument_issue, "tool": tool}
                observation = {
                    "id": call_id,
                    "tool": tool,
                    "skipped": True,
                    "skipReason": argument_issue,
                    "result": result,
                }
                working[call_id] = result
                observations.append(observation)
                self._emit(on_tool_event, "skipped", observation)
                continue

            self._emit(on_tool_event, "running", {"id": call_id, "tool": tool, "arguments": self._compact_args(arguments)})
            try:
                result = self.tools.execute(tool, arguments)
                if not isinstance(result, dict):
                    result = {"ok": False, "error": "tool_result_invalid", "tool": tool}
            except Exception as error:
                result = {"ok": False, "error": f"tool_execution_failed:{error}", "tool": tool}

            observation = {
                "id": call_id,
                "tool": tool,
                "arguments": self._compact_args(arguments),
                "result": result,
                "skipped": False,
            }
            working[call_id] = result
            observations.append(observation)
            tool_chain.append(tool)
            summary = self.summarize_observation(observation)
            tool_records.append({
                "id": call_id,
                "tool": tool,
                "arguments": observation.get("arguments") or {},
                "result": result,
                "summary": summary,
                "ok": result.get("ok") is not False,
                "skipped": False,
                "phase": "completed" if result.get("ok") is not False else "failed",
                "error": result.get("error"),
                **summary,
            })
            self._emit(
                on_tool_event,
                "completed" if result.get("ok") is not False else "failed",
                observation,
            )

        summary = {
            "calls": [self.summarize_observation(item) for item in observations],
            "executedCount": sum(1 for item in observations if not item.get("skipped")),
            "failedCount": sum(1 for item in observations if (item.get("result") or {}).get("ok") is False),
            "skippedCount": sum(1 for item in observations if item.get("skipped")),
        }
        return {
            "observations": observations,
            "scope": working,
            "summary": summary,
            "tool_chain": tool_chain,
            "tool_records": tool_records,
        }

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
                        # $default 可以是字面量或嵌套 $ref
                        default_value = value["$default"]
                        if isinstance(default_value, dict) and "$ref" in default_value:
                            issue = cls._dependency_issue(default_value, scope)
                            if issue:
                                return issue
                            resolved = cls._read(scope, str(default_value.get("$ref") or ""))
                        else:
                            resolved = default_value
                    else:
                        return f"dependency_field_missing:{reference}"
                if cls._empty_dependency(resolved):
                    if "$default" in value:
                        default_value = value["$default"]
                        if isinstance(default_value, dict) and "$ref" in default_value:
                            issue = cls._dependency_issue(default_value, scope)
                            if issue:
                                return issue
                        # 空主引用时允许回退 default，不在此判 empty
                    else:
                        return f"dependency_empty:{reference}"
            for item in value.values():
                if isinstance(value, dict) and "$ref" in value and item is value.get("$default"):
                    continue
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
        if tool == "showEvidence" and not any(
            arguments.get(field) for field in ("keyframeIds", "shipSegmentIds", "registryReferenceIds")
        ):
            return "argument_missing:evidence"
        return None

    @staticmethod
    def _emit(callback: Callable[[dict[str, Any]], None] | None, phase: str, observation: dict[str, Any]) -> None:
        if not callback:
            return
        try:
            payload = {
                "phase": phase,
                "id": observation.get("id"),
                "tool": observation.get("tool"),
            }
            if isinstance(observation.get("arguments"), dict):
                payload["arguments"] = observation["arguments"]
            if observation.get("skipped"):
                payload["skipped"] = True
                payload["error"] = observation.get("skipReason")
            elif phase in {"completed", "failed"}:
                payload["summary"] = PlanExecutor.summarize_observation(observation)
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
        if self._empty_dependency(resolved) and "$default" in value:
            default_value = value["$default"]
            if isinstance(default_value, dict) and "$ref" in default_value:
                resolved = self._resolve(default_value, scope)
            else:
                resolved = default_value
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
        if not isinstance(condition, dict) or "ref" not in condition:
            return True
        current = self._read(scope, condition["ref"])
        if "equals" in condition:
            return current == condition["equals"]
        if "in" in condition:
            return current in condition["in"]
        return bool(current)

    @staticmethod
    def _read(value: Any, path: str) -> Any:
        present, current = PlanExecutor._read_with_presence(value, path)
        return current if present else None

    @staticmethod
    def _compact_args(arguments: dict[str, Any]) -> dict[str, Any]:
        compact = {}
        for key, value in (arguments or {}).items():
            if isinstance(value, list) and value and isinstance(value[0], dict):
                compact[key] = [
                    item.get("keyframeId") or item.get("referenceId") or item.get("trackId")
                    for item in value
                ]
            elif isinstance(value, dict) and len(str(value)) > 500:
                compact[key] = f"<object:{len(value)} keys>"
            else:
                compact[key] = value
        return compact

    @classmethod
    def summarize_observation(cls, observation: dict[str, Any]) -> dict[str, Any]:
        summary: dict[str, Any] = {
            "id": observation.get("id"),
            "tool": observation.get("tool"),
            "skipped": bool(observation.get("skipped")),
        }
        if summary["skipped"]:
            summary["skipReason"] = observation.get("skipReason")
            result = observation.get("result") or {}
            if result.get("error"):
                summary["error"] = result["error"]
            return summary
        result = observation.get("result") or {}
        summary["ok"] = result.get("ok") is not False
        if result.get("error"):
            summary["error"] = result["error"]
        tracks = result.get("tracks")
        if isinstance(tracks, list):
            summary["trackCount"] = len(tracks)
            summary["trackIds"] = [
                item.get("trackId") for item in tracks[:20] if isinstance(item, dict) and item.get("trackId") is not None
            ]
        keyframes = result.get("keyframes")
        if isinstance(keyframes, list):
            summary["keyframeCount"] = len(keyframes)
            summary["keyframeIds"] = [
                item.get("keyframeId")
                for item in keyframes[:20]
                if isinstance(item, dict) and item.get("keyframeId") is not None
            ]
        matches = result.get("matches")
        if isinstance(matches, list):
            summary["matchCount"] = len(matches)
        # 库项数与参考图数分开统计，避免 searchable 参考图为 0 时把库项数盖成 0
        if isinstance(result.get("registryItems"), list):
            summary["registryCount"] = len(result["registryItems"])
            summary["registryItemCount"] = len(result["registryItems"])
        if isinstance(result.get("registryReferences"), list):
            summary["registryReferenceCount"] = len(result["registryReferences"])
            if summary.get("registryCount") is None:
                summary["registryCount"] = len(result["registryReferences"])
        if isinstance(result.get("exactMatches"), dict):
            summary["exactMatchHullCount"] = len(result["exactMatches"])
        for key in (
            "found", "decision", "matchMode", "totalTrackCount", "returnedTrackCount",
            "highThresholdShipCount", "lowThresholdShipCount", "hasMore", "offset", "limit",
            "matchedHullNumbers", "shipSegmentId",
        ):
            if result.get(key) not in (None, "", []):
                summary[key] = result.get(key)
        return summary

    @staticmethod
    def sanitize_calls(calls: Any, *, max_calls: int = 8) -> list[dict[str, Any]]:
        if not isinstance(calls, list):
            return []
        cleaned: list[dict[str, Any]] = []
        used: set[str] = set()
        for index, item in enumerate(calls[:max_calls]):
            if not isinstance(item, dict):
                continue
            tool = str(item.get("tool") or "").strip()
            if not tool:
                continue
            call_id = str(item.get("id") or f"step-{index + 1}").strip() or f"step-{index + 1}"
            base = call_id
            n = 1
            while call_id in used:
                n += 1
                call_id = f"{base}-{n}"
            used.add(call_id)
            arguments = item.get("arguments") if isinstance(item.get("arguments"), dict) else {}
            entry: dict[str, Any] = {"id": call_id, "tool": tool, "arguments": arguments}
            if isinstance(item.get("condition"), dict):
                entry["condition"] = item["condition"]
            cleaned.append(entry)
        return cleaned
