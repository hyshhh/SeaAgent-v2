"""将自然语言问题转为受控的海域监控查询规格。"""
from __future__ import annotations

import json
import re
from datetime import datetime, timedelta
from typing import Any, Callable

from services import AgentLLMService


class Planner:
    """模型按规则表选择意图，程序只做格式校验、时间/舷号抽取和策略编译。"""

    _SCOPES = {"track_memory", "registry", "both"}
    _OPERATIONS = {"existence", "list", "time", "count", "explain"}
    _TARGET_KINDS = {"hull", "description", "all"}
    _REGISTRY_RELATIONS = {"any", "in", "out"}
    _WEAK_TARGETS = {"", "船", "船舶", "船只", "目标", "在库船", "未在库船", "库船"}

    def __init__(self, llm: AgentLLMService, allowed_tools: set[str]):
        self.llm = llm
        self.allowed_tools = allowed_tools

    def classify(self, question: str) -> dict[str, Any]:
        """优先让模型按提示词规则表判断；失败时才使用轻量兜底。"""
        text = question.strip()
        base = {
            "questionType": "",
            "targetScope": "track_memory",
            "targetKind": "all",
            "operation": "list",
            "registryRelation": "any",
            "timeRange": self._time_range(text),
            "hullNumber": self._extract_hull(text),
            "description": None,
            "selectedRules": [],
            "intentConfidence": None,
            "intentSource": "heuristic",
            "explicitScope": False,
            "expectedOutcome": None,
            "successCriteria": None,
            "nextAgentFocus": None,
        }
        model_spec = self._model_intent(text)
        if model_spec:
            base.update(model_spec)
            base["intentSource"] = "model"
        else:
            base.update(self._fallback_intent(text))
            base["intentSource"] = "heuristic"
        # 结构化字段始终由程序抽取，避免模型漏掉时间/舷号
        if not base.get("hullNumber"):
            base["hullNumber"] = self._extract_hull(text)
        if base.get("timeRange") is None:
            base["timeRange"] = self._time_range(text)
        base = self._validate_spec(text, base)
        base = self._fill_acceptance(text, base)
        base["strategy"] = self._strategy(base)
        base["questionType"] = self._question_type(base)
        return base

    def _model_intent(self, question: str) -> dict[str, Any] | None:
        if not self.llm:
            return None
        try:
            prompt = self.llm._prompt("planner_intent") if hasattr(self.llm, "_prompt") else self.llm.prompts.get("planner_intent")
        except Exception:
            prompt = self.llm.prompts.get("planner_intent") if getattr(self.llm, "prompts", None) else None
        if not prompt:
            return None
        try:
            inferred = self.llm.complete_json(prompt + "\n用户问题：" + question)
        except Exception:
            return None
        if not isinstance(inferred, dict):
            return None

        selected = inferred.get("selectedRules") or inferred.get("rules") or []
        if isinstance(selected, str):
            selected = [selected]
        selected = [str(item) for item in selected if item][:3]

        scope = inferred.get("targetScope")
        kind = inferred.get("targetKind")
        operation = inferred.get("operation")
        relation = inferred.get("registryRelation")
        target_text = str(inferred.get("targetText") or "").strip()
        hull = str(inferred.get("hullNumber") or "").strip().upper() or None
        confidence = inferred.get("confidence")
        try:
            confidence = float(confidence) if confidence is not None else None
        except (TypeError, ValueError):
            confidence = None

        # 若模型字段不合法，视为失败，走兜底
        if scope not in self._SCOPES or kind not in self._TARGET_KINDS or operation not in self._OPERATIONS:
            return None
        if relation not in self._REGISTRY_RELATIONS:
            relation = "any"
        if kind == "description" and (not target_text or target_text in self._WEAK_TARGETS):
            # 模型说 description 却没给出有效外观，降为 all
            kind = "all"
            target_text = ""
        if kind != "description":
            target_text = ""
        if kind == "hull" and not hull:
            hull = self._extract_hull(question)
        if kind != "hull":
            # 非舷号问题不强制保留模型给的 hull，除非问题本身含舷号
            hull = hull if self._extract_hull(question) else None

        expected_outcome = str(inferred.get("expectedOutcome") or inferred.get("expected_outcome") or "").strip() or None
        success_criteria = str(inferred.get("successCriteria") or inferred.get("success_criteria") or "").strip() or None
        next_focus = str(inferred.get("nextAgentFocus") or inferred.get("next_agent_focus") or "").strip() or None

        return {
            "targetScope": scope,
            "targetKind": kind,
            "operation": operation,
            "registryRelation": relation,
            "description": target_text or None,
            "hullNumber": hull,
            "selectedRules": selected,
            "intentConfidence": confidence,
            "explicitScope": True,
            "expectedOutcome": expected_outcome,
            "successCriteria": success_criteria,
            "nextAgentFocus": next_focus,
        }

    def _fallback_intent(self, question: str) -> dict[str, Any]:
        """模型不可用时的规则兜底，只覆盖高频句式。"""
        hull = self._extract_hull(question)
        out_tokens = ("未在库", "不在库", "库外", "未入库", "非在库")
        in_tokens = ("在库", "属于数据库", "属于库", "库内", "登记在库", "在数据库", "入库船")
        relation = "out" if any(token in question for token in out_tokens) else "in" if any(token in question for token in in_tokens) else "any"

        registry_only = any(token in question for token in ("数据库里", "库里", "先验库", "注册库", "船库中", "库中有没有", "库中有哪些"))
        registry = registry_only or any(token in question for token in ("数据库", "先验库", "库中", "库里", "注册库"))
        tracks = any(token in question for token in ("视频", "监控", "画面", "视野", "轨迹", "镜头", "录像", "出现"))

        # 视频侧在库/未在库：永远优先 track_memory，避免误成 both/registry
        if relation in {"in", "out"}:
            scope = "track_memory"
        elif tracks and registry and not registry_only:
            # “视频+数据库”但无明确在库关系时，若像对应核验才 both，否则视频
            if any(token in question for token in ("对应", "是否一致", "能不能匹配", "库中的", "库里的")):
                scope = "both"
            else:
                scope = "track_memory"
                if "数据库" in question or "先验库" in question:
                    relation = "in"
        elif registry_only or (registry and not tracks):
            scope = "registry"
        else:
            scope = "track_memory"

        if hull:
            kind = "hull"
            description = None
        else:
            description = self._soft_description(question)
            kind = "description" if description else "all"

        if any(token in question for token in ("多少", "几艘", "数量", "几只")):
            operation = "count"
        elif any(token in question for token in ("什么时候", "何时", "出现时间")):
            operation = "time"
        elif any(token in question for token in ("为什么", "依据", "证据", "怎么判断")):
            operation = "explain"
        elif any(token in question for token in ("有没有", "是否有", "是否出现", "出现过", "存在吗")) and "哪些" not in question:
            operation = "existence"
        else:
            operation = "list"

        return {
            "targetScope": scope,
            "targetKind": kind,
            "operation": operation,
            "registryRelation": relation,
            "description": description,
            "hullNumber": hull,
            "selectedRules": ["FALLBACK"],
            "intentConfidence": 0.0,
            "explicitScope": registry or tracks or relation != "any",
        }


    def _validate_spec(self, question: str, spec: dict[str, Any]) -> dict[str, Any]:
        """只做合法性约束，不重新发明用户意图。"""
        if spec.get("targetScope") not in self._SCOPES:
            spec["targetScope"] = "track_memory"
        if spec.get("targetKind") not in self._TARGET_KINDS:
            spec["targetKind"] = "all"
        if spec.get("operation") not in self._OPERATIONS:
            spec["operation"] = "list"
        if spec.get("registryRelation") not in self._REGISTRY_RELATIONS:
            spec["registryRelation"] = "any"

        description = str(spec.get("description") or "").strip()
        if description in self._WEAK_TARGETS:
            description = ""
        # 去掉明显问法残留
        description = re.sub(r"^(有没有|是否|哪些|查出|查找)", "", description).strip()
        if description in self._WEAK_TARGETS:
            description = ""

        if spec.get("hullNumber"):
            spec["targetKind"] = "hull"
            description = ""
        elif spec.get("targetKind") == "description" and not description:
            # description 无内容时降级，避免 matchText("船")
            spec["targetKind"] = "all"
        elif description and spec.get("targetKind") == "all":
            # 模型给了有效外观文本，则升级为 description
            spec["targetKind"] = "description"

        # 在库/未在库且无外观时，保持 all，避免 relation_description
        if spec.get("registryRelation") in {"in", "out"} and not description and spec.get("targetKind") != "hull":
            spec["targetKind"] = "all"

        # “视频中哪些船属于先验库”不应变成 both
        if spec.get("registryRelation") in {"in", "out"} and spec.get("targetScope") == "both" and not description:
            spec["targetScope"] = "track_memory"

        spec["description"] = description or None
        if spec.get("targetKind") != "description":
            spec["description"] = None
        if spec.get("targetKind") != "hull":
            # 保留问题中真实舷号，不保留模型臆造
            extracted = self._extract_hull(question)
            spec["hullNumber"] = extracted
        return spec

    @staticmethod
    def _soft_description(question: str) -> str | None:
        """兜底时的弱外观抽取：只有明显颜色/船型才返回。"""
        colors = ("黄色", "白色", "灰色", "黑色", "蓝色", "红色", "绿色", "橙色")
        types = ("无人艇", "快艇", "货船", "巡逻艇", "渔船", "军舰", "游艇", "拖船", "客船")
        hits = [token for token in colors + types if token in question]
        if not hits:
            return None
        # 保序去重后拼成简短目标
        ordered = []
        for token in colors + types:
            if token in hits and token not in ordered:
                ordered.append(token)
        return "".join(ordered)

    @staticmethod
    def _extract_hull(question: str) -> str | None:
        explicit = re.search(r"[舷弦]号\s*[:：]?\s*([0-9A-Za-z-]+)", question, re.I)
        if explicit:
            return explicit.group(1).upper()
        # 裸舷号：要求问题像在问该编号
        if not any(token in question for token in ("船", "出现", "轨迹", "编号", "时间", "什么时候", "何时", "为什么", "有没有", "是否", "库")):
            return None
        for value in re.findall(r"(?<![\d:：-])([0-9A-Za-z]{3,8})(?![\d:：-])", question):
            if value.isdigit() and len(value) < 3:
                continue
            # 排除纯时间片段
            if re.fullmatch(r"\d{1,2}", value):
                continue
            return value.upper()
        return None

    @staticmethod
    def _strategy(spec: dict[str, Any]) -> str:
        scope = spec["targetScope"]
        kind = spec["targetKind"]
        operation = spec["operation"]
        relation = spec["registryRelation"]
        if operation == "count":
            if scope == "registry":
                return "registry_description_count" if kind == "description" else "registry_count"
            return "track_description_count" if kind == "description" else "track_count"
        if kind == "hull":
            return "registry_hull" if scope == "registry" else "hull_lookup"
        if relation in {"in", "out"} and scope != "registry":
            return "track_relation_description" if kind == "description" else "registry_relation"
        if scope == "registry":
            if kind == "all":
                return "registry_list"
            return "registry_description"
        if scope == "both":
            return "cross_reference"
        if kind == "all":
            return "track_list"
        return "track_description"

    @staticmethod
    def _question_type(spec: dict[str, Any]) -> str:
        return {
            "hull_lookup": "hull",
            "registry_hull": "registry_hull",
            "track_description": "description",
            "registry_description": "registry_description",
            "cross_reference": "cross_reference",
            "registry_relation": "out_of_registry" if spec["registryRelation"] == "out" else "in_registry",
            "track_count": "count",
            "registry_count": "registry_count",
            "registry_description_count": "registry_description_count",
            "track_description_count": "description_count",
            "track_list": "track_list",
            "registry_list": "registry_list",
            "track_relation_description": "relation_description",
        }[spec["strategy"]]

    def _fill_acceptance(self, question: str, spec: dict[str, Any]) -> dict[str, Any]:
        """为后续 Plan/Observe/Reflect 补齐验收标准；模型未给时按策略生成简版。"""
        if not spec.get("expectedOutcome") or not spec.get("successCriteria") or not spec.get("nextAgentFocus"):
            scope = spec.get("targetScope")
            kind = spec.get("targetKind")
            operation = spec.get("operation")
            relation = spec.get("registryRelation")
            hull = spec.get("hullNumber") or ""
            desc = spec.get("description") or ""
            if kind == "hull" and hull:
                target = f"舷号 {hull}"
            elif kind == "description" and desc:
                target = f"描述“{desc}”"
            elif relation == "in":
                target = "在库船舶"
            elif relation == "out":
                target = "未在库船舶"
            else:
                target = "相关船舶"

            if not spec.get("expectedOutcome"):
                if operation == "count":
                    spec["expectedOutcome"] = f"统计范围内{target}的数量（必要时去重）"
                elif operation == "time":
                    spec["expectedOutcome"] = f"给出{target}的出现时间范围"
                elif operation == "explain":
                    spec["expectedOutcome"] = f"解释判定{target}的关键证据"
                elif operation == "existence":
                    spec["expectedOutcome"] = f"确认{target}是否出现/存在"
                else:
                    prefix = "列出先验库中" if scope == "registry" else "列出视频中"
                    spec["expectedOutcome"] = f"{prefix}{target}"

            if not spec.get("successCriteria"):
                if scope == "registry":
                    spec["successCriteria"] = "已读取先验库目标条目或明确库中无匹配"
                elif kind == "description":
                    spec["successCriteria"] = "已获得描述匹配结果（含分数/灰区核验）或明确无匹配"
                elif relation in {"in", "out"}:
                    spec["successCriteria"] = "已完成在库/未在库判定并得到轨迹列表"
                elif operation == "count":
                    spec["successCriteria"] = "已得到可审计的去重数量"
                else:
                    spec["successCriteria"] = "已获得直接轨迹命中或匹配证据，可回答用户问题"

            if not spec.get("nextAgentFocus"):
                if scope == "registry":
                    spec["nextAgentFocus"] = "先查先验库条目与参考图"
                elif kind == "hull":
                    spec["nextAgentFocus"] = "先查轨迹舷号与库项，不足时用库图匹配关键帧"
                elif kind == "description":
                    spec["nextAgentFocus"] = "先取正式关键帧做文本匹配，灰区再核验"
                elif relation in {"in", "out"}:
                    spec["nextAgentFocus"] = "先按时段取轨迹，再精确匹配舷号并做库图匹配"
                elif operation == "count":
                    spec["nextAgentFocus"] = "先取全量轨迹与关键帧，再跨轨迹去重"
                else:
                    spec["nextAgentFocus"] = "先按条件筛选轨迹并收集关键帧证据"
        return spec


    def decide_tools(
        self,
        question: str,
        intent: dict[str, Any],
        history: list[dict[str, Any]] | None = None,
        memory_scope: dict[str, Any] | None = None,
        on_delta: Callable[[str], None] | None = None,
    ) -> dict[str, Any]:
        """完全自主模式：由模型输出本轮工具调用计划，再做白名单与参数校验。"""
        history = history or []
        memory_scope = memory_scope or {}
        payload = {
            "question": question,
            "intent": {
                "questionType": intent.get("questionType"),
                "strategy": intent.get("strategy"),
                "targetScope": intent.get("targetScope"),
                "targetKind": intent.get("targetKind"),
                "operation": intent.get("operation"),
                "registryRelation": intent.get("registryRelation"),
                "hullNumber": intent.get("hullNumber"),
                "description": intent.get("description"),
                "timeRange": list(intent["timeRange"]) if intent.get("timeRange") else None,
                "selectedRules": intent.get("selectedRules") or [],
                "expectedOutcome": intent.get("expectedOutcome"),
                "successCriteria": intent.get("successCriteria"),
                "nextAgentFocus": intent.get("nextAgentFocus"),
            },
            "round": len(history) + 1,
            "maxRounds": intent.get("maxRounds"),
            "previousRounds": history,
            "availableResultKeys": sorted(memory_scope.keys()),
            "allowedTools": sorted(self.allowed_tools),
        }
        model_plan: dict[str, Any] = {}
        raw = ""
        try:
            prompt = self.llm._prompt("planner_autonomous")
            raw = self.llm.complete_text_stream(prompt + "\n输入：" + json.dumps(payload, ensure_ascii=False), on_delta) if on_delta else self.llm.complete_text(prompt + "\n输入：" + json.dumps(payload, ensure_ascii=False))
            model_plan = self._extract_json_object(raw)
        except Exception as error:
            model_plan = self._fallback_autonomous_plan(intent)
            model_plan["modelFallback"] = str(error)
            if on_delta and model_plan.get("modelFallback"):
                on_delta(f"\n[兜底规划] {model_plan['goal']}")

        calls = self._sanitize_calls(model_plan.get("calls") or [])
        plan_repair: str | None = None
        executable, issue = self._calls_are_executable(calls, memory_scope)
        if not calls or not executable:
            fallback = self._fallback_autonomous_plan(intent)
            calls = self._sanitize_calls(fallback.get("calls") or [])
            plan_repair = issue or "自主计划未生成有效工具调用，已使用受控兜底计划"
            # 修复后的计划必须与实际执行工具一致，避免前端展示与执行脱节。
            for key in ("goal", "proposedState", "reason", "evidenceGap", "answerHint"):
                if key in fallback:
                    model_plan[key] = fallback[key]
        state = str(model_plan.get("proposedState") or "replan").lower()
        if state not in {"sufficient", "replan", "conflict", "uncertain"}:
            state = "replan"
        goal = str(model_plan.get("goal") or "根据意图检索轨迹记忆与先验库证据").strip()
        reason = str(model_plan.get("reason") or "按自主规划执行工具并收集证据").strip()
        evidence_gap = model_plan.get("evidenceGap")
        if evidence_gap is not None:
            evidence_gap = str(evidence_gap).strip() or None
        answer_hint = str(model_plan.get("answerHint") or "").strip()
        plan = {
            "goal": goal,
            "intent": intent or {},
            "scope": intent.get("timeRange"),
            "calls": calls,
            "evidenceGap": evidence_gap,
            "proposedState": state,
            "reason": reason,
            "answerHint": answer_hint,
            "stopCondition": "证据足够、证据冲突、已读完全部候选或达到轮次上限",
            "modelPlan": {
                "summary": (f"计划已修正：{plan_repair}。{goal}" if plan_repair else (raw.strip()[:500] if raw else goal)),
                "goal": goal,
                "proposedState": state,
                "reason": reason,
                "answerHint": answer_hint,
            },
            "planMode": "autonomous",
        }
        if plan_repair:
            plan["planRepair"] = plan_repair
        if model_plan.get("modelFallback"):
            plan["modelFallback"] = model_plan["modelFallback"]
        return plan

    def _sanitize_calls(self, calls: Any) -> list[dict[str, Any]]:
        if not isinstance(calls, list):
            return []
        cleaned: list[dict[str, Any]] = []
        used_ids: set[str] = set()
        for index, item in enumerate(calls[:6]):
            if not isinstance(item, dict):
                continue
            tool = str(item.get("tool") or "").strip()
            if tool not in self.allowed_tools:
                continue
            call_id = str(item.get("id") or f"call{index + 1}").strip() or f"call{index + 1}"
            call_id = re.sub(r"[^0-9A-Za-z_\-]", "", call_id) or f"call{index + 1}"
            if call_id in used_ids:
                call_id = f"{call_id}_{index + 1}"
            used_ids.add(call_id)
            arguments = item.get("arguments") if isinstance(item.get("arguments"), dict) else {}
            arguments = self._sanitize_arguments(tool, arguments)
            call: dict[str, Any] = {"id": call_id, "tool": tool, "arguments": arguments}
            condition = item.get("condition")
            if isinstance(condition, dict) and condition.get("ref"):
                call["condition"] = condition
            cleaned.append(call)
        return cleaned

    def _calls_are_executable(self, calls: list[dict[str, Any]], memory_scope: dict[str, Any]) -> tuple[bool, str | None]:
        """校验自主计划的工具依赖，阻止空候选直接进入匹配工具。"""
        if not calls:
            return False, "自主计划没有合法工具调用"
        available = set(memory_scope)
        for call in calls:
            tool = call["tool"]
            arguments = call.get("arguments") or {}
            requirements: tuple[tuple[str, str], ...] = ()
            if tool == "getFrames":
                requirements = (("trackIds", "轨迹编号"),)
            elif tool == "getClip":
                requirements = (("trackId", "轨迹编号"),)
            elif tool == "getRegistry":
                requirements = (("hullNumber", "舷号"),)
            elif tool == "matchHull":
                requirements = (("hullNumberArray", "舷号数组"),)
            elif tool == "matchText":
                requirements = (("description", "描述"), ("galleryImages", "候选图像"))
            elif tool == "matchImage":
                requirements = (("queryImages", "查询图像"), ("galleryImages", "候选图像"))
            elif tool == "dedupTracks":
                requirements = (("tracks", "轨迹列表"), ("keyframesByTrack", "关键帧集合"))
            elif tool == "verifyTarget":
                sources = ("description", "registryReferenceIds", "keyframeIds", "shipSegmentIds")
                if not any(self._has_call_value(arguments.get(name), available, memory_scope) for name in sources):
                    return False, "verifyTarget 缺少可核验的文字、参考图、关键帧或片段"

            for field, label in requirements:
                if not self._has_call_value(arguments.get(field), available, memory_scope):
                    return False, f"{tool} 缺少有效{label}"
            available.add(call["id"])
        return True, None

    @staticmethod
    def _has_call_value(value: Any, available: set[str], memory_scope: dict[str, Any]) -> bool:
        if isinstance(value, dict) and "$ref" in value:
            reference = str(value.get("$ref") or "").strip()
            root = reference.split(".", 1)[0]
            return bool(root) and (root in available or root in memory_scope)
        if isinstance(value, (list, tuple, set, dict)):
            return bool(value)
        return value not in (None, "")

    def _sanitize_arguments(self, tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
        allowed_fields = {
            "getTrack": {"timeRange", "hullNumber", "finalMatchType", "offset", "limit"},
            "getFrames": {"trackIds"},
            "getClip": {"trackId", "timeRange"},
            "getRegistry": {"hullNumber"},
            "listRegistry": set(),
            "matchHull": {"hullNumberArray"},
            "matchText": {"description", "galleryImages", "topK"},
            "matchImage": {"queryImages", "galleryImages", "topK"},
            "verifyTarget": {"description", "registryReferenceIds", "keyframeIds", "shipSegmentIds"},
            "showEvidence": {"keyframeIds", "shipSegmentIds", "registryReferenceIds"},
            "dedupTracks": {"tracks", "keyframesByTrack"},
        }.get(tool, set())
        cleaned: dict[str, Any] = {}
        for key, value in arguments.items():
            if key not in allowed_fields:
                continue
            cleaned[key] = self._sanitize_value(value)
        # 时间范围统一
        if "timeRange" in cleaned:
            cleaned["timeRange"] = self._normalize_time_range(cleaned.get("timeRange"))
        if tool == "getTrack":
            if "offset" in cleaned:
                try:
                    cleaned["offset"] = max(0, int(cleaned["offset"]))
                except (TypeError, ValueError):
                    cleaned["offset"] = 0
            if "limit" in cleaned:
                try:
                    cleaned["limit"] = max(0, min(200, int(cleaned["limit"])))
                except (TypeError, ValueError):
                    cleaned["limit"] = 60
            if "hullNumber" in cleaned and cleaned["hullNumber"] is not None:
                cleaned["hullNumber"] = str(cleaned["hullNumber"]).strip().upper() or None
        if tool == "getRegistry" and "hullNumber" in cleaned:
            cleaned["hullNumber"] = str(cleaned.get("hullNumber") or "").strip().upper()
        if tool == "getClip" and "trackId" in cleaned and not isinstance(cleaned["trackId"], dict):
            cleaned["trackId"] = str(cleaned["trackId"])
        if tool == "getFrames" and "trackIds" in cleaned and not isinstance(cleaned["trackIds"], dict):
            values = cleaned["trackIds"] if isinstance(cleaned["trackIds"], list) else [cleaned["trackIds"]]
            cleaned["trackIds"] = [str(item) for item in values if item not in (None, "")]
        if tool == "matchHull" and "hullNumberArray" in cleaned and not isinstance(cleaned["hullNumberArray"], dict):
            values = cleaned["hullNumberArray"] if isinstance(cleaned["hullNumberArray"], list) else [cleaned["hullNumberArray"]]
            cleaned["hullNumberArray"] = [str(item).strip().upper() for item in values if item not in (None, "")]
        if tool in {"matchText", "matchImage"} and "topK" in cleaned and cleaned["topK"] is not None and not isinstance(cleaned["topK"], dict):
            try:
                cleaned["topK"] = max(1, min(50, int(cleaned["topK"])))
            except (TypeError, ValueError):
                cleaned.pop("topK", None)
        return cleaned

    def _sanitize_value(self, value: Any) -> Any:
        if isinstance(value, dict):
            if "$ref" in value:
                ref = str(value.get("$ref") or "").strip()
                if not ref:
                    return None
                result: dict[str, Any] = {"$ref": ref}
                if value.get("$map"):
                    result["$map"] = str(value["$map"])
                if value.get("$list"):
                    result["$list"] = True
                if value.get("$compact"):
                    result["$compact"] = True
                if "$default" in value:
                    result["$default"] = value["$default"]
                return result
            return {str(key): self._sanitize_value(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._sanitize_value(item) for item in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)

    @staticmethod
    def _normalize_time_range(value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, dict) and "$ref" in value:
            return value
        if isinstance(value, (list, tuple)) and len(value) == 2:
            try:
                return [float(value[0]), float(value[1])]
            except (TypeError, ValueError):
                return None
        return None

    def _fallback_autonomous_plan(self, intent: dict[str, Any]) -> dict[str, Any]:
        time_range = list(intent["timeRange"]) if intent.get("timeRange") else None
        hull = intent.get("hullNumber")
        description = intent.get("description")
        target_scope = intent.get("targetScope") or "track_memory"
        target_kind = intent.get("targetKind") or "all"
        operation = intent.get("operation") or "list"
        page = int(intent.get("retrievalPageSize") or 60)
        if target_scope == "registry" and target_kind == "hull" and hull:
            return {
                "goal": f"查询先验库舷号 {hull}",
                "calls": [{"id": "registry", "tool": "getRegistry", "arguments": {"hullNumber": hull}}],
                "proposedState": "sufficient",
                "reason": "库项查询可直接返回库记录",
                "evidenceGap": None,
                "answerHint": f"返回舷号 {hull} 的库项信息",
            }
        if target_scope == "registry" and operation == "list" and target_kind == "all":
            return {
                "goal": "列出先验库全部条目",
                "calls": [{"id": "registryList", "tool": "listRegistry", "arguments": {}}],
                "proposedState": "sufficient",
                "reason": "直接读取先验库目录",
                "evidenceGap": None,
                "answerHint": "列出全部库项",
            }
        if target_kind == "hull" and hull:
            return {
                "goal": f"检索视频中舷号 {hull} 的轨迹与库项",
                "calls": [
                    {"id": "tracks", "tool": "getTrack", "arguments": {"hullNumber": hull, "timeRange": time_range, "offset": 0, "limit": page}},
                    {"id": "registry", "tool": "getRegistry", "arguments": {"hullNumber": hull}},
                ],
                "proposedState": "replan",
                "reason": "先查直接轨迹与库项，再决定是否向量匹配",
                "evidenceGap": "若无直接轨迹且库项可检索，需对关键帧做图像匹配",
                "answerHint": "",
            }
        if target_kind == "description" and description:
            return {
                "goal": f"按描述检索候选轨迹：{description}",
                "calls": [
                    {"id": "tracks", "tool": "getTrack", "arguments": {"timeRange": time_range, "offset": 0, "limit": page}},
                    {"id": "frames", "tool": "getFrames", "arguments": {"trackIds": {"$ref": "tracks.trackIds"}}},
                    {"id": "matches", "tool": "matchText", "arguments": {"description": description, "galleryImages": {"$ref": "frames.keyframes"}, "topK": 3}},
                ],
                "proposedState": "replan",
                "reason": "描述问题需要先取关键帧再文本匹配",
                "evidenceGap": "根据匹配分数决定是否灰区核验或继续分页",
                "answerHint": "",
            }
        if operation == "count":
            return {
                "goal": "统计时间范围内船舶数量并去重",
                "calls": [
                    {"id": "tracks", "tool": "getTrack", "arguments": {"timeRange": time_range, "offset": 0, "limit": 0}},
                    {"id": "frames", "tool": "getFrames", "arguments": {"trackIds": {"$ref": "tracks.trackIds"}}},
                    {"id": "dedup", "tool": "dedupTracks", "arguments": {"tracks": {"$ref": "tracks.tracks"}, "keyframesByTrack": {"$ref": "frames.keyframesByTrack"}}},
                ],
                "proposedState": "sufficient",
                "reason": "数量统计依赖轨迹集合与跨轨迹去重",
                "evidenceGap": None,
                "answerHint": "按去重结果统计数量",
            }
        return {
            "goal": "读取时间范围内轨迹列表",
            "calls": [{"id": "tracks", "tool": "getTrack", "arguments": {"timeRange": time_range, "offset": 0, "limit": page}}],
            "proposedState": "sufficient",
            "reason": "默认先读取轨迹列表作为证据",
            "evidenceGap": None,
            "answerHint": "返回轨迹列表",
        }

    @staticmethod
    def _extract_json_object(text: str) -> dict[str, Any]:
        content = (text or "").strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        try:
            value = json.loads(content)
            if isinstance(value, dict):
                return value
        except Exception:
            pass
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if not match:
            raise ValueError("模型未返回可解析 JSON 计划")
        value = json.loads(match.group())
        if not isinstance(value, dict):
            raise ValueError("计划 JSON 必须是对象")
        return value


    def build(self, goal: str, calls: list[dict[str, Any]], scope: Any = None, evidence_gap: str | None = None, on_delta: Callable[[str], None] | None = None, intent: dict[str, Any] | None = None) -> dict[str, Any]:
        invalid = [call["tool"] for call in calls if call["tool"] not in self.allowed_tools]
        if invalid:
            raise ValueError(f"工具不在白名单：{invalid}")
        plan = {
            "goal": goal,
            "intent": intent or {},
            "scope": scope,
            "calls": calls,
            "evidenceGap": evidence_gap,
            "planMode": "guided",
            "stopCondition": "证据足够、证据冲突、已读取全部候选或达到最大轮次",
        }
        try:
            model_plan = self.llm.role(
                "planner",
                {
                    "goal": goal,
                    "intent": intent or {},
                    "scope": scope,
                    "calls": [{"id": item["id"], "tool": item["tool"]} for item in calls],
                    "evidenceGap": evidence_gap,
                },
                on_delta,
            )
            plan["modelPlan"] = model_plan
        except Exception as error:
            plan["modelFallback"] = str(error)
        return plan

    @staticmethod
    def _time_range(question: str) -> tuple[float, float] | None:
        cn_digits = {"零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}

        def parse_amount(token: str) -> float | None:
            token = token.strip()
            if not token:
                return None
            if re.fullmatch(r"\d+(?:\.\d+)?", token):
                return float(token)
            if token == "半":
                return 0.5
            if token == "十":
                return 10.0
            if token.startswith("十") and len(token) == 2 and token[1] in cn_digits:
                return 10.0 + cn_digits[token[1]]
            if token.endswith("十") and len(token) == 2 and token[0] in cn_digits:
                return cn_digits[token[0]] * 10.0
            if len(token) == 1 and token in cn_digits:
                return float(cn_digits[token])
            return None

        relative_match = re.search(r"最近\s*([0-9一二两三四五六七八九十半]+(?:\.\d+)?)\s*分(?:钟)?", question)
        if relative_match:
            amount = parse_amount(relative_match.group(1))
            if amount is not None:
                end = datetime.now().astimezone().timestamp()
                return end - amount * 60, end
        clock_match = re.search(
            r"(\d{1,2}):(\d{2})(?::(\d{2}))?\s*(?:到|至|-|—|~)\s*(\d{1,2}):(\d{2})(?::(\d{2}))?",
            question,
        )
        if clock_match:
            now = datetime.now().astimezone()
            start = now.replace(hour=int(clock_match.group(1)), minute=int(clock_match.group(2)), second=int(clock_match.group(3) or 0), microsecond=0)
            end = now.replace(hour=int(clock_match.group(4)), minute=int(clock_match.group(5)), second=int(clock_match.group(6) or 0), microsecond=0)
            if end < start:
                end += timedelta(days=1)
            return start.timestamp(), end.timestamp()
        hour_match = re.search(
            r"(\d{1,2})\s*点(?:\s*(\d{1,2})\s*分)?\s*(?:到|至|-|—|~)\s*(\d{1,2})\s*点(?:\s*(\d{1,2})\s*分)?",
            question,
        )
        if hour_match:
            now = datetime.now().astimezone()
            start = now.replace(hour=int(hour_match.group(1)), minute=int(hour_match.group(2) or 0), second=0, microsecond=0)
            end = now.replace(hour=int(hour_match.group(3)), minute=int(hour_match.group(4) or 0), second=0, microsecond=0)
            if end < start:
                end += timedelta(days=1)
            return start.timestamp(), end.timestamp()
        return None
