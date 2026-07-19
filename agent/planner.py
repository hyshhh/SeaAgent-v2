"""将自然语言问题转为受控的海域监控查询规格。"""
from __future__ import annotations

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
        }

    def _fallback_intent(self, question: str) -> dict[str, Any]:
        """模型不可用时的轻量兜底，只覆盖高频句式。"""
        hull = self._extract_hull(question)
        relation = "out" if any(token in question for token in ("未在库", "不在库", "库外", "未入库")) else "in" if any(token in question for token in ("在库", "属于先验库", "属于库", "库内", "已入库")) else "any"
        registry = any(token in question for token in ("数据库", "先验库", "库中", "库里", "注册库", "库项"))
        tracks = any(token in question for token in ("视频", "监控", "画面", "视野", "轨迹", "拍到", "看到"))
        # “出现”单独出现不足以判定视频层，避免“先验库有没有出现...”被判 both
        if relation in {"in", "out"}:
            scope = "track_memory"
        elif registry and tracks:
            scope = "both"
        elif registry:
            scope = "registry"
        else:
            scope = "track_memory"

        if hull:
            kind = "hull"
            description = None
        else:
            description = self._soft_description(question)
            kind = "description" if description else "all"

        if any(token in question for token in ("多少", "数量", "几艘", "几只")):
            operation = "count"
        elif any(token in question for token in ("什么时候", "何时", "何时出现")):
            operation = "time"
        elif any(token in question for token in ("为什么", "依据", "证据", "怎么判断")):
            operation = "explain"
        elif any(token in question for token in ("有没有", "是否有", "是否出现", "有无", "存在吗")) and "哪些" not in question:
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
