"""将自然语言问题转为受控的海域监控查询规格。"""
from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Any, Callable

from services import AgentLLMService


class Planner:
    """模型负责理解意图，程序负责把意图编译为可审计的工具链。"""

    _SCOPES = {"track_memory", "registry", "both"}
    _OPERATIONS = {"existence", "list", "time", "count", "explain"}
    _TARGET_KINDS = {"hull", "description", "all"}
    _REGISTRY_RELATIONS = {"any", "in", "out"}

    def __init__(self, llm: AgentLLMService, allowed_tools: set[str]):
        self.llm = llm
        self.allowed_tools = allowed_tools

    def classify(self, question: str) -> dict[str, Any]:
        """构建任务规格；显式来源词、舷号和时间始终优先于模型猜测。"""
        text = question.strip()
        hull_match = self._extract_hull(text)
        explicit_scope = self._explicit_scope(text)
        target_kind = self._target_kind(text, hull_match)
        registry_relation = self._registry_relation(text)
        heuristic = {
            "questionType": "",
            "targetScope": explicit_scope or "track_memory",
            "targetKind": target_kind,
            "operation": self._operation(text),
            "registryRelation": registry_relation,
            "timeRange": self._time_range(text),
            "hullNumber": hull_match.upper() if hull_match else None,
            "description": self._description_text(text) if target_kind == "description" else None,
            "explicitScope": explicit_scope is not None,
        }
        result = self._refine_intent(text, heuristic)
        result["strategy"] = self._strategy(result)
        result["questionType"] = self._question_type(result)
        return result

    def _refine_intent(self, question: str, heuristic: dict[str, Any]) -> dict[str, Any]:
        """模型只补充歧义项；舷号、显式来源、外观描述和明确操作不被覆盖。"""
        locked_kind = heuristic.get("targetKind") in {"hull", "description"} or bool(heuristic.get("hullNumber"))
        locked_scope = bool(heuristic.get("explicitScope"))
        locked_relation = heuristic.get("registryRelation") in {"in", "out"}
        locked_operation = heuristic.get("operation") in {"count", "time", "explain"} or any(
            token in question for token in ("有没有", "是否有", "是否出现", "有无", "出现没有", "存在吗")
        )
        locked_description = bool(heuristic.get("description"))

        prompt = self.llm.prompts.get("planner_intent") if self.llm else None
        if not prompt or not self.llm:
            heuristic["intentSource"] = "heuristic"
            return self._normalize_spec(question, heuristic)

        try:
            inferred = self.llm.complete_json(prompt + "\n用户问题：" + question)
        except Exception:
            heuristic["intentSource"] = "heuristic"
            return self._normalize_spec(question, heuristic)

        if not locked_scope and inferred.get("targetScope") in self._SCOPES:
            heuristic["targetScope"] = inferred["targetScope"]
        if not locked_kind and inferred.get("targetKind") in self._TARGET_KINDS:
            heuristic["targetKind"] = inferred["targetKind"]
        if not locked_operation and inferred.get("operation") in self._OPERATIONS:
            heuristic["operation"] = inferred["operation"]
        if not locked_relation and inferred.get("registryRelation") in self._REGISTRY_RELATIONS:
            heuristic["registryRelation"] = inferred["registryRelation"]

        target_text = str(inferred.get("targetText") or "").strip()
        if heuristic.get("targetKind") == "description":
            if 1 <= len(target_text) <= 120:
                # 模型描述优先，但禁止把完整问句原样塞回
                if target_text not in question or len(target_text) <= max(8, len(question) // 2):
                    heuristic["description"] = target_text
            elif not heuristic.get("description"):
                heuristic["description"] = self._description_text(question)
        elif not locked_description:
            heuristic["description"] = None

        heuristic["intentSource"] = "model"
        return self._normalize_spec(question, heuristic)

    def _normalize_spec(self, question: str, spec: dict[str, Any]) -> dict[str, Any]:
        """最终约束：有外观描述就不能退化成轨迹列表。"""
        if spec.get("hullNumber"):
            spec["targetKind"] = "hull"
            spec["description"] = None
        else:
            visual = self._has_visual_target(question) or self._has_visual_target(str(spec.get("description") or ""))
            if visual:
                spec["targetKind"] = "description"
                if not spec.get("description"):
                    spec["description"] = self._description_text(question)
            elif spec.get("targetKind") == "description" and not spec.get("description"):
                # 没有可检索描述时，才允许退回列表
                if any(token in question for token in ("哪些", "列出", "有什么", "全部", "所有")):
                    spec["targetKind"] = "all"
                    spec["description"] = None

        if spec.get("targetKind") == "description" and not spec.get("description"):
            spec["description"] = self._description_text(question)

        # 有没有/是否 类问题保持 existence，除非已是 count/time/explain
        if spec.get("operation") not in {"count", "time", "explain"} and any(
            token in question for token in ("有没有", "是否有", "是否出现", "有无", "存在吗")
        ):
            spec["operation"] = "existence"

        # 描述存在时，禁止 all/list 直接列表化
        if spec.get("description") and spec.get("targetKind") == "all":
            spec["targetKind"] = "description"
        return spec

    @staticmethod
    def _has_visual_target(text: str) -> bool:
        if not text:
            return False
        visual_tokens = (
            "黄色", "白色", "灰色", "黑色", "蓝色", "红色", "绿色",
            "无人艇", "快艇", "货船", "巡逻艇", "渔船", "军舰", "游艇", "拖船", "客船",
            "上层建筑", "船体",
        )
        return any(token in text for token in visual_tokens)

    @staticmethod
    def _explicit_scope(question: str) -> str | None:
        relation = Planner._registry_relation(question)
        registry = any(token in question for token in ("数据库", "先验库", "库中", "库里", "注册库", "库项", "参考图"))
        tracks = any(token in question for token in ("视频", "监控", "画面", "视野", "轨迹", "镜头", "现场", "看到", "拍到"))
        # 在库/库外关系默认作用在视频轨迹层
        if relation in {"in", "out"}:
            return "track_memory"
        # 同时明确提到两层记忆，才做跨记忆对应
        if registry and tracks:
            return "both"
        if registry:
            return "registry"
        if tracks or "出现" in question:
            return "track_memory"
        return None

    @staticmethod
    def _extract_hull(question: str) -> str | None:
        explicit = re.search(r"[舷弦]号\s*[:：]?\s*([0-9A-Za-z-]+)", question, re.I)
        if explicit:
            return explicit.group(1)
        if not any(token in question for token in ("船", "出现", "轨迹", "编号", "时间", "什么时候", "何时")):
            return None
        for value in re.findall(r"(?<![\d:：-])([0-9A-Za-z]{3,8})(?![\d:：-])", question):
            if value.isdigit() and len(value) < 3:
                continue
            return value
        return None

    @staticmethod
    def _target_kind(question: str, hull_match: str | None) -> str:
        if hull_match:
            return "hull"
        if Planner._has_visual_target(question):
            return "description"
        # 只有明确要“全部/列表”且没有外观约束时，才视为 all
        if any(token in question for token in ("哪些船", "什么船", "所有船", "全部船", "列出所有", "列表")) and not any(
            token in question for token in ("黄色", "白色", "灰色", "黑色", "蓝色", "红色", "绿色", "无人艇", "快艇", "货船")
        ):
            return "all"
        # 默认按描述检索，避免“有没有出现...”被误判成列表
        if any(token in question for token in ("有没有", "是否", "出现", "找一下", "查找", "看见", "拍到")):
            return "description"
        if any(token in question for token in ("哪些", "有什么", "列出", "分别")):
            return "all"
        return "description"

    @staticmethod
    def _registry_relation(question: str) -> str:
        if any(token in question for token in ("未在库", "不在库", "库外", "未入库", "非库内")):
            return "out"
        if any(token in question for token in ("在库", "属于先验库", "属于库", "已入库", "库内", "属于注册库")):
            return "in"
        return "any"

    @staticmethod
    def _operation(question: str) -> str:
        if any(token in question for token in ("多少", "数量", "几艘", "几条")):
            return "count"
        if any(token in question for token in ("什么时候", "什么时间", "几点", "何时")):
            return "time"
        if any(token in question for token in ("为什么", "依据", "证据", "怎么判断")):
            return "explain"
        if any(token in question for token in ("哪些", "有什么", "列出", "分别")):
            return "list"
        return "existence"

    @staticmethod
    def _description_text(question: str) -> str:
        text = question
        text = re.sub(r"\d{1,2}:\d{2}(?::\d{2})?\s*(?:到|至|-|—|~)\s*\d{1,2}:\d{2}(?::\d{2})?", "", text)
        text = re.sub(r"最近[^，。？?]*?分(?:钟)?", "", text)
        text = re.sub(r"\d{1,2}\s*点(?:\s*\d{1,2}\s*分)?\s*(?:到|至|-|—|~)\s*\d{1,2}\s*点(?:\s*\d{1,2}\s*分)?", "", text)
        text = re.sub(r"从", "", text)
        text = re.sub(r"和先验库.*$|与先验库.*$", "", text)
        text = re.sub(r"对应关系", "", text)
        patterns = (
            r"^(?:请|帮我|麻烦)?(?:查找|找出|找一下|查询|看看|列出|统计|告诉我)?",
            r"(?:先验库|数据库|视频|监控|画面|当前监控记忆)(?:中|里|内|的)?",
            r"(?:有没有|是否有|有无|有哪些|有多少|多少艘|几艘|数量是多少|什么时间|什么时候|何时|为什么|怎么判断|依据|证据)",
            r"(?:出现|存在|拍到|看到|属于先验库|属于库|在库|不在库|库外|已入库|未入库)",
            r"(?:的船舶?|船只|船)?(?:吗|呢)?$",
        )
        for pattern in patterns:
            text = re.sub(pattern, "", text)
        text = re.sub(r"^(?:一个|一只|一艘|一些|几个)", "", text)
        text = re.sub(r"\s+", "", text)
        cleaned = text.strip("？?。 ，,：:的和与")
        return cleaned or question

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
        plan = {"goal": goal, "intent": intent or {}, "scope": scope, "calls": calls, "evidenceGap": evidence_gap, "stopCondition": "证据足够、证据冲突、已读取全部候选或达到最大轮次"}
        try:
            model_plan = self.llm.role("planner", {"goal": goal, "intent": intent or {}, "scope": scope, "calls": [{"id": item["id"], "tool": item["tool"]} for item in calls], "evidenceGap": evidence_gap}, on_delta)
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

