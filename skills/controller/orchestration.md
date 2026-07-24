# Controller · LangGraph 四 Agent 编排

## 主路径
```
用户问题
  → IntentAgent（ReAct + 专属工具 + handoff_to_plan）
  → loop:
        PlanAgent（handoff_to_observe | handoff_to_reflect）
        → ObserveAgent（业务工具多轮 + handoff_to_reflect）
        → ReflectAgent
              handoff_to_plan_replan → 继续
              handoff_finish → 结束
  → Controller 合成 answer()
```

## 三步结构
1. **角色与工具集**：`agent/roles.py` + `agent/lc_tools.py`
2. **移交工具**：`handoff_to_plan` / `handoff_to_observe` / `handoff_to_reflect` / `handoff_finish`
3. **工作流图**：`agent/graph.py`（LangGraph StateGraph + langchain.agents.create_agent）

## Skills
- catalog 按需选用（always + 匹配 + loadSkill）
- 不整包注入全部 md
