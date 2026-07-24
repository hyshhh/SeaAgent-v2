# IntentAgent · 目标与意图识别规则

## 字段取值
- `targetScope`: `track_memory` | `registry` | `both`
- `targetKind`: `hull` | `description` | `all`
- `operation`: `existence` | `list` | `time` | `count` | `explain`
- `registryRelation`: `any` | `in` | `out`

## 判断要点
1. 提到先验库/库船/在库/未在库 → 涉及 `registry` 或 `registryRelation`。
2. 明确舷号（字母数字组合）→ `targetKind=hull`，填 `hullNumber`。
3. 外观/颜色/形状描述 → `targetKind=description`，填 `description`。
4. “有几艘/多少/数量” → `operation=count`。
5. “是否出现/有没有” → `operation=existence`。
6. “什么时候/出现时间” → `operation=time`。
7. 多艘目标分别描述时，填 `targetItems` 数组，不要合并为一个描述。

## 推荐工具
- `parseTargets(question)`：拆多目标（规则见 `target_parsing.yaml`）
- `extractHull(question)`：抽舷号

## 验收字段（给后续 Agent）
- `expectedOutcome`：用户真正想得到什么
- `successCriteria`：怎样算完成
- `nextAgentFocus`：PlanAgent 应优先做什么

## 策略编译（questionType 参考）
程序会根据字段编译 `questionType` / `strategy`，你只需填准上述枚举字段。
