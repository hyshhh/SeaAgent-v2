# IntentAgent · 验收字段

## 必须填写（给 Plan / Reflect）
- `expectedOutcome`：用户真正想得到什么（一句话）
- `successCriteria`：怎样算完成（可检查的条件）
- `nextAgentFocus`：PlanAgent 本轮优先做什么

## 写法提示
| 场景 | expectedOutcome | successCriteria | nextAgentFocus |
|------|-----------------|-----------------|----------------|
| 计数 | 得到去重后数量 | 轨迹+关键帧+去重完成 | 先取全量轨迹与关键帧再去重 |
| 存在 | 判断是否出现 | 有轨迹或明确无结果 | 按条件筛轨迹并收集证据 |
| 描述 | 返回匹配候选 | 描述匹配可信 | 轨迹→关键帧→matchText |
| 舷号 | 定位该舷号 | 轨迹或库项+证据 | 按舷号筛轨迹/查库 |
| 在库关系 | 在库/未在库列表 | 完成关系分类 | matchHull + 必要时库列表 |

不要写空话；后续 Reflect 会对照这些字段判定是否 sufficient。
