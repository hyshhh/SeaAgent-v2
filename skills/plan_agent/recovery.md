# PlanAgent · 失败修复与补洞

## 何时使用
- `planValidationError` / `previousInvalidPlan` 存在
- Reflect `evidenceGap` / `nextAction` 指出缺证据
- 上轮工具失败或空结果

## 做法
1. 先读校验错误：缺参数 → 补参数；非法工具 → 换白名单工具；坏 `$ref` → 改成已有 callId/字段。
2. 空轨迹：可放宽 limit/时间，或改查库；不要重复完全相同的失败调用。
3. 有轨迹无帧：补 `getFrames`。
4. 有帧无匹配：补 `matchText`/`matchImage`/`matchHull`。
5. 计数缺去重：补 `dedupTracks`。
6. 仍无法形成可执行计划：空 calls + `proposedState=uncertain` 与明确 `evidenceGap`（勿假装成功）。
