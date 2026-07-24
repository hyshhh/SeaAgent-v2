# ReflectAgent · 再规划指示

## 何时 replan
仍缺关键证据，且未触达冲突/轮次上限/连续空计划。

## nextAction 写法
- 给 **PlanAgent** 可执行的下一步，例如：
  - “补 getFrames 后对描述做 matchText”
  - “对未命中舷号放宽时间再 getTrack”
  - “完成 dedupTracks 再计数”
- 避免空泛的“继续查”
- `evidenceGap` 与 `nextAction` 应对齐，便于前端展示与下一轮规划
