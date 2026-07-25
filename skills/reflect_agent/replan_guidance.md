# ReflectAgent · 再规划指示

## 何时 replan
仍缺关键证据，且未触达冲突/轮次上限/连续空计划。

## 常见缺口 → nextAction 示例
- 视频轨迹 0 条，但需对照先验库 → 「下一轮 getRegistry(hullNumber=…) 或 listRegistry+matchHull，确认是否在库」
- 有轨迹无关键帧 → 「补 getFrames($ref trackIds)」
- 有帧无描述匹配 → 「补 matchText(description, galleryImages=$ref frames.keyframes)」
- 在库/未在库未做精确舷号匹配 → 「补 matchHull(hullNumberArray)」
- 计数未去重 → 「补 dedupTracks」
- 上轮默认兜底计划过粗 → 「按 intent 重写最小 calls，去掉无关 matchText」

## nextAction 写法
- 给 **PlanAgent** 可执行的下一步，点名工具与关键参数字段
- 避免空泛的「继续查」
- `evidenceGap` 与 `nextAction` 对齐
