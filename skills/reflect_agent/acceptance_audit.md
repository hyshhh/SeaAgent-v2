# ReflectAgent · 验收进度审计

## 用法
`acceptanceProgress` 是软参考：
- `acceptanceSatisfied=true` → **强烈倾向** `sufficient`，除非 observation 明显与目标矛盾
- 仍有 `pendingRequirements` 且未到 `maxRounds` → 倾向 `replan`，`nextAction` 写清补哪一项
- 验收已满但模型仍想 replan → 可在 reason 说明“验收已满足”，优先 sufficient
- 验收未满但证据实质已够回答 → 可 `sufficient`，并在 reason 说明为何可结束

不要把验收清单当成必须 100% 勾选的硬脚本。
