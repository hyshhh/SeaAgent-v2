function useQuestion(text) {
  const input = document.getElementById('agentQuestion');
  input.value = text;
  input.focus();
}

function valueText(value) {
  if (value === null || value === undefined || value === '') return '—';
  if (Array.isArray(value)) return value.join('、');
  if (typeof value === 'object') return JSON.stringify(value);
  return String(value);
}

function stateLabel(state) {
  return ({sufficient: '证据充分', replan: '继续规划', conflict: '证据冲突', uncertain: '无法确认'})[state] || valueText(state);
}

function questionTypeLabel(type) {
  return ({hull: '舷号查询', description: '描述目标查询', out_of_registry: '未在库船查询', in_registry: '在库船查询', count: '数量统计'})[type] || valueText(type);
}

function renderTracks(tracks) {
  if (!tracks?.length) return '';
  const rows = tracks.slice(0, 3).map((track) => {
    const hull = track.finalHullNumber || track.hullNumber || '无稳定舷号';
    const start = Number(track.startTime ?? track.start_time);
    const end = Number(track.endTime ?? track.end_time);
    const time = Number.isFinite(start) && Number.isFinite(end) ? `${start.toFixed(1)}秒—${end.toFixed(1)}秒` : '时间未知';
    const score = Number(track.embeddingScore);
    return `<div class="track-summary"><strong>${escapeHtml(track.trackId || '未知轨迹')}</strong><span>${escapeHtml(hull)} · ${time}${Number.isFinite(score) ? ` · 相似度 ${score.toFixed(3)}` : ''}</span></div>`;
  }).join('');
  return `<div class="answer-tracks">${rows}</div>`;
}

function renderAgentAnswer(result) {
  const scope = Array.isArray(result.queryScope) ? `${result.queryScope[0]}秒—${result.queryScope[1]}秒` : '全视频';
  const chain = (result.toolChain || []).map((item) => `<span class="tool-tag">${escapeHtml(item)}</span>`).join('');
  document.getElementById('agentAnswer').className = 'agent-answer';
  document.getElementById('agentAnswer').innerHTML = `
    <div class="answer-head"><strong>${escapeHtml(result.conclusion || '问答完成')}</strong><span class="status-tag ${result.uncertainty === 'sufficient' ? 'ok' : 'off'}">${escapeHtml(stateLabel(result.uncertainty))}</span></div>
    <p>${escapeHtml(result.answerText || '未生成回答')}</p>
    <div class="answer-meta"><span>问题类型：${escapeHtml(questionTypeLabel(result.questionType))}</span><span>查询范围：${escapeHtml(scope)}</span></div>
    ${renderTracks(result.tracks)}
    ${chain ? `<div class="tool-tags">${chain}</div>` : ''}`;
}

function callText(call) {
  const argumentsText = call.arguments && Object.keys(call.arguments).length ? ` ${JSON.stringify(call.arguments)}` : '';
  return `${call.tool || '未知工具'}(${call.id || '调用'})${argumentsText}`;
}

function renderAgentRounds(rounds) {
  const container = document.getElementById('agentRounds');
  if (!rounds?.length) {
    container.innerHTML = '<div class="empty-msg">未产生工具调用轮次</div>';
    return;
  }
  container.innerHTML = rounds.map((round, index) => {
    const calls = round.plan?.calls || [];
    const observations = round.observation?.calls || [];
    const callTags = calls.length ? calls.map((call) => `<span class="tool-tag">${escapeHtml(callText(call))}</span>`).join('') : '<span>无工具调用</span>';
    const observed = observations.length ? observations.map((item) => `${item.tool || item.id}：${item.skipped ? '跳过' : item.ok === false ? '失败' : '完成'}`).join('；') : '无观察结果';
    return `<article class="round-card">
      <div class="round-title"><strong>第 ${index + 1} 轮</strong><span>${escapeHtml(stateLabel(round.reflection?.state))}</span></div>
      <div class="agent-step"><strong>规划智能体</strong><div><div>${escapeHtml(round.plan?.goal || '执行当前证据计划')}</div><div class="tool-tags">${callTags}</div></div></div>
      <div class="agent-step"><strong>观察智能体</strong><div>${escapeHtml(observed)}</div></div>
      <div class="agent-step"><strong>反思智能体</strong><div>${escapeHtml(round.reflection?.reason || '检查证据充分性')}</div></div>
    </article>`;
  }).join('');
}

function evidenceItem(type, id) {
  const prefix = type === 'video' ? '目标船片段' : type === 'keyframe' ? '正式关键帧' : '先验库参考图';
  const route = type === 'video' ? 'clips' : type === 'keyframe' ? 'keyframes' : 'registry';
  return {type, id, label: `${prefix} ${id}`, url: `/api/evidence/${route}/${encodeURIComponent(id)}`};
}

function evidenceCard(item) {
  const media = item.type === 'video'
    ? `<video controls preload="metadata" src="${item.url}"></video>`
    : `<img loading="lazy" src="${item.url}" alt="${escapeHtml(item.label)}">`;
  return `<article class="evidence-card">${media}<span title="${escapeHtml(item.label)}">${escapeHtml(item.label)}</span></article>`;
}

function renderEvidence(evidence, displayGroups) {
  const container = document.getElementById('evidenceGallery');
  const groups = (displayGroups || []).slice(0, 3).map((group) => {
    const items = [];
    if (group.shipSegmentIds?.[0]) items.push(evidenceItem('video', group.shipSegmentIds[0]));
    if (group.keyframeIds?.[0]) items.push(evidenceItem('keyframe', group.keyframeIds[0]));
    if (group.registryReferenceIds?.[0]) items.push(evidenceItem('registry', group.registryReferenceIds[0]));
    return {trackId: group.trackId, items};
  }).filter((group) => group.items.length);
  if (groups.length) {
    container.className = 'evidence-grid evidence-groups';
    container.innerHTML = groups.map((group) => `<section class="evidence-group"><strong>轨迹 ${escapeHtml(group.trackId)}</strong><div class="evidence-group-media">${group.items.map(evidenceCard).join('')}</div></section>`).join('');
    return;
  }
  const pools = [
    (evidence?.shipSegmentIds || []).map((id) => evidenceItem('video', id)),
    (evidence?.keyframeIds || []).map((id) => evidenceItem('keyframe', id)),
    (evidence?.registryReferenceIds || []).map((id) => evidenceItem('registry', id)),
  ];
  const selected = pools.flatMap((items) => items.slice(0, 1));
  container.className = 'evidence-grid';
  container.innerHTML = selected.length ? selected.map(evidenceCard).join('') : '<div class="empty-msg">本次回答没有可展示的视觉证据</div>';
}

async function askAgent() {
  const question = document.getElementById('agentQuestion').value.trim();
  const button = document.getElementById('btnAskAgent');
  if (!question) return showToast('请输入问题', 'error');
  try {
    button.disabled = true;
    button.textContent = '推理中…';
    document.getElementById('agentAnswer').className = 'agent-answer empty-state';
    document.getElementById('agentAnswer').textContent = '正在执行规划、观察与反思…';
    document.getElementById('agentRounds').innerHTML = '<div class="empty-msg">正在生成工具调用链…</div>';
    document.getElementById('evidenceGallery').innerHTML = '<div class="empty-msg">正在收集证据…</div>';
    const result = await apiFetch('/api/agent/query', {
      method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({question})
    });
    renderAgentAnswer(result);
    renderAgentRounds(result.rounds);
    renderEvidence(result.evidence, result.displayGroups);
  } catch (error) {
    document.getElementById('agentAnswer').className = 'agent-answer empty-state';
    document.getElementById('agentAnswer').textContent = `执行失败：${error.message}`;
    document.getElementById('agentRounds').innerHTML = '<div class="empty-msg">工具调用链未完成</div>';
    document.getElementById('evidenceGallery').innerHTML = '<div class="empty-msg">没有可展示证据</div>';
    showToast(error.message, 'error');
  } finally {
    button.disabled = false;
    button.textContent = '执行闭环推理';
  }
}

window.useQuestion = useQuestion;
window.askAgent = askAgent;