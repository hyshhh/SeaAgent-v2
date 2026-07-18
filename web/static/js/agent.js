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

async function loadAgentMemorySummary() {
  const trackCount = document.getElementById('agentTrackCount');
  const registryStatus = document.getElementById('agentRegistryStatus');
  const modelName = document.getElementById('agentModelName');
  try {
    const [tracks, registry, config] = await Promise.all([
      apiFetch('/api/memory/tracks'),
      apiFetch('/api/ships'),
      apiFetch('/api/config'),
    ]);
    if (trackCount) trackCount.textContent = `${tracks.total || 0} 条轨迹`;
    if (registryStatus) registryStatus.textContent = `${registry.total || 0} 个库项`;
    if (modelName) modelName.textContent = config.models?.recognition || '模型已连接';
    const limit = document.getElementById('agentRoundLimit');
    if (limit) limit.textContent = `最多 ${config.pipeline?.maxRounds || 3} 轮`;
  } catch (error) {
    if (trackCount) trackCount.textContent = '读取失败';
    if (registryStatus) registryStatus.textContent = '读取失败';
    if (modelName) modelName.textContent = '服务未连接';
  }
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
  if (item.type === 'unavailable') {
    return `<article class="evidence-card unavailable"><div class="evidence-placeholder"><strong>目标船片段暂不可用</strong><small>${escapeHtml(item.reason)}</small></div><span>目标船片段</span></article>`;
  }
  const media = item.type === 'video'
    ? `<video controls preload="metadata" src="${item.url}"></video>`
    : `<img loading="lazy" src="${item.url}" alt="${escapeHtml(item.label)}">`;
  return `<article class="evidence-card">${media}<span title="${escapeHtml(item.label)}">${escapeHtml(item.label)}</span></article>`;
}

function clipErrorText(error) {
  return ({
    track_not_found: '未找到对应轨迹',
    trajectory_not_found: '轨迹框序列不存在',
    source_evidence_unavailable: '原视频或轨迹框不可用',
    segment_codec_unavailable: '视频编码器不可用',
    empty_target_segment: '指定范围内没有可用目标帧',
    clip_unavailable: '片段尚未生成',
  })[error] || valueText(error);
}

function renderEvidence(evidence, displayGroups) {
  const container = document.getElementById('evidenceGallery');
  const groups = (displayGroups || []).slice(0, 3).map((group) => {
    const items = [];
    if (group.shipSegmentIds?.[0]) items.push(evidenceItem('video', group.shipSegmentIds[0]));
    else if (group.clipError) items.push({type: 'unavailable', reason: clipErrorText(group.clipError)});
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

function modelSummaryText(value) {
  if (!value) return '';
  if (typeof value === 'string') return value;
  if (typeof value.summary === 'string') return value.summary;
  return JSON.stringify(value);
}

const agentRoles = {
  planner: {name: 'PlanAgent', label: '规划智能体', hint: '目标分析与工具规划'},
  observer: {name: 'ObserveAgent', label: '观察智能体', hint: '工具执行与证据整理'},
  reflector: {name: 'ReflectAgent', label: '反思智能体', hint: '充分性检查与退出判断'},
};

function setThinkingState(text, state = '') {
  const label = document.getElementById('agentThinkingState');
  const dot = document.getElementById('agentThinkingDot');
  if (label) label.textContent = text;
  if (dot) dot.className = state;
}

function resetThoughtStream() {
  const stream = document.getElementById('agentThoughtStream');
  if (stream) stream.innerHTML = '';
  setThinkingState('正在推理', 'active');
}

function ensureThoughtRound(roundNumber) {
  const stream = document.getElementById('agentThoughtStream');
  if (!stream) return null;
  let round = stream.querySelector(`.thought-round[data-round="${roundNumber}"]`);
  if (round) return round;
  round = document.createElement('section');
  round.className = 'thought-round';
  round.dataset.round = roundNumber;
  round.innerHTML = `
    <div class="thought-round-head"><strong>第 ${roundNumber} 轮推理</strong><span>PlanAgent → ObserveAgent → ReflectAgent</span></div>
    <div class="thought-agent-grid">
      ${Object.entries(agentRoles).map(([role, meta]) => `
        <article class="agent-thought-card ${role}" data-role="${role}">
          <div class="agent-thought-head"><div><strong>${meta.name}</strong><span>${meta.label}</span></div><em>等待</em></div>
          <p class="agent-thought-hint">${meta.hint}</p>
          <div class="agent-stream-text"><span class="agent-stream-placeholder">等待本轮执行…</span></div>
          <div class="agent-thought-tags"></div>
        </article>`).join('')}
    </div>`;
  stream.appendChild(round);
  stream.scrollTop = stream.scrollHeight;
  return round;
}

function ensureAgentCard(roundNumber, role) {
  const round = ensureThoughtRound(roundNumber);
  return round?.querySelector(`.agent-thought-card[data-role="${role}"]`) || null;
}

function agentTags(event) {
  if (event.role === 'planner') {
    return (event.calls || []).map((call) => `<span>${escapeHtml(call.tool || '工具')}</span>`).join('');
  }
  if (event.role === 'observer') {
    return (event.calls || []).map((call) => `<span class="${call.ok === false ? 'failed' : ''}">${escapeHtml(call.tool || call.id || '工具')} · ${call.skipped ? '跳过' : call.ok === false ? '失败' : '完成'}</span>`).join('');
  }
  if (event.role === 'reflector') {
    return `<span>${escapeHtml(stateLabel(event.state))}</span>${event.evidenceGap ? `<span>缺口：${escapeHtml(event.evidenceGap)}</span>` : ''}`;
  }
  return '';
}

function appendSystemThought(event) {
  const stream = document.getElementById('agentThoughtStream');
  if (!stream) return;
  const item = document.createElement('div');
  item.className = `thought-system-card ${event.type}`;
  let detail = '';
  if (event.type === 'classification') {
    const scope = event.queryScope ? `${event.queryScope[0]}秒—${event.queryScope[1]}秒` : '全视频';
    detail = `${questionTypeLabel(event.questionType)} · ${scope}`;
  } else if (event.type === 'synthesis') {
    detail = `${stateLabel(event.state)} · 候选轨迹 ${Number(event.trackCount || 0)}`;
  }
  item.innerHTML = `<strong>${escapeHtml(event.title || '系统事件')}</strong><span>${escapeHtml(detail || event.message || '')}</span><time>${new Date().toLocaleTimeString('zh-CN', {hour12: false})}</time>`;
  stream.appendChild(item);
  stream.scrollTop = stream.scrollHeight;
}

function appendThoughtEvent(event) {
  if (event.type === 'complete') {
    setThinkingState('推理完成', 'completed');
    return;
  }
  if (event.type === 'error') {
    setThinkingState('推理失败', 'failed');
    appendSystemThought(event);
    return;
  }
  if (event.type === 'agent_start') {
    const card = ensureAgentCard(event.round, event.role);
    if (!card) return;
    card.classList.add('active');
    card.querySelector('.agent-thought-head em').textContent = '生成中';
    const content = card.querySelector('.agent-stream-text');
    content.innerHTML = '<span class="agent-stream-cursor"></span>';
  } else if (event.type === 'agent_delta') {
    const card = ensureAgentCard(event.round, event.role);
    if (!card || !event.delta) return;
    const content = card.querySelector('.agent-stream-text');
    const cursor = content.querySelector('.agent-stream-cursor');
    content.insertBefore(document.createTextNode(event.delta), cursor);
  } else if (event.type === 'agent_end') {
    const card = ensureAgentCard(event.round, event.role);
    if (!card) return;
    const content = card.querySelector('.agent-stream-text');
    const streamedText = content.textContent.trim();
    const fallbackText = modelSummaryText(event.modelSummary);
    if (!streamedText && fallbackText) content.textContent = fallbackText;
    if (!content.textContent.trim()) content.textContent = event.fallback || '模型未返回可展示内容';
    card.querySelector('.agent-stream-cursor')?.remove();
    card.classList.remove('active');
    card.classList.toggle('failed', Boolean(event.fallback));
    card.querySelector('.agent-thought-head em').textContent = event.fallback ? '调用失败' : '完成';
    card.querySelector('.agent-thought-tags').innerHTML = agentTags(event);
  } else {
    appendSystemThought(event);
  }
  const stream = document.getElementById('agentThoughtStream');
  if (stream) stream.scrollTop = stream.scrollHeight;
}

async function streamAgentQuery(question) {
  const response = await fetch('/api/agent/query/stream', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({question}),
  });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail || `请求失败：${response.status}`);
  }
  if (!response.body) throw new Error('当前浏览器不支持流式响应');

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let result = null;
  while (true) {
    const {value, done} = await reader.read();
    buffer += decoder.decode(value || new Uint8Array(), {stream: !done});
    const lines = buffer.split('\n');
    buffer = lines.pop() || '';
    for (const line of lines) {
      if (!line.trim()) continue;
      const event = JSON.parse(line);
      appendThoughtEvent(event);
      if (event.type === 'complete') result = event.result;
      if (event.type === 'error') throw new Error(event.message || '闭环推理失败');
    }
    if (done) break;
  }
  if (buffer.trim()) {
    const event = JSON.parse(buffer);
    appendThoughtEvent(event);
    if (event.type === 'complete') result = event.result;
    if (event.type === 'error') throw new Error(event.message || '闭环推理失败');
  }
  if (!result) throw new Error('未收到最终回答');
  return result;
}

async function askAgent() {
  const question = document.getElementById('agentQuestion').value.trim();
  const button = document.getElementById('btnAskAgent');
  if (!question) return showToast('请输入问题', 'error');
  try {
    button.disabled = true;
    button.textContent = '推理中…';
    resetThoughtStream();
    document.getElementById('agentAnswer').className = 'agent-answer empty-state';
    document.getElementById('agentAnswer').textContent = '正在基于轨迹记忆生成回答…';
    document.getElementById('agentRounds').innerHTML = '<div class="empty-msg">正在生成闭环协作记录…</div>';
    document.getElementById('evidenceGallery').innerHTML = '<div class="empty-msg">正在收集视觉证据…</div>';
    const result = await streamAgentQuery(question);
    renderAgentAnswer(result);
    renderAgentRounds(result.rounds);
    renderEvidence(result.evidence, result.displayGroups);
  } catch (error) {
    setThinkingState('推理失败', 'failed');
    document.getElementById('agentAnswer').className = 'agent-answer empty-state';
    document.getElementById('agentAnswer').textContent = `执行失败：${error.message}`;
    document.getElementById('agentRounds').innerHTML = '<div class="empty-msg">闭环协作记录未完成</div>';
    document.getElementById('evidenceGallery').innerHTML = '<div class="empty-msg">没有可展示证据</div>';
    showToast(error.message, 'error');
  } finally {
    button.disabled = false;
    button.textContent = '执行闭环推理';
  }
}

document.addEventListener('DOMContentLoaded', () => {
  const input = document.getElementById('agentQuestion');
  input?.addEventListener('keydown', (event) => {
    if (event.ctrlKey && event.key === 'Enter') {
      event.preventDefault();
      askAgent();
    }
  });
});

window.useQuestion = useQuestion;
window.askAgent = askAgent;
window.loadAgentMemorySummary = loadAgentMemorySummary;
