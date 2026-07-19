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


function scopeLabel(scope) {
  return ({track_memory: '视频轨迹记忆', registry: '先验库', both: '跨记忆对照'})[scope] || valueText(scope);
}

function targetKindLabel(kind) {
  return ({hull: '舷号目标', description: '外观描述目标', all: '全部目标'})[kind] || valueText(kind);
}

function operationLabel(op) {
  return ({existence: '存在判断', list: '列表查询', time: '时间定位', count: '数量统计', explain: '证据解释'})[op] || valueText(op);
}

function relationLabel(rel) {
  return ({any: '不限库关系', in: '在库', out: '未在库'})[rel] || valueText(rel);
}

function intentSourceLabel(source) {
  return ({model: '规则表+模型', heuristic: '规则兜底'})[source] || valueText(source);
}

function questionTypeLabel(type) {
  return ({
    hull: '舷号查询',
    registry_hull: '先验库舷号查询',
    description: '描述目标查询',
    registry_description: '先验库描述查询',
    cross_reference: '跨记忆对应查询',
    track_list: '轨迹列表查询',
    registry_list: '先验库列表查询',
    relation_description: '描述+库关系查询',
    out_of_registry: '未在库船查询',
    in_registry: '在库船查询',
    count: '数量统计',
    description_count: '描述数量统计',
    registry_count: '先验库数量统计',
    registry_description_count: '先验库描述数量统计'
  })[type] || valueText(type);
}

async function loadAgentMemorySummary(showNotice = false) {
  const trackCount = document.getElementById('agentTrackCount');
  const registryStatus = document.getElementById('agentRegistryStatus');
  const modelName = document.getElementById('agentModelName');
  const refreshButton = document.getElementById('agentMemoryRefreshButton');
  const refreshStatus = document.getElementById('agentMemoryRefreshStatus');
  if (refreshButton) {
    refreshButton.disabled = true;
    refreshButton.textContent = '刷新中…';
  }
  if (refreshStatus) refreshStatus.textContent = '正在读取记忆状态…';
  try {
    const summary = await apiFetch('/api/agent/memory-summary');
    const numberOfTracks = summary.trackCount ?? summary.total ?? summary.tracks?.length ?? 0;
    const numberOfRegistryItems = summary.registryCount ?? summary.registryTotal ?? 0;
    if (trackCount) trackCount.textContent = `${numberOfTracks} 条轨迹`;
    if (registryStatus) registryStatus.textContent = `${numberOfRegistryItems} 个库项`;
    if (modelName) modelName.textContent = summary.recognitionModel || '模型已连接';
    const limit = document.getElementById('agentRoundLimit');
    if (limit) limit.textContent = `最多 ${summary.maxRounds || 3} 轮`;
    if (refreshStatus) {
      const updateTime = new Date().toLocaleTimeString('zh-CN', { hour12: false });
      refreshStatus.textContent = `已更新 ${updateTime}`;
    }
    if (showNotice && typeof showToast === 'function') showToast('记忆状态已刷新');
  } catch (error) {
    if (refreshStatus) refreshStatus.textContent = `刷新失败：${error.message || '服务不可用'}`;
    if (showNotice && typeof showToast === 'function') showToast(`记忆状态刷新失败：${error.message || '服务不可用'}`, 'error');
  } finally {
    if (refreshButton) {
      refreshButton.disabled = false;
      refreshButton.textContent = '刷新记忆状态';
    }
  }
}

function isRangeListQuestion(type) {
  return [
    'in_registry',
    'out_of_registry',
    'track_list',
    'relation_description',
    'count',
    'description_count',
    'cross_reference',
  ].includes(type);
}

function renderTracks(tracks, questionType = '') {
  if (!tracks?.length) return '';
  const rangeList = isRangeListQuestion(questionType);
  // topk/display_limit 只限制“单点检索展示”；范围枚举类问题有多少命中展示多少
  const visible = rangeList ? tracks : tracks.slice(0, 3);
  const rows = visible.map((track) => {
    const hull = track.finalHullNumber || track.hullNumber || '无稳定舷号';
    const start = Number(track.startTime ?? track.start_time);
    const end = Number(track.endTime ?? track.end_time);
    const time = Number.isFinite(start) && Number.isFinite(end) ? `${formatMonitorTime(start)}—${formatMonitorTime(end)}` : '时间未知';
    const score = Number(track.embeddingScore);
    return `<div class="track-summary"><strong>${escapeHtml(track.trackId || '未知轨迹')}</strong><span>${escapeHtml(hull)} · ${time}${Number.isFinite(score) ? ` · 相似度 ${score.toFixed(3)}` : ''}</span></div>`;
  }).join('');
  const more = (!rangeList && tracks.length > 3)
    ? `<div class="track-summary-more">仅展示前 3 条单点匹配，其余 ${tracks.length - 3} 条见证据区</div>`
    : (rangeList ? `<div class="track-summary-more">范围匹配共 ${tracks.length} 条</div>` : '');
  return `<div class="answer-tracks ${rangeList ? 'range-list' : ''}">${rows}${more}</div>`;
}


function renderRegistryHits(result) {
  const items = result?.registryItems || result?.registryMatches || [];
  if (!items.length) return '';
  const rows = items.slice(0, 5).map((item, index) => {
    const hull = item.hullNumber || item.hull || '未知舷号';
    const registryId = item.registryId || item.matchedRegistryId || '未知库项';
    const score = Number(item.embeddingScore);
    const band = item.scoreBand || item.verifyDecision || '';
    const desc = item.description || '';
    return `<div class="track-summary"><strong>${escapeHtml(hull)}</strong><span>库项 ${escapeHtml(String(registryId))}${Number.isFinite(score) ? ` · 相似度 ${score.toFixed(3)}` : ''}${band ? ` · ${escapeHtml(band)}` : ''}${desc ? ` · ${escapeHtml(desc)}` : ''}</span></div>`;
  }).join('');
  return `<div class="answer-tracks">${rows}</div>`;
}

function renderAgentAnswer(result) {
    const scope = Array.isArray(result.queryScope) ? `${formatMonitorTime(result.queryScope[0])}—${formatMonitorTime(result.queryScope[1])}` : '全部监控时间';
  const chain = (result.toolChain || []).map((item) => `<span class="tool-tag">${escapeHtml(item)}</span>`).join('');
  document.getElementById('agentAnswer').className = 'agent-answer';
  document.getElementById('agentAnswer').innerHTML = `
    <div class="answer-head"><strong>${escapeHtml(result.conclusion || '问答完成')}</strong><span class="status-tag ${result.uncertainty === 'sufficient' ? 'ok' : 'off'}">${escapeHtml(stateLabel(result.uncertainty))}</span></div>
    <p>${escapeHtml(result.answerText || '未生成回答')}</p>
    <div class="answer-meta"><span>问题类型：${escapeHtml(questionTypeLabel(result.questionType))}</span><span>查询范围：${escapeHtml(scope)}</span><span>命中数量：${Number((result.tracks || []).length)}</span></div>
    ${renderRegistryHits(result)}
    ${renderTracks(result.tracks, result.questionType)}
    ${chain ? `<div class="tool-tags">${chain}</div>` : ''}`;
}

function evidenceItem(type, id, trackId = null) {
  const prefix = type === 'video' ? 'Clip' : type === 'keyframe' ? 'Keyframe' : 'Database Reference';
  const route = type === 'video' ? 'clips' : type === 'keyframe' ? 'keyframes' : 'registry';
  const owner = trackId == null ? '' : `Track ${trackId} · `;
  return {type, id, label: `${owner}${prefix} ${id}`, url: `/api/evidence/${route}/${encodeURIComponent(id)}`};
}

function evidenceCard(item) {
  if (item.type === 'unavailable') {
    const owner = item.trackId == null ? 'Clip Evidence' : `Track ${escapeHtml(item.trackId)} · Clip Evidence`;
    return `<article class="evidence-card unavailable"><div class="evidence-placeholder"><strong>Clip Unavailable</strong><small>${escapeHtml(item.reason)}</small></div><span>${owner}</span></article>`;
  }
  const media = item.type === 'video'
    ? `<video controls preload="metadata" src="${item.url}"></video>`
    : `<img loading="lazy" src="${item.url}" alt="${escapeHtml(item.label)}">`;
  return `<article class="evidence-card">${media}<span title="${escapeHtml(item.label)}">${escapeHtml(item.label)}</span></article>`;
}

function clipErrorText(error) {
  return ({
    track_not_found: 'Track not found',
    trajectory_not_found: 'Trajectory data unavailable',
    source_evidence_unavailable: 'Source video or trajectory unavailable',
    segment_codec_unavailable: 'Video encoder unavailable',
    empty_target_segment: 'No target frames in the selected range',
    clip_unavailable: 'Clip has not been generated',
  })[error] || valueText(error);
}

function evidenceColumn(title, subtitle, items, emptyText) {
  const content = items.length ? items.map(evidenceCard).join('') : `<div class="evidence-column-empty">${escapeHtml(emptyText)}</div>`;
  return `<section class="evidence-column"><div class="evidence-column-head"><strong>${title}</strong><span>${subtitle}</span></div><div class="evidence-column-media">${content}</div></section>`;
}

function renderEvidence(evidence, displayGroups, emptyText = 'No evidence available') {
  const container = document.getElementById('evidenceGallery');
  const groups = (displayGroups || []).slice(0, 3);
  let clips = groups.map((group) => group.shipSegmentIds?.[0]
    ? evidenceItem('video', group.shipSegmentIds[0], group.trackId)
    : group.clipError ? {type: 'unavailable', reason: clipErrorText(group.clipError), trackId: group.trackId} : null).filter(Boolean);
  let keyframes = groups.map((group) => group.keyframeIds?.[0]
    ? evidenceItem('keyframe', group.keyframeIds[0], group.trackId) : null).filter(Boolean);
  const registryGroup = groups.find((group) => group.registryReferenceIds?.length);
  const registryIds = [...new Set(groups.flatMap(group => group.registryReferenceIds || []))].slice(0, 6);
  let database = registryIds.map(id => evidenceItem('registry', id, registryGroup?.trackId));

  if (!clips.length) clips = (evidence?.shipSegmentIds || []).slice(0, 3).map((id) => evidenceItem('video', id));
  if (!keyframes.length) keyframes = (evidence?.keyframeIds || []).slice(0, 3).map((id) => evidenceItem('keyframe', id));
  if (!database.length && evidence?.registryReferenceIds?.length) {
    database = [...new Set(evidence.registryReferenceIds)].slice(0, 6).map(id => evidenceItem('registry', id));
  }

  container.className = 'evidence-columns';
  container.innerHTML = [
    evidenceColumn('Clip Evidence', 'Target Vessel Clips', clips, emptyText),
    evidenceColumn('Keyframe Evidence', 'Recognition Frames', keyframes, emptyText),
    evidenceColumn('Database Evidence', 'Registry Reference Images', database, emptyText),
  ].join('');
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
  if (event.type === 'classification') {
    renderIntentAgentCard(event);
    return;
  }

  if (event.type === 'status' && (event.planMode || /规划模式|PlanAgent/.test(`${event.title || ''}${event.message || ''}`))) {
    let mode = stream.querySelector('.plan-mode-banner');
    if (!mode) {
      mode = document.createElement('div');
      mode.className = 'plan-mode-banner';
      stream.prepend(mode);
    }
    const label = event.planMode === 'autonomous' ? '完全自主' : (event.planMode === 'guided' ? '硬编码辅助' : (event.message || ''));
    mode.textContent = `PlanAgent 模式：${label}`;
    if (event.planMode) mode.dataset.mode = event.planMode;
  }

  if (event.type === 'status' && /IntentAgent|意图/.test(`${event.title || ''}${event.message || ''}`)) {
    let card = stream.querySelector('.intent-agent-card.pending, .intent-agent-card');
    if (!card || card.classList.contains('ready')) {
      card = document.createElement('section');
      card.className = 'intent-agent-card pending';
      stream.prepend(card);
    }
    card.className = 'intent-agent-card pending';
    card.innerHTML = `
      <div class="intent-agent-head">
        <div><span class="eyebrow">IntentAgent</span><strong>意图识别</strong></div>
        <em>解析中</em>
      </div>
      <div class="intent-agent-body"><p>${escapeHtml(event.message || '正在按规则表解析用户意图')}</p></div>
      <div class="intent-agent-meta"><time>${new Date().toLocaleTimeString('zh-CN', {hour12: false})}</time></div>`;
    stream.scrollTop = 0;
    return;
  }
  const item = document.createElement('div');
  item.className = `thought-system-card ${event.type}`;
  let detail = '';
  if (event.type === 'synthesis') {
    detail = `${stateLabel(event.state)} · 候选轨迹 ${Number(event.trackCount || 0)}`;
  }
  item.innerHTML = `<strong>${escapeHtml(event.title || '系统事件')}</strong><span>${escapeHtml(detail || event.message || '')}</span><time>${new Date().toLocaleTimeString('zh-CN', {hour12: false})}</time>`;
  stream.appendChild(item);
  stream.scrollTop = stream.scrollHeight;
}

function renderIntentAgentCard(event) {
  const stream = document.getElementById('agentThoughtStream');
  if (!stream) return;
  const timeScope = event.queryScope ? `${formatMonitorTime(event.queryScope[0])}—${formatMonitorTime(event.queryScope[1])}` : '全部监控时间';
  const rules = Array.isArray(event.selectedRules) && event.selectedRules.length ? event.selectedRules.join(' / ') : '未返回规则编号';
  const targetText = event.hullNumber || event.description || '无具体目标文本';
  const confidence = Number(event.intentConfidence);
  const confidenceText = Number.isFinite(confidence) ? confidence.toFixed(2) : '—';
  const chips = [
    `规则 ${rules}`,
    `范围 ${scopeLabel(event.targetScope)}`,
    `目标 ${targetKindLabel(event.targetKind)}`,
    `操作 ${operationLabel(event.operation)}`,
    `库关系 ${relationLabel(event.registryRelation)}`,
    `策略 ${questionTypeLabel(event.questionType)}`,
    `来源 ${intentSourceLabel(event.intentSource)}`,
  ];
  const acceptance = [
    event.expectedOutcome ? `验收 ${event.expectedOutcome}` : '',
    event.successCriteria ? `成功标准 ${event.successCriteria}` : '',
    event.nextAgentFocus ? `下一跳 ${event.nextAgentFocus}` : '',
  ].filter(Boolean);
  let card = stream.querySelector('.intent-agent-card');
  if (!card) {
    card = document.createElement('section');
    card.className = 'intent-agent-card';
    stream.prepend(card);
  }
  card.classList.remove('pending');
  card.classList.add('ready');
  card.innerHTML = `
    <div class="intent-agent-head">
      <div><span class="eyebrow">IntentAgent</span><strong>意图识别结果</strong></div>
      <em>完成</em>
    </div>
    <div class="intent-agent-summary">
      <p><strong>选定规则：</strong>${escapeHtml(rules)}</p>
      <p><strong>编译策略：</strong>${escapeHtml(questionTypeLabel(event.questionType))}（${escapeHtml(valueText(event.strategy))}）</p>
      <p><strong>查询目标：</strong>${escapeHtml(String(targetText))}</p>
      <p><strong>时间范围：</strong>${escapeHtml(timeScope)}</p>
      ${event.expectedOutcome ? `<p><strong>后续验收：</strong>${escapeHtml(String(event.expectedOutcome))}</p>` : ''}
      ${event.successCriteria ? `<p><strong>成功标准：</strong>${escapeHtml(String(event.successCriteria))}</p>` : ''}
      ${event.nextAgentFocus ? `<p><strong>下一跳重点：</strong>${escapeHtml(String(event.nextAgentFocus))}</p>` : ''}
    </div>
    <div class="intent-agent-chips">${[...chips, ...acceptance].map((item) => `<span>${escapeHtml(item)}</span>`).join('')}</div>
    <div class="intent-agent-meta">
      <span>置信度 ${escapeHtml(confidenceText)}</span>
      <span>${escapeHtml(event.message || '已选择规则并编译检索策略')}</span>
      <time>${new Date().toLocaleTimeString('zh-CN', {hour12: false})}</time>
    </div>`;
  stream.scrollTop = 0;
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
    if (event.planRepair) {
      content.textContent = `计划已修正：${event.planRepair}`;
    } else if (!streamedText && fallbackText) {
      content.textContent = fallbackText;
    }
    if (!content.textContent.trim()) content.textContent = event.fallback || '模型未返回可展示文本';
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
    renderEvidence(null, null, 'Collecting evidence...');
    const result = await streamAgentQuery(question);
    renderAgentAnswer(result);
    renderEvidence(result.evidence, result.displayGroups);
  } catch (error) {
    setThinkingState('推理失败', 'failed');
    document.getElementById('agentAnswer').className = 'agent-answer empty-state';
    document.getElementById('agentAnswer').textContent = `执行失败：${error.message}`;
    renderEvidence(null, null);
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
