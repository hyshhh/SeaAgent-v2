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
  const qaMemoryCount = document.getElementById('agentQaMemoryCount');
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
    if (qaMemoryCount) qaMemoryCount.textContent = `问答记忆 ${summary.qaSessionCount || 0} 条`;
    const limit = document.getElementById('agentRoundLimit');
    if (limit) {
      const baseRounds = summary.baseMaxRounds || summary.maxRounds || 3;
      const recoveryRounds = summary.acceptanceRecoveryRounds || 0;
      limit.textContent = recoveryRounds > 0 ? `最多 ${baseRounds} 轮 + ${recoveryRounds} 轮验收补偿` : `最多 ${baseRounds} 轮`;
    }
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

async function clearAgentMemory() {
  const askButton = document.getElementById('btnAskAgent');
  const clearButton = document.getElementById('agentMemoryClearButton');
  if (askButton?.disabled) return showToast('当前正在推理，请等待本轮完成后再清除', 'error');
  if (!window.confirm('确定清除全部问答会话、推理轮次和问答证据吗？轨迹记忆与先验库不会删除。')) return;
  try {
    if (clearButton) {
      clearButton.disabled = true;
      clearButton.textContent = '清除中…';
    }
    const result = await apiFetch('/api/agent/memory', {method: 'DELETE'});
    resetThoughtStream();
    const answer = document.getElementById('agentAnswer');
    if (answer) {
      answer.className = 'agent-answer empty-state';
      answer.textContent = '问答记忆已清除，可以开始新的查询。';
    }
    renderEvidence(null, null, 'No evidence available');
    await loadAgentMemorySummary(false);
    showToast(result.message || '问答记忆已清除');
  } catch (error) {
    showToast(`问答记忆清除失败：${error.message || '服务不可用'}`, 'error');
  } finally {
    if (clearButton) {
      clearButton.disabled = false;
      clearButton.textContent = '清除问答记忆';
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

function showAgentProcessView() {
  const processView = document.getElementById('agentProcessView');
  const finalView = document.getElementById('agentFinalView');
  if (processView) processView.hidden = false;
  if (finalView) finalView.hidden = true;
}

function showAgentFinalView() {
  const processView = document.getElementById('agentProcessView');
  const finalView = document.getElementById('agentFinalView');
  if (processView) processView.hidden = true;
  if (finalView) finalView.hidden = false;
}

function renderAgentAnswer(result) {
  showAgentFinalView();
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

function evidenceItem(type, id, trackId = null, options = {}) {
  const prefix = type === 'video' ? 'Clip' : type === 'keyframe' ? 'Keyframe' : 'Database Reference';
  const route = type === 'video' ? 'clips' : type === 'keyframe' ? 'keyframes' : 'registry';
  const owner = trackId == null ? '' : `Track ${trackId} · `;
  const scope = Array.isArray(options.timeRange)
    ? `?startTime=${encodeURIComponent(options.timeRange[0])}&endTime=${encodeURIComponent(options.timeRange[1])}`
    : '';
  const url = type === 'video' && options.trackClip
    ? `/api/evidence/tracks/${encodeURIComponent(id)}/clip${scope}`
    : `/api/evidence/${route}/${encodeURIComponent(id)}`;
  const posterUrl = type !== 'video' ? null : options.posterKeyframeId
    ? `/api/evidence/keyframes/${encodeURIComponent(options.posterKeyframeId)}`
    : `/api/evidence/clips/${encodeURIComponent(id)}/poster`;
  return {type, id, label: options.label || `${owner}${prefix} ${id}`, url, posterUrl};
}

let evidenceVideoObserver = null;
const visibleEvidenceVideos = new Set();

function balanceEvidencePlayback() {
  const videos = [...visibleEvidenceVideos].filter((video) => video.isConnected);
  videos.sort((left, right) => {
    const position = left.compareDocumentPosition(right);
    return position & Node.DOCUMENT_POSITION_FOLLOWING ? -1 : 1;
  });
  videos.forEach((video, index) => {
    if (index === 0) video.play().catch(() => {});
    else video.pause();
  });
}

function evidenceCard(item) {
  if (item.type === 'unavailable' || item.type === 'missing') {
    const owner = item.label || (item.trackId == null ? 'Evidence' : `Track ${item.trackId} · Evidence`);
    const title = item.type === 'unavailable' ? 'Clip Unavailable' : 'No Matched Evidence';
    return `<article class="evidence-card unavailable"><div class="evidence-placeholder"><strong>${title}</strong><small>${escapeHtml(item.reason)}</small></div><span title="${escapeHtml(owner)}">${escapeHtml(owner)}</span></article>`;
  }
  const media = item.type === 'video'
    ? `<button class="evidence-video-loader" type="button" data-src="${escapeHtml(item.url)}" data-poster="${escapeHtml(item.posterUrl || '')}" onclick="loadEvidenceVideo(this, true)">${item.posterUrl ? `<img loading="lazy" src="${escapeHtml(item.posterUrl)}" alt="${escapeHtml(item.label)}">` : '<div class="evidence-video-placeholder"></div>'}<span class="evidence-play-badge">▶</span></button>`
    : `<img loading="lazy" src="${item.url}" alt="${escapeHtml(item.label)}">`;
  return `<article class="evidence-card">${media}<span title="${escapeHtml(item.label)}">${escapeHtml(item.label)}</span></article>`;
}

function loadEvidenceVideo(trigger, shouldPlay = true) {
  const source = trigger?.dataset?.src;
  if (!source || trigger.dataset.loading === 'true') return null;
  trigger.dataset.loading = 'true';
  evidenceVideoObserver?.unobserve(trigger);
  const video = document.createElement('video');
  video.controls = true;
  video.preload = 'metadata';
  video.playsInline = true;
  video.autoplay = true;
  video.defaultMuted = true;
  video.muted = true;
  video.loop = true;
  video.src = source;
  if (trigger.dataset.poster) video.poster = trigger.dataset.poster;
  trigger.replaceWith(video);
  evidenceVideoObserver?.observe(video);
  if (shouldPlay) video.play().catch(() => {});
  return video;
}

function observeEvidenceVideos(container) {
  evidenceVideoObserver?.disconnect();
  evidenceVideoObserver = null;
  visibleEvidenceVideos.clear();
  const loaders = [...container.querySelectorAll('.evidence-video-loader')];
  if (!loaders.length) return;
  if (!('IntersectionObserver' in window)) {
    loaders.slice(0, 1).forEach((loader) => loadEvidenceVideo(loader, true));
    return;
  }
  evidenceVideoObserver = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      const target = entry.target;
      if (target.matches('.evidence-video-loader')) {
        if (entry.isIntersecting && entry.intersectionRatio >= 0.35) {
          evidenceVideoObserver.unobserve(target);
          loadEvidenceVideo(target, false);
        }
        return;
      }
      if (target.tagName !== 'VIDEO') return;
      if (entry.isIntersecting && entry.intersectionRatio >= 0.35) {
        visibleEvidenceVideos.add(target);
      } else {
        visibleEvidenceVideos.delete(target);
        target.pause();
      }
      balanceEvidencePlayback();
    });
  }, {root: null, rootMargin: '0px', threshold: [0, 0.35, 0.7]});
  loaders.forEach((loader) => evidenceVideoObserver.observe(loader));
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
  const available = items.filter((item) => !['missing', 'unavailable'].includes(item.type)).length;
  const count = items.length ? `${available}/${items.length}` : '0';
  return `<section class="evidence-column"><div class="evidence-column-head"><strong>${title}</strong><span>${subtitle} · ${count}</span></div><div class="evidence-column-media">${content}</div></section>`;
}

function evidenceTrackCell(title, subtitle, item) {
  return `<section class="evidence-track-cell"><div class="evidence-track-cell-head"><strong>${title}</strong><span>${subtitle}</span></div>${evidenceCard(item)}</section>`;
}

function evidenceTrackRow(group, index) {
  const trackId = group.trackId ?? index + 1;
  const clip = group.shipSegmentIds?.[0]
    ? evidenceItem('video', group.shipSegmentIds[0], trackId)
    : group.clipTrackId
      ? evidenceItem('video', group.clipTrackId, trackId, {trackClip: true, posterKeyframeId: group.keyframeIds?.[0], timeRange: group.clipTimeRange})
      : group.clipError
        ? {type: 'unavailable', reason: clipErrorText(group.clipError), trackId, label: `Track ${trackId} · Clip Evidence`}
        : missingEvidence(trackId, 'clip');
  const keyframe = group.keyframeIds?.[0]
    ? evidenceItem('keyframe', group.keyframeIds[0], trackId)
    : missingEvidence(trackId, 'keyframe');
  const database = group.registryReferenceIds?.[0]
    ? evidenceItem('registry', group.registryReferenceIds[0], trackId, {label: `Track ${trackId} · Database Reference`})
    : missingEvidence(trackId, 'registry');
  const hull = String(group.hullNumber || '').trim();
  return `<article class="evidence-track-row"><div class="evidence-track-row-head"><div><span>Track Evidence</span><strong>Track ${escapeHtml(trackId)}</strong></div>${hull ? `<em>Hull ${escapeHtml(hull)}</em>` : '<em>Hull Unknown</em>'}</div><div class="evidence-track-row-media">${evidenceTrackCell('Clip', 'Target Vessel Segment', clip)}${evidenceTrackCell('Keyframe', 'Recognition Frame', keyframe)}${evidenceTrackCell('Registry', 'Database Reference', database)}</div></article>`;
}

function representativeRegistryEvidence(registryItems) {
  const seen = new Set();
  const items = [];
  for (const registry of registryItems || []) {
    const owner = String(registry.hullNumber || registry.registryId || '').trim();
    const referenceId = (registry.references || []).find((item) => item?.referenceId)?.referenceId
      || (registry.registryReferenceIds || [])[0];
    if (!referenceId || seen.has(owner)) continue;
    seen.add(owner);
    items.push(evidenceItem('registry', referenceId, null, {label: `Hull ${owner} · Database Reference`}));
  }
  return items;
}

function missingEvidence(trackId, category) {
  const labels = {
    clip: `Track ${trackId} · Clip Evidence`,
    keyframe: `Track ${trackId} · Keyframe Evidence`,
    registry: `Track ${trackId} · Database Evidence`,
  };
  const reasons = {
    clip: 'No target vessel clip',
    keyframe: 'No active keyframe',
    registry: 'No matched registry reference',
  };
  return {type: 'missing', trackId, label: labels[category], reason: reasons[category]};
}

function synchronizeEvidenceColumns(container) {
  const columns = [...container.querySelectorAll('.evidence-column-media')];
  let syncing = false;
  columns.forEach((source) => source.addEventListener('scroll', () => {
    if (syncing) return;
    syncing = true;
    columns.forEach((target) => {
      if (target !== source) target.scrollTop = source.scrollTop;
    });
    requestAnimationFrame(() => { syncing = false; });
  }, {passive: true}));
}

function renderEvidence(evidence, displayGroups, emptyText = 'No evidence available', registryItems = [], result = null) {
  const container = document.getElementById('evidenceGallery');
  evidenceVideoObserver?.disconnect();
  visibleEvidenceVideos.forEach((video) => video.pause());
  visibleEvidenceVideos.clear();
  const groups = displayGroups || [];
  const registryOnly = result?.targetScope === 'registry';
  if (registryOnly) {
    let database = representativeRegistryEvidence(registryItems);
    if (!database.length) {
      database = [...new Set(evidence?.registryReferenceIds || [])].map((id) => evidenceItem('registry', id));
    }
    container.className = 'evidence-columns registry-only';
    container.innerHTML = evidenceColumn('Database Evidence', 'Registry Reference Images', database, emptyText);
    return;
  }

  let rows = groups;
  if (!rows.length) {
    const rawClips = evidence?.shipSegmentIds || [];
    const rawKeyframes = evidence?.keyframeIds || [];
    const rawDatabase = evidence?.registryReferenceIds || [];
    const rowCount = Math.max(rawClips.length, rawKeyframes.length, rawDatabase.length);
    rows = Array.from({length: rowCount}, (_, index) => ({
      trackId: index + 1,
      shipSegmentIds: rawClips[index] ? [rawClips[index]] : [],
      keyframeIds: rawKeyframes[index] ? [rawKeyframes[index]] : [],
      registryReferenceIds: rawDatabase[index] ? [rawDatabase[index]] : [],
    }));
  }

  container.className = 'evidence-track-list';
  container.innerHTML = rows.length
    ? rows.map((group, index) => evidenceTrackRow(group, index)).join('')
    : `<div class="evidence-track-empty">${escapeHtml(emptyText)}</div>`;
  observeEvidenceVideos(container);
}

function modelSummaryText(value) {
  if (!value) return '';
  if (typeof value === 'string') return value;
  if (typeof value.summary === 'string') return value.summary;
  return [value.goal, value.reason, value.answerHint].filter(Boolean).join('；');
}

function compactAgentValue(value, maxLength = 160) {
  const text = String(value || '').replace(/\s+/g, ' ').trim();
  return text.length > maxLength ? `${text.slice(0, maxLength - 1)}…` : text;
}
function compactToolCall(call) {
  const tool = call.tool || call.id || '工具';
  const status = call.skipped ? '跳过' : call.ok === false ? '失败' : '完成';
  const details = [];
  if (Number.isFinite(Number(call.trackCount))) details.push(`${call.trackCount}条轨迹`);
  if (Number.isFinite(Number(call.keyframeCount))) details.push(`${call.keyframeCount}张关键帧`);
  if (Number.isFinite(Number(call.matchCount))) details.push(`${call.matchCount}条匹配`);
  if (Number.isFinite(Number(call.registryCount))) details.push(`${call.registryCount}个库项`);
  if (call.decision) details.push(String(call.decision));
  if (call.hasMore) details.push('还有下一页');
  if (call.error) details.push(compactAgentValue(call.error, 70));
  return `${tool} · ${status}${details.length ? ` · ${details.join('，')}` : ''}`;
}

function compactAgentText(event) {
  const summary = modelSummaryText(event.modelSummary).trim();
  const calls = Array.isArray(event.calls) ? event.calls : [];
  if (event.role === 'planner') {
    const model = event.modelSummary || {};
    const lines = [];
    if (model.goal) lines.push(`目标：${compactAgentValue(model.goal, 100)}`);
    if (calls.length) lines.push(`计划：${calls.map((call) => call.tool || '工具').join(' → ')}`);
    if (model.reason) lines.push(`依据：${compactAgentValue(model.reason, 140)}`);
    if (event.planRepair) lines.push('校验：上一版计划无效，已重新规划');
    return lines.join('\n') || '本轮未生成可执行计划';
  }
  if (event.role === 'observer') {
    const toolLines = calls.map(compactToolCall);
    return [toolLines.length ? toolLines.join('\n') : '', summary ? `观察：${compactAgentValue(summary, 160)}` : '']
      .filter(Boolean).join('\n') || '本轮没有工具结果';
  }
  if (event.role === 'reflector') {
    return [
      event.acceptanceGoal ? `验收：${compactAgentValue(event.acceptanceGoal, 110)}` : '',
      `判断：${stateLabel(event.state)}`,
      event.evidenceGap ? `缺口：${compactAgentValue(event.evidenceGap, 100)}` : '',
      event.nextAction ? `下一步：${compactAgentValue(event.nextAction, 110)}` : '',
      summary ? `依据：${compactAgentValue(summary, 160)}` : '',
    ].filter(Boolean).join('\n');
  }
  return summary;
}

const agentRoles = {
  planner: {name: 'PlanAgent', label: '规划智能体', hint: '只展示模型规划，不执行工具', stage: '01'},
  observer: {name: 'ObserveAgent', label: '观察智能体', hint: '执行工具并返回关键结果', stage: '02'},
  reflector: {name: 'ReflectAgent', label: '反思智能体', hint: '判断证据是否足够及下一步', stage: '03'},
};

function setThinkingState(text, state = '') {
  const label = document.getElementById('agentThinkingState');
  const dot = document.getElementById('agentThinkingDot');
  if (label) label.textContent = text;
  if (dot) dot.className = state;
}

let agentPlanItems = [];
let agentPlanLocked = false;
let activeAgentRound = 0;

const toolActionLabels = {
  getTrack: '读取目标轨迹',
  getFrames: '读取正式关键帧',
  getClip: '生成目标船片段',
  getRegistry: '读取指定先验库项',
  matchHull: '精确匹配舷号',
  listRegistry: '读取完整先验库',
  matchText: '执行描述特征匹配',
  matchImage: '执行图像特征匹配',
  verifyTarget: '核验灰区视觉证据',
  showEvidence: '整理并展示证据',
  dedupTracks: '执行跨轨迹去重',
};

function planStatusLabel(status) {
  return ({pending: '等待', running: '执行中', completed: '完成', failed: '失败', skipped: '跳过'})[status] || '等待';
}

function initializePlanBlueprint(blueprint) {
  if (agentPlanLocked || !Array.isArray(blueprint) || !blueprint.length) return;
  agentPlanItems = blueprint.map((step, index) => {
    const tools = Array.isArray(step.tools) ? step.tools.map(String) : [];
    return {
      key: String(step.stepId || `plan-step-${index + 1}`),
      stepId: String(step.stepId || `plan-step-${index + 1}`),
      id: '',
      title: String(step.title || toolActionLabels[tools[0]] || '执行计划步骤'),
      tools,
      tool: tools[0] || '',
      round: 0,
      status: 'pending',
      optional: Boolean(step.optional),
      detail: step.optional ? '条件步骤，证据进入灰区时执行' : '等待执行',
    };
  });
  agentPlanLocked = true;
  const round = document.getElementById('agentPlanRound');
  const decision = document.getElementById('agentPlanDecision');
  if (round) round.textContent = `完整计划 · ${agentPlanItems.length} 步`;
  if (decision) {
    decision.className = 'agent-plan-decision active';
    decision.textContent = '完整计划已生成，后续各轮只更新原步骤状态。';
  }
  renderPlanProgress();
}
function renderPlanProgress() {
  const container = document.getElementById('agentPlanProgress');
  const meter = document.getElementById('agentPlanMeter');
  if (!container) return;
  const total = agentPlanItems.length;
  const completed = agentPlanItems.filter((item) => ['completed', 'skipped'].includes(item.status)).length;
  if (meter) meter.style.width = total ? `${Math.round((completed / total) * 100)}%` : '0%';
  if (!total) {
    container.innerHTML = '<div class="agent-plan-empty">PlanAgent 生成计划后，将在这里显示待执行步骤。</div>';
    return;
  }
  const visibleItems = agentPlanItems;
  container.innerHTML = visibleItems.map((item, index) => {
    const symbol = item.status === 'completed' ? '✓' : item.status === 'failed' ? '!' : item.status === 'skipped' ? '—' : '';
    const action = item.title || toolActionLabels[item.tool] || '执行工具步骤';
    const detail = item.detail ? `<small>${escapeHtml(compactAgentValue(item.detail, 90))}</small>` : `<small>第 ${item.round} 轮计划</small>`;
    return `
      <div class="agent-plan-step ${escapeHtml(item.status)}" data-plan-key="${escapeHtml(item.key)}">
        <span class="agent-plan-icon">${escapeHtml(symbol)}</span>
        <div class="agent-plan-copy"><strong>${escapeHtml(action)}</strong>${detail}</div>
        <div class="agent-plan-tool"><code>${escapeHtml((item.tools || [item.tool]).filter(Boolean).join(" / "))}</code><em>${escapeHtml(planStatusLabel(item.status))}</em></div>
      </div>`;
  }).join('');
  const running = container.querySelector('.agent-plan-step.running');
  if (running) running.scrollIntoView({block: 'nearest'});
}

function resetPlanProgress() {
  agentPlanItems = [];
  agentPlanLocked = false;
  const round = document.getElementById('agentPlanRound');
  const decision = document.getElementById('agentPlanDecision');
  if (round) round.textContent = '等待计划';
  if (decision) {
    decision.className = 'agent-plan-decision';
    decision.textContent = '等待 ReflectAgent 验收当前计划。';
  }
  renderPlanProgress();
}

function syncPlanProgress(event) {
  initializePlanBlueprint(event.planBlueprint);
  const roundNumber = Number(event.round || 1);
  const calls = Array.isArray(event.calls) ? event.calls : [];
  calls.forEach((call, index) => {
    const stepId = String(call.planStepId || '');
    const tool = String(call.tool || call.id || '工具');
    let item = agentPlanItems.find((entry) => stepId && entry.stepId === stepId);
    if (!item) item = agentPlanItems.find((entry) => (entry.tools || []).includes(tool));
    if (!item && !agentPlanLocked) {
      item = {
        key: stepId || `fallback-step-${index + 1}`,
        stepId: stepId || `fallback-step-${index + 1}`,
        id: '',
        title: toolActionLabels[tool] || '执行工具步骤',
        tools: [tool],
        tool,
        round: 0,
        status: 'pending',
        optional: false,
        detail: '等待执行',
      };
      agentPlanItems.push(item);
    }
    if (!item) return;
    item.id = String(call.id || item.id || '');
    item.round = roundNumber;
    item.status = 'pending';
    item.detail = `第 ${roundNumber} 轮等待执行`;
  });
  const round = document.getElementById('agentPlanRound');
  const decision = document.getElementById('agentPlanDecision');
  if (round) round.textContent = `完整计划 · 第 ${roundNumber} 轮`;
  if (decision) {
    decision.className = 'agent-plan-decision active';
    decision.textContent = calls.length ? `ObserveAgent 正在执行第 ${roundNumber} 轮计划。` : '本轮没有可执行步骤，等待 ReflectAgent 判断。';
  }
  renderPlanProgress();
}

function updatePlanProgressFromTool(event) {
  const roundNumber = Number(event.round || activeAgentRound || 1);
  const id = String(event.id || '');
  const stepId = String(event.planStepId || '');
  const tool = String(event.tool || event.id || '工具');
  let item = agentPlanItems.find((entry) => stepId && entry.stepId === stepId);
  if (!item) item = agentPlanItems.find((entry) => (entry.tools || []).includes(tool));
  if (!item && !agentPlanLocked) {
    item = {
      key: stepId || `fallback-step-${agentPlanItems.length + 1}`,
      stepId: stepId || `fallback-step-${agentPlanItems.length + 1}`,
      id,
      title: toolActionLabels[tool] || '执行工具步骤',
      tools: [tool],
      tool,
      round: roundNumber,
      status: 'pending',
      optional: false,
      detail: '',
    };
    agentPlanItems.push(item);
  }
  if (!item) return;
  item.id = id || item.id;
  item.round = roundNumber;
  if (event.phase === 'completed' && event.ok !== false) item.status = 'completed';
  else if (event.phase === 'failed' || event.ok === false) item.status = 'failed';
  else if (event.phase === 'skipped' || event.skipped) item.status = 'skipped';
  else item.status = 'running';
  item.detail = event.error || event.summary || `第 ${roundNumber} 轮${planStatusLabel(item.status)}`;
  renderPlanProgress();
}
function updatePlanFromObservation(event) {
  (event.calls || []).forEach((call) => {
    updatePlanProgressFromTool({
      ...call,
      round: event.round,
      phase: call.skipped ? 'skipped' : call.ok === false ? 'failed' : 'completed',
    });
  });
}

function updatePlanReflection(event) {
  const decision = document.getElementById('agentPlanDecision');
  if (!decision) return;
  const state = String(event.state || 'uncertain');
  decision.className = `agent-plan-decision ${state}`;
  if (state === 'sufficient') {
    agentPlanItems.forEach((item) => { if (!['failed', 'skipped'].includes(item.status)) item.status = item.optional && item.status === 'pending' ? 'skipped' : 'completed'; });
    renderPlanProgress();
    decision.textContent = `第 ${event.round || activeAgentRound} 轮验收通过，正在生成最终回答。`;
  } else if (state === 'replan') {
    decision.textContent = `继续规划：${compactAgentValue(event.nextAction || event.evidenceGap || '需要补充证据', 150)}`;
  } else if (state === 'conflict') {
    decision.textContent = `证据冲突：${compactAgentValue(event.evidenceGap || event.nextAction || '需要重新检查证据', 150)}`;
  } else {
    decision.textContent = `当前无法确认：${compactAgentValue(event.evidenceGap || event.nextAction || '证据不足', 150)}`;
  }
}

function resetFixedAgentCards() {
  activeAgentRound = 0;
  document.querySelectorAll('#agentThoughtStream .agent-thought-card').forEach((card) => {
    const role = card.dataset.role;
    card.dataset.round = '0';
    card.dataset.streamText = '';
    card.classList.remove('active', 'failed');
    card._toolLogs = [];
    const round = card.querySelector('.agent-card-round');
    const state = card.querySelector('.agent-thought-head em');
    const content = card.querySelector('.agent-stream-text');
    const tags = card.querySelector('.agent-thought-tags');
    if (round) round.textContent = '等待轮次';
    if (state) state.textContent = '等待';
    if (content) content.innerHTML = `<span class="agent-stream-placeholder">${role === 'planner' ? '等待规划…' : role === 'observer' ? '等待执行…' : '等待验收…'}</span>`;
    if (tags) tags.innerHTML = '';
  });
}

function prepareAgentRound(roundNumber) {
  const normalizedRound = Number(roundNumber || 1);
  if (activeAgentRound === normalizedRound) return;
  activeAgentRound = normalizedRound;
  document.querySelectorAll('#agentThoughtStream .agent-thought-card').forEach((card) => {
    const role = card.dataset.role;
    card.dataset.round = String(normalizedRound);
    card.dataset.streamText = '';
    card.classList.remove('active', 'failed');
    card._toolLogs = [];
    const round = card.querySelector('.agent-card-round');
    const state = card.querySelector('.agent-thought-head em');
    const content = card.querySelector('.agent-stream-text');
    const tags = card.querySelector('.agent-thought-tags');
    if (round) round.textContent = `第 ${normalizedRound} 轮`;
    if (state) state.textContent = '等待';
    if (content) content.innerHTML = `<span class="agent-stream-placeholder">${role === 'planner' ? '等待本轮规划…' : role === 'observer' ? '等待本轮执行…' : '等待本轮验收…'}</span>`;
    if (tags) tags.innerHTML = '';
  });
}

function resetThoughtStream() {
  showAgentProcessView();
  resetFixedAgentCards();
  resetPlanProgress();
  const intent = document.getElementById('agentIntentResult');
  const intentState = document.getElementById('agentIntentState');
  const mode = document.getElementById('agentPlanMode');
  if (intent) {
    intent.className = 'intent-agent-card pending compact';
    intent.innerHTML = '<div class="intent-agent-body"><p>正在解析问题并生成验收目标…</p></div>';
  }
  if (intentState) intentState.textContent = '识别中';
  if (mode) mode.textContent = 'PlanAgent 负责规划，ObserveAgent 负责执行，ReflectAgent 负责验收。';
  setThinkingState('正在推理', 'active');
}

function ensureAgentCard(roundNumber, role) {
  prepareAgentRound(roundNumber);
  return document.querySelector(`#agentThoughtStream .agent-thought-card[data-role="${role}"]`);
}

function agentTags(event) {
  if (event.role === 'planner') {
    const tools = (event.calls || []).map((call) => `<span>${escapeHtml(call.tool || '工具')}</span>`).join('');
    const repair = event.planRepair ? `<span class="failed" title="${escapeHtml(event.planRepair)}">计划已纠正</span>` : '';
    return tools + repair;
  }
  if (event.role === 'observer') {
    return (event.calls || []).map((call) => {
      const failed = call.ok === false;
      const error = failed && call.error ? `：${valueText(call.error)}` : '';
      const title = failed && call.error ? ` title="${escapeHtml(valueText(call.error))}"` : '';
      return `<span class="${failed ? 'failed' : ''}"${title}>${escapeHtml(call.tool || call.id || '工具')} · ${call.skipped ? '跳过' : failed ? `失败${escapeHtml(error)}` : '完成'}</span>`;
    }).join('');
  }
  if (event.role === 'reflector') {
    return `<span>${escapeHtml(stateLabel(event.state))}</span>${event.evidenceGap ? `<span>缺口：${escapeHtml(event.evidenceGap)}</span>` : ''}`;
  }
  return '';
}

function appendSystemThought(event) {
  if (event.type === 'classification') {
    renderIntentAgentCard(event);
    return;
  }
  if (event.type === 'status' && (event.planMode || /规划模式|PlanAgent/.test(`${event.title || ''}${event.message || ''}`))) {
    const mode = document.getElementById('agentPlanMode');
    const label = event.planMode === 'autonomous' ? '自主规划模式：PlanAgent 自主选择工具，ObserveAgent 负责执行。' : event.planMode === 'guided' ? '辅助规划模式：控制器校验必要证据与工具依赖。' : (event.message || '');
    if (mode) mode.textContent = label;
    return;
  }
  if (event.type === 'status' && /IntentAgent|意图/.test(`${event.title || ''}${event.message || ''}`)) {
    const card = document.getElementById('agentIntentResult');
    const state = document.getElementById('agentIntentState');
    if (card) {
      card.className = 'intent-agent-card pending compact';
      card.innerHTML = `<div class="intent-agent-body"><p>${escapeHtml(event.message || '正在解析用户意图')}</p></div>`;
    }
    if (state) state.textContent = '识别中';
    return;
  }
  if (event.type === 'synthesis') {
    const decision = document.getElementById('agentPlanDecision');
    if (decision) decision.textContent = `${stateLabel(event.state)}，候选轨迹 ${Number(event.trackCount || 0)} 条。`;
  }
}

function renderIntentAgentCard(event) {
  const card = document.getElementById('agentIntentResult');
  const state = document.getElementById('agentIntentState');
  if (!card) return;
  const timeScope = event.queryScope ? `${formatMonitorTime(event.queryScope[0])}—${formatMonitorTime(event.queryScope[1])}` : '全部监控时间';
  const rules = Array.isArray(event.selectedRules) && event.selectedRules.length ? event.selectedRules.join(' / ') : '模型判定';
  const targetText = event.hullNumber || event.description || targetKindLabel(event.targetKind);
  const confidence = Number(event.intentConfidence);
  const confidenceText = Number.isFinite(confidence) ? confidence.toFixed(2) : '—';
  const route = `${scopeLabel(event.targetScope)} · ${operationLabel(event.operation)} · ${relationLabel(event.registryRelation)}`;
  const acceptance = event.successCriteria || event.expectedOutcome || '按查询目标返回可核验证据';
  card.className = 'intent-agent-card ready compact';
  card.innerHTML = `
    <div class="intent-agent-summary compact">
      <p><strong>目标：</strong>${escapeHtml(String(targetText))}</p>
      <p><strong>路径：</strong>${escapeHtml(route)}</p>
      <p><strong>时间：</strong>${escapeHtml(timeScope)}</p>
      <p><strong>验收：</strong>${escapeHtml(String(acceptance))}</p>
    </div>
    <div class="intent-agent-chips compact">
      <span>规则 ${escapeHtml(rules)}</span>
      <span>来源 ${escapeHtml(intentSourceLabel(event.intentSource))}</span>
      <span>置信度 ${escapeHtml(confidenceText)}</span>
    </div>`;
  if (state) state.textContent = '已识别';
  initializePlanBlueprint(event.planBlueprint);
}

function updateObserverToolEvent(event) {
  const card = ensureAgentCard(event.round, 'observer');
  if (!card) return;
  const logs = card._toolLogs || [];
  const index = logs.findIndex((item) => item.id === event.id);
  const label = event.tool || event.id || '工具';
  const status = event.phase === 'running' ? '执行中' : event.phase === 'skipped' || event.skipped ? '跳过' : event.ok === false || event.phase === 'failed' ? '失败' : '完成';
  const detail = event.error ? `：${compactAgentValue(event.error, 70)}` : event.summary ? `：${compactAgentValue(event.summary, 90)}` : '';
  const item = {id: event.id, text: `${label} · ${status}${detail}`};
  if (index >= 0) logs[index] = item; else logs.push(item);
  card._toolLogs = logs.slice(-6);
  const content = card.querySelector('.agent-stream-text');
  if (content) content.textContent = card._toolLogs.map((entry) => entry.text).join('\n');
  updatePlanProgressFromTool(event);
}

function appendThoughtEvent(event) {
  if (event.type === 'complete') {
    setThinkingState('推理完成', 'completed');
    const decision = document.getElementById('agentPlanDecision');
    if (decision && !decision.classList.contains('sufficient')) {
      decision.className = 'agent-plan-decision sufficient';
      decision.textContent = '闭环推理完成，最终回答与证据已生成。';
    }
    return;
  }
  if (event.type === 'error') {
    setThinkingState('推理失败', 'failed');
    const decision = document.getElementById('agentPlanDecision');
    if (decision) {
      decision.className = 'agent-plan-decision conflict';
      decision.textContent = `执行失败：${event.message || '未知错误'}`;
    }
    return;
  }
  if (event.type === 'agent_start') {
    if (event.role === 'planner') prepareAgentRound(event.round);
    const card = ensureAgentCard(event.round, event.role);
    if (!card) return;
    card.classList.add('active');
    card.dataset.streamText = '';
    if (event.role === 'observer') card._toolLogs = [];
    const state = card.querySelector('.agent-thought-head em');
    if (state) state.textContent = event.role === 'observer' ? '执行中' : event.role === 'reflector' ? '验收中' : '规划中';
    const pendingText = event.role === 'planner'
      ? '正在生成本轮工具规划…'
      : event.role === 'observer'
        ? '正在执行工具并整理结果…'
        : '正在对照验收目标检查证据…';
    card.querySelector('.agent-stream-text').innerHTML = `${escapeHtml(pendingText)}<span class="agent-stream-cursor"></span>`;
    if (event.role === 'planner') {
      const round = document.getElementById('agentPlanRound');
      const decision = document.getElementById('agentPlanDecision');
      if (round) round.textContent = `第 ${event.round || 1} 轮规划中`;
      if (decision) {
        decision.className = 'agent-plan-decision active';
        decision.textContent = '正在根据当前验收缺口生成计划。';
      }
    }
  } else if (event.type === 'agent_delta') {
    const card = ensureAgentCard(event.round, event.role);
    if (!card || !event.delta) return;
    card.dataset.streamText = `${card.dataset.streamText || ''}${event.delta}`.slice(-520);
    const content = card.querySelector('.agent-stream-text');
    content.textContent = card.dataset.streamText;
    const cursor = document.createElement('span');
    cursor.className = 'agent-stream-cursor';
    content.appendChild(cursor);
  } else if (event.type === 'agent_tool') {
    updateObserverToolEvent(event);
  } else if (event.type === 'agent_end') {
    const card = ensureAgentCard(event.round, event.role);
    if (!card) return;
    if (event.role === 'planner') syncPlanProgress(event);
    if (event.role === 'observer') updatePlanFromObservation(event);
    if (event.role === 'reflector') updatePlanReflection(event);
    const content = card.querySelector('.agent-stream-text');
    const summaryText = compactAgentText(event) || '';
    const isPlanCard = event.role === 'planner' || event.planOnly;
    const planNote = isPlanCard ? (event.planRepair || event.fallback || '') : '';
    content.textContent = summaryText || planNote || event.fallback || '本轮没有可展示信息';
    card.classList.remove('active');
    const hasFailedCall = event.role === 'observer' && (event.calls || []).some((call) => call.skipped || call.ok === false);
    const markFailed = isPlanCard ? false : Boolean(event.fallback || hasFailedCall);
    card.classList.toggle('failed', markFailed);
    let statusLabel = '完成';
    if (isPlanCard && event.planRepair) statusLabel = '已修正';
    else if (!isPlanCard && event.fallback) statusLabel = '摘要失败';
    else if (hasFailedCall) statusLabel = '部分失败';
    else if (event.role === 'reflector') statusLabel = stateLabel(event.state);
    const state = card.querySelector('.agent-thought-head em');
    if (state) state.textContent = statusLabel;
    card.querySelector('.agent-thought-tags').innerHTML = agentTags(event);
  } else {
    appendSystemThought(event);
  }
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
  const clearButton = document.getElementById('agentMemoryClearButton');
  if (!question) return showToast('请输入问题', 'error');
  try {
    button.disabled = true;
    if (clearButton) clearButton.disabled = true;
    button.textContent = '推理中…';
    resetThoughtStream();
    document.getElementById('agentAnswer').className = 'agent-answer empty-state';
    document.getElementById('agentAnswer').textContent = '正在基于轨迹记忆生成回答…';
    renderEvidence(null, null, 'Collecting evidence...');
    const result = await streamAgentQuery(question);
    renderAgentAnswer(result);
    renderEvidence(result.evidence, result.displayGroups, 'No evidence available', result.registryItems || [], result);
  } catch (error) {
    showAgentFinalView();
    setThinkingState('推理失败', 'failed');
    document.getElementById('agentAnswer').className = 'agent-answer empty-state';
    document.getElementById('agentAnswer').textContent = `执行失败：${error.message}`;
    renderEvidence(null, null);
    showToast(error.message, 'error');
  } finally {
    button.disabled = false;
    button.textContent = '执行闭环推理';
    if (clearButton) clearButton.disabled = false;
    await loadAgentMemorySummary(false);
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
window.clearAgentMemory = clearAgentMemory;
