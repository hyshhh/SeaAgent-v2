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
  return ({
    model: '规则表+模型',
    heuristic: '规则兜底',
    langgraph_react: 'LangGraph/ReAct',
    langgraph_fallback: 'LangGraph 兜底',
  })[source] || valueText(source);
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

function initializeTopKControl(defaultValue = 3) {
  const enabled = document.getElementById('agentTopKEnabled');
  const input = document.getElementById('agentTopKValue');
  if (!enabled || !input || input.dataset.initialized === 'true') return;
  const savedEnabled = localStorage.getItem('seaagent-topk-enabled');
  const savedValue = Number(localStorage.getItem('seaagent-topk-value'));
  enabled.checked = savedEnabled == null ? true : savedEnabled === 'true';
  input.value = String(Number.isFinite(savedValue) && savedValue >= 1 && savedValue <= 20 ? savedValue : defaultValue);
  input.disabled = !enabled.checked;
  enabled.closest('.agent-topk-control')?.classList.toggle('disabled', !enabled.checked);
  input.dataset.initialized = 'true';
}

function bindTopKControl() {
  const enabled = document.getElementById('agentTopKEnabled');
  const input = document.getElementById('agentTopKValue');
  if (!enabled || !input) return;
  enabled.addEventListener('change', () => {
    input.disabled = !enabled.checked;
    enabled.closest('.agent-topk-control')?.classList.toggle('disabled', !enabled.checked);
    localStorage.setItem('seaagent-topk-enabled', String(enabled.checked));
  });
  input.addEventListener('change', () => {
    input.value = String(Math.max(1, Math.min(20, Number(input.value) || 3)));
    localStorage.setItem('seaagent-topk-value', input.value);
  });
}

function selectedTopK() {
  const enabled = document.getElementById('agentTopKEnabled');
  const input = document.getElementById('agentTopKValue');
  if (!enabled?.checked) return null;
  return Math.max(1, Math.min(20, Number(input?.value) || 3));
}

const evidenceDisplayDefaults = Object.freeze({videoScale: 0.25, imageScale: 0.5, maxItems: 0});
let latestEvidencePayload = null;

function clampEvidenceValue(value, minimum, maximum, fallback) {
  const number = Number(value);
  return Number.isFinite(number) ? Math.max(minimum, Math.min(maximum, number)) : fallback;
}

function evidenceDisplaySettings() {
  const videoScale = clampEvidenceValue(localStorage.getItem('seaagent-evidence-video-scale'), 0.25, 1, evidenceDisplayDefaults.videoScale);
  const imageScale = clampEvidenceValue(localStorage.getItem('seaagent-evidence-image-scale'), 0.25, 1, evidenceDisplayDefaults.imageScale);
  const maxItems = Math.round(clampEvidenceValue(localStorage.getItem('seaagent-evidence-max-items'), 0, 10000, evidenceDisplayDefaults.maxItems));
  return {videoScale, imageScale, maxItems};
}

function updateEvidenceDisplayLabels() {
  const video = document.getElementById('evidenceVideoScale');
  const image = document.getElementById('evidenceImageScale');
  const maximum = document.getElementById('evidenceMaxItems');
  const videoLabel = document.getElementById('evidenceVideoScaleValue');
  const imageLabel = document.getElementById('evidenceImageScaleValue');
  const maximumLabel = document.getElementById('evidenceMaxItemsValue');
  if (videoLabel && video) videoLabel.textContent = `${Math.round(Number(video.value) * 100)}%`;
  if (imageLabel && image) imageLabel.textContent = `${Math.round(Number(image.value) * 100)}%`;
  if (maximumLabel && maximum) maximumLabel.textContent = Number(maximum.value) > 0 ? `${maximum.value} 条` : '全部';
}

function renderLatestEvidence() {
  if (!latestEvidencePayload) return;
  renderEvidence(
    latestEvidencePayload.evidence,
    latestEvidencePayload.displayGroups,
    latestEvidencePayload.emptyText,
    latestEvidencePayload.registryItems,
    latestEvidencePayload.result,
  );
}

function saveEvidenceDisplaySettings() {
  const video = document.getElementById('evidenceVideoScale');
  const image = document.getElementById('evidenceImageScale');
  const maximum = document.getElementById('evidenceMaxItems');
  if (!video || !image || !maximum) return;
  video.value = String(clampEvidenceValue(video.value, 0.25, 1, evidenceDisplayDefaults.videoScale));
  image.value = String(clampEvidenceValue(image.value, 0.25, 1, evidenceDisplayDefaults.imageScale));
  maximum.value = String(Math.round(clampEvidenceValue(maximum.value, 0, 10000, evidenceDisplayDefaults.maxItems)));
  localStorage.setItem('seaagent-evidence-video-scale', video.value);
  localStorage.setItem('seaagent-evidence-image-scale', image.value);
  localStorage.setItem('seaagent-evidence-max-items', maximum.value);
  updateEvidenceDisplayLabels();
  renderLatestEvidence();
}

function initializeEvidenceDisplayControls() {
  const video = document.getElementById('evidenceVideoScale');
  const image = document.getElementById('evidenceImageScale');
  const maximum = document.getElementById('evidenceMaxItems');
  if (!video || !image || !maximum || video.dataset.initialized === 'true') return;
  const settings = evidenceDisplaySettings();
  video.value = String(settings.videoScale);
  image.value = String(settings.imageScale);
  maximum.value = String(settings.maxItems);
  video.dataset.initialized = 'true';
  updateEvidenceDisplayLabels();
}

function bindEvidenceDisplayControls() {
  const video = document.getElementById('evidenceVideoScale');
  const image = document.getElementById('evidenceImageScale');
  const maximum = document.getElementById('evidenceMaxItems');
  if (!video || !image || !maximum || video.dataset.bound === 'true') return;
  [video, image].forEach((input) => input.addEventListener('input', updateEvidenceDisplayLabels));
  [video, image, maximum].forEach((input) => input.addEventListener('change', saveEvidenceDisplaySettings));
  maximum.addEventListener('input', updateEvidenceDisplayLabels);
  video.dataset.bound = 'true';
}

function evidenceSimilarity(value) {
  const score = Number(value?.embeddingScore ?? value?.score ?? value?.matchScore);
  return Number.isFinite(score) ? score : null;
}

function sortEvidenceGroups(groups) {
  return (groups || []).map((group, index) => ({group, index, score: evidenceSimilarity(group)})).sort((left, right) => {
    if (left.score !== null && right.score !== null) return right.score - left.score || left.index - right.index;
    if (left.score !== null) return -1;
    if (right.score !== null) return 1;
    return left.index - right.index;
  }).map((item) => item.group);
}

function limitEvidenceItems(items) {
  const maximum = evidenceDisplaySettings().maxItems;
  return maximum > 0 ? items.slice(0, maximum) : items;
}

function evidenceCountText(shown, total, unit) {
  return shown === total ? `${total} ${unit}` : `显示 ${shown} / ${total} ${unit}`;
}

async function loadAgentMemorySummary(showNotice = false) {
  const refreshButton = document.getElementById('agentMemoryRefreshButton');
  if (refreshButton) {
    refreshButton.disabled = true;
    refreshButton.textContent = '刷新中…';
  }
  try {
    const summary = await apiFetch('/api/agent/memory-summary');
    initializeTopKControl(summary.retrievalTopK || 3);
    const limit = document.getElementById('agentRoundLimit');
    if (limit) {
      const maxRounds = summary.maxRounds || 3;
      limit.textContent = `最多 ${maxRounds} 轮`;
    }
    if (showNotice && typeof showToast === 'function') showToast('记忆状态已刷新');
  } catch (error) {
    if (showNotice && typeof showToast === 'function') showToast(`记忆状态刷新失败：${error.message || '服务不可用'}`, 'error');
  } finally {
    if (refreshButton) {
      refreshButton.disabled = false;
      refreshButton.textContent = '刷新';
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

function resultScoreBand(item) {
  return String(item?.scoreBand || item?.verifyDecision || '').trim().toLowerCase();
}

function bestMatchByResultId(matches, kind) {
  const best = new Map();
  (matches || []).forEach((match) => {
    const id = kind === 'registry'
      ? match?.matchedRegistryId || match?.registryId
      : match?.matchedTrackId || match?.trackId;
    if (id == null) return;
    const key = String(id);
    const score = Number(match?.embeddingScore ?? match?.score);
    const previous = best.get(key);
    const previousScore = Number(previous?.embeddingScore ?? previous?.score);
    if (!previous || !Number.isFinite(previousScore) || (Number.isFinite(score) && score > previousScore)) best.set(key, match);
  });
  return best;
}

function classifiedResultItems(result) {
  const matches = Array.isArray(result?.matches) ? result.matches : [];
  const registryBest = bestMatchByResultId(matches, 'registry');
  const trackBest = bestMatchByResultId(matches, 'track');
  const registryItems = (Array.isArray(result?.registryItems) ? result.registryItems : []).map((item) => {
    const match = registryBest.get(String(item?.registryId || item?.matchedRegistryId || ''));
    return match ? {...item, embeddingScore: match.embeddingScore, scoreBand: match.scoreBand || item.scoreBand} : item;
  });
  const tracks = (Array.isArray(result?.tracks) ? result.tracks : []).map((item) => {
    const match = trackBest.get(String(item?.trackId || item?.matchedTrackId || ''));
    return match ? {...item, embeddingScore: match.embeddingScore, scoreBand: match.scoreBand || item.scoreBand} : item;
  });
  const registryClassifiable = registryItems.some((item) => ['match', 'uncertain'].includes(resultScoreBand(item)));
  const trackClassifiable = tracks.some((item) => ['match', 'uncertain'].includes(resultScoreBand(item)));
  const items = registryClassifiable ? registryItems : trackClassifiable ? tracks : [];
  if (!items.length) return null;
  return {
    kind: registryClassifiable ? 'registry' : 'track',
    confirmed: items.filter((item) => resultScoreBand(item) === 'match'),
    pending: items.filter((item) => resultScoreBand(item) === 'uncertain'),
  };
}

function classifiedResultRow(item, kind) {
  const score = Number(item?.embeddingScore ?? item?.score);
  if (kind === 'registry') {
    const hull = item?.hullNumber || item?.hull || '未知舷号';
    const registryId = item?.registryId || item?.matchedRegistryId || '未知库项';
    const description = item?.description || '';
    return `<div class="classified-result-row"><strong>${escapeHtml(hull)}</strong><span>库项 ${escapeHtml(String(registryId))}${Number.isFinite(score) ? ` · 相似度 ${score.toFixed(3)}` : ''}${description ? ` · ${escapeHtml(description)}` : ''}</span></div>`;
  }
  const trackId = item?.trackId || item?.matchedTrackId || '未知轨迹';
  const hull = item?.finalHullNumber || item?.hullNumber || '无稳定舷号';
  const start = Number(item?.startTime ?? item?.start_time);
  const end = Number(item?.endTime ?? item?.end_time);
  const time = Number.isFinite(start) && Number.isFinite(end) ? `${formatMonitorTime(start)}—${formatMonitorTime(end)}` : '时间未知';
  return `<div class="classified-result-row"><strong>轨迹 ${escapeHtml(String(trackId))}</strong><span>${escapeHtml(hull)} · ${escapeHtml(time)}${Number.isFinite(score) ? ` · 相似度 ${score.toFixed(3)}` : ''}</span></div>`;
}

function renderClassifiedResults(result) {
  const grouped = classifiedResultItems(result);
  if (!grouped) return '';
  const confirmedRows = grouped.confirmed.map((item) => classifiedResultRow(item, grouped.kind)).join('');
  const pendingRows = grouped.pending.map((item) => classifiedResultRow(item, grouped.kind)).join('');
  return `<section class="answer-classification">
    <div class="answer-result-groups">
      <section class="answer-result-group confirmed"><header><div><strong>确定结果</strong><span>达到确认阈值</span></div><em>${grouped.confirmed.length}</em></header><div class="answer-result-list">${confirmedRows || '<div class="answer-result-empty">暂无确定结果</div>'}</div></section>
      <section class="answer-result-group pending"><header><div><strong>待确认</strong><span>灰区匹配，建议人工复核</span></div><em>${grouped.pending.length}</em></header><div class="answer-result-list">${pendingRows || '<div class="answer-result-empty">暂无待确认结果</div>'}</div></section>
    </div>
  </section>`;
}

function dedupTrackIds(group) {
  return (group?.trackIds || group?.mergedTrackIds || []).map((item) => String(item));
}

function dedupResultRow(group, kind) {
  const trackIds = dedupTrackIds(group);
  const score = Number(group?.minimumScore);
  const title = trackIds.length ? `轨迹 ${trackIds.join(' + ')}` : '轨迹组';
  let detail = kind === 'confirmed'
    ? '高阈值确认属于同一艘船'
    : '低阈值候选合并，计入最低统计口径';
  if (kind === 'pending' && Array.isArray(group?.currentGroups) && group.currentGroups.length) {
    const current = group.currentGroups.map((item) => `[${(item || []).join(' + ')}]`).join(' ↔ ');
    detail += ` · 当前分组 ${current}`;
  }
  if (Number.isFinite(score)) detail += ` · 最低组内相似度 ${score.toFixed(3)}`;
  return `<div class="classified-result-row dedup-result-row"><strong>${escapeHtml(title)}</strong><span>${escapeHtml(detail)}</span></div>`;
}

function renderDedupResults(result) {
  const summary = result?.dedupSummary;
  if (!summary || result?.operation !== 'count') return '';
  const confirmed = Array.isArray(summary.confirmedMergeGroups) ? summary.confirmedMergeGroups : [];
  const pending = Array.isArray(summary.pendingMergeGroups) ? summary.pendingMergeGroups : [];
  const minimum = Number(summary.minimumShipCount ?? result.minimumCount ?? result.count);
  const confirmedCount = Number(summary.confirmedShipCount ?? result.confirmedCount ?? minimum);
  const confirmedRows = confirmed.map((item) => dedupResultRow(item, 'confirmed')).join('');
  const pendingRows = pending.map((item) => dedupResultRow(item, 'pending')).join('');
  const range = Number.isFinite(minimum) && Number.isFinite(confirmedCount) && minimum !== confirmedCount
    ? `<span>统计区间 <b>${minimum}—${confirmedCount}</b> 艘</span>`
    : `<span>稳定统计 <b>${Number.isFinite(minimum) ? minimum : 0}</b> 艘</span>`;
  return `<section class="answer-classification dedup-classification">
    <div class="dedup-count-strip"><strong>最低统计 ${Number.isFinite(minimum) ? minimum : 0} 艘</strong>${range}<em>确认口径 ${Number.isFinite(confirmedCount) ? confirmedCount : 0} 艘</em></div>
    <div class="answer-result-groups">
      <section class="answer-result-group confirmed"><header><div><strong>确认合并的轨迹</strong><span>高阈值通过，同组轨迹确定归为一艘船</span></div><em>${confirmed.length} 组</em></header><div class="answer-result-list">${confirmedRows || '<div class="answer-result-empty">暂无需要确认合并的重复轨迹</div>'}</div></section>
      <section class="answer-result-group pending"><header><div><strong>待确认合并的轨迹</strong><span>影响最低船数，需要结合关键帧复核</span></div><em>${pending.length} 组</em></header><div class="answer-result-list">${pendingRows || '<div class="answer-result-empty">暂无待确认合并关系</div>'}</div></section>
    </div>
  </section>`;
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
  const records = Array.isArray(result.toolRecords) && result.toolRecords.length
    ? result.toolRecords
    : (result.toolChain || []).map((item, index) => ({round: index + 1, legacy: item}));
  const chain = records.map((item) => `<span class="tool-tag">${escapeHtml(formatToolCall(item.round, item))}</span>`).join('');
  const dedupResults = renderDedupResults(result);
  const classified = renderClassifiedResults(result);
  const fallbackResults = `${renderRegistryHits(result)}${renderTracks(result.tracks, result.questionType)}`;
  const rawCount = Number(result.count);
  const rawMatchCount = Number(result.matchCount);
  const hitCount = Number.isFinite(rawCount) ? rawCount : Number.isFinite(rawMatchCount) ? rawMatchCount : Number((result.tracks || []).length);
  const hitLabel = result?.dedupSummary ? '最低船数' : '命中数量';
  document.getElementById('agentAnswer').className = 'agent-answer';
  document.getElementById('agentAnswer').innerHTML = `
    <div class="answer-overview">
      <div class="answer-head"><strong>${escapeHtml(result.conclusion || '问答完成')}</strong><span class="status-tag ${result.uncertainty === 'sufficient' ? 'ok' : 'off'}">${escapeHtml(stateLabel(result.uncertainty))}</span></div>
      <p>${escapeHtml(result.answerText || '未生成回答')}</p>
      <div class="answer-meta"><span>问题类型：${escapeHtml(questionTypeLabel(result.questionType))}</span><span>查询范围：${escapeHtml(scope)}</span><span>${hitLabel}：${hitCount}</span></div>
    </div>
    <div class="answer-results-scroll">${dedupResults || classified || fallbackResults}</div>
    ${chain ? `<section class="answer-tool-chain"><header><strong>工具调用链</strong><span>${records.length} 条记录</span></header><div class="tool-tags">${chain}</div></section>` : ''}`;
}

function evidenceItem(type, id, trackId = null, options = {}) {
  const prefix = type === 'video' ? 'Clip' : type === 'keyframe' ? 'Keyframe' : 'Database Reference';
  const route = type === 'video' ? 'clips' : type === 'keyframe' ? 'keyframes' : 'registry';
  const owner = trackId == null ? '' : `Track ${trackId} · `;
  const settings = evidenceDisplaySettings();
  const videoQuery = new URLSearchParams();
  if (Array.isArray(options.timeRange)) {
    videoQuery.set('startTime', String(options.timeRange[0]));
    videoQuery.set('endTime', String(options.timeRange[1]));
  }
  if (type === 'video' && options.trackClip) videoQuery.set('scale', String(settings.videoScale));
  const videoScope = videoQuery.size ? `?${videoQuery.toString()}` : '';
  const imageScope = `?scale=${encodeURIComponent(settings.imageScale)}`;
  const url = type === 'video' && options.trackClip
    ? `/api/evidence/tracks/${encodeURIComponent(id)}/clip${videoScope}`
    : `/api/evidence/${route}/${encodeURIComponent(id)}${type === 'video' ? '' : imageScope}`;
  const posterUrl = type !== 'video' ? null : options.posterKeyframeId
    ? `/api/evidence/keyframes/${encodeURIComponent(options.posterKeyframeId)}${imageScope}`
    : `/api/evidence/clips/${encodeURIComponent(id)}/poster${imageScope}`;
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

function evidenceTrackCell(label, item) {
  return `<section class="evidence-track-cell" role="cell" aria-label="${escapeHtml(label)}" data-label="${escapeHtml(label)}">${evidenceCard(item)}</section>`;
}

function evidenceTrackTableHead() {
  return '<div class="evidence-track-table-head" role="row"><span role="columnheader">轨迹信息</span><span role="columnheader">Clip</span><span role="columnheader">Keyframe</span><span role="columnheader">Registry</span></div>';
}

function evidenceTrackRow(group, index) {
  const trackId = group.trackId ?? index + 1;
  const score = evidenceSimilarity(group);
  const clip = group.clipTrackId
    ? evidenceItem('video', group.clipTrackId, trackId, {trackClip: true, posterKeyframeId: group.keyframeIds?.[0], timeRange: group.clipTimeRange})
    : group.shipSegmentIds?.[0]
      ? evidenceItem('video', group.shipSegmentIds[0], trackId)
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
  const badges = `${hull ? `<em>舷号 ${escapeHtml(hull)}</em>` : '<em>舷号未知</em>'}${score === null ? '' : `<em>相似度 ${score.toFixed(3)}</em>`}`;
  return `<article class="evidence-track-row" role="row"><div class="evidence-track-meta" role="rowheader"><span>轨迹</span><strong>${escapeHtml(trackId)}</strong><div class="evidence-track-badges">${badges}</div></div>${evidenceTrackCell('Clip', clip)}${evidenceTrackCell('Keyframe', keyframe)}${evidenceTrackCell('Registry', database)}</article>`;
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

function dedupEvidenceGroup(group, index) {
  const pending = group?.groupType === 'pending';
  const trackIds = (group?.mergedTrackIds || []).map((item) => String(item));
  const members = Array.isArray(group?.memberEvidence) && group.memberEvidence.length
    ? group.memberEvidence
    : trackIds.map((trackId, memberIndex) => ({trackId, keyframeId: group?.keyframeIds?.[memberIndex]}));
  const score = Number(group?.minimumScore);
  const cards = members.map((member) => {
    const trackId = String(member?.trackId ?? '-');
    const item = member?.keyframeId
      ? evidenceItem('keyframe', member.keyframeId, trackId, {label: `轨迹 ${trackId} · 合并判定关键帧`})
      : missingEvidence(trackId, 'keyframe');
    return evidenceCard(item);
  }).join('');
  const current = pending && Array.isArray(group?.currentGroups) && group.currentGroups.length
    ? `<span>当前分组：${escapeHtml(group.currentGroups.map((item) => `[${(item || []).join(' + ')}]`).join(' ↔ '))}</span>`
    : '';
  const scoreText = Number.isFinite(score) ? `<em>最低组内相似度 ${score.toFixed(3)}</em>` : '';
  return `<article class="dedup-evidence-group ${pending ? 'pending' : 'confirmed'}">
    <header><div><span>${pending ? '待确认合并组' : '确认合并组'} ${index + 1}</span><strong>轨迹 ${escapeHtml(trackIds.join(' + ') || '-')}</strong>${current}</div>${scoreText}</header>
    <div class="dedup-evidence-members">${cards || '<div class="evidence-track-empty">该组暂无可展示关键帧</div>'}</div>
  </article>`;
}

function renderDedupEvidence(container, resultCount, groups, emptyText) {
  const confirmedAll = groups.filter((group) => group?.groupType === 'confirmed');
  const pendingAll = groups.filter((group) => group?.groupType === 'pending');
  const confirmed = limitEvidenceItems(confirmedAll);
  const pending = limitEvidenceItems(pendingAll);
  if (resultCount) {
    resultCount.textContent = `${confirmedAll.length} 组确认合并 · ${pendingAll.length} 组待确认`;
  }
  const section = (kind, title, subtitle, rows, total) => `<section class="dedup-evidence-section ${kind}">
    <header><div><strong>${title}</strong><span>${subtitle}</span></div><em>${rows.length === total ? total : `${rows.length}/${total}`} 组</em></header>
    <div class="dedup-evidence-section-body">${rows.length ? rows.map(dedupEvidenceGroup).join('') : `<div class="dedup-evidence-empty">${escapeHtml(emptyText)}</div>`}</div>
  </section>`;
  container.className = 'evidence-dedup-list';
  container.innerHTML = `<div class="dedup-evidence-sections">
    ${section('confirmed', '确认合并证据', '关键帧达到高阈值，确定归并为同一艘船', confirmed, confirmedAll.length)}
    ${section('pending', '待确认合并证据', '灰区关键帧决定最低船数，需要人工复核', pending, pendingAll.length)}
  </div>`;
}

function renderEvidence(evidence, displayGroups, emptyText = 'No evidence available', registryItems = [], result = null) {
  latestEvidencePayload = {evidence, displayGroups, emptyText, registryItems, result};
  const container = document.getElementById('evidenceGallery');
  const resultCount = document.getElementById('evidenceResultCount');
  evidenceVideoObserver?.disconnect();
  visibleEvidenceVideos.forEach((video) => video.pause());
  visibleEvidenceVideos.clear();
  const groups = sortEvidenceGroups(displayGroups || []);
  if (result?.operation === 'count' && result?.dedupSummary) {
    renderDedupEvidence(container, resultCount, groups, '暂无该类轨迹合并关系');
    return;
  }
  const registryOnly = result?.targetScope === 'registry';
  if (registryOnly) {
    let database = representativeRegistryEvidence(registryItems);
    if (!database.length) {
      database = [...new Set(evidence?.registryReferenceIds || [])].map((id) => evidenceItem('registry', id));
    }
    const visibleDatabase = limitEvidenceItems(database);
    if (resultCount) resultCount.textContent = evidenceCountText(visibleDatabase.length, database.length, '个库项');
    container.className = 'evidence-columns registry-only';
    container.innerHTML = evidenceColumn('Database Evidence', 'Registry Reference Images', visibleDatabase, emptyText);
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

  const totalRows = rows.length;
  rows = limitEvidenceItems(rows);
  if (resultCount) resultCount.textContent = totalRows ? evidenceCountText(rows.length, totalRows, '条结果轨迹') : '无匹配轨迹';
  container.className = 'evidence-track-list';
  container.innerHTML = rows.length
    ? `<div class="evidence-track-table" role="table" aria-label="多模态轨迹证据">${evidenceTrackTableHead()}<div class="evidence-track-table-body" role="rowgroup">${rows.map((group, index) => evidenceTrackRow(group, index)).join('')}</div></div>`
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
function normalizedSkillReads(event) {
  const reads = Array.isArray(event?.skillReads) ? event.skillReads : [];
  if (reads.length) return reads.filter((item) => item && item.skillId);
  return (Array.isArray(event?.enabledSkills) ? event.enabledSkills : [])
    .filter(Boolean)
    .map((skillId) => ({skillId, title: skillId, source: 'auto', ok: true}));
}

function compactSkillReads(event) {
  const reads = normalizedSkillReads(event);
  if (!reads.length) return '';
  const failed = reads.filter((item) => item.ok === false).length;
  return `Read ${reads.length} skills${failed ? ` · ${failed} failed` : ''}`;
}

function skillReadTags(event) {
  const reads = normalizedSkillReads(event);
  if (!reads.length) return '';
  const failed = reads.filter((item) => item.ok === false).length;
  return `<span class="skill-read${failed ? ' failed' : ''}">Skills ${reads.length}${failed ? ` · ${failed} failed` : ''}</span>`;
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
    if (event.fallback) lines.push(`说明：${compactAgentValue(event.fallback, 120)}`);
    return lines.join('\n') || '本轮未生成可执行计划';
  }
  if (event.role === 'intent') {
    const model = event.modelSummary || {};
    return [
      model.goal ? `移交：${compactAgentValue(model.goal, 120)}` : '',
      model.reason ? `依据：${compactAgentValue(model.reason, 140)}` : '',
      summary ? compactAgentValue(summary, 200) : '',
    ].filter(Boolean).join('\n') || '意图识别完成';
  }
  if (event.role === 'observer') {
    return summary ? `观察：${compactAgentValue(summary, 260)}` : calls.length ? `本轮已完成 ${calls.length} 个工具步骤` : '本轮没有工具结果';
  }
  if (event.role === 'reflector') {
    const progress = event.acceptanceProgress || {};
    const requirements = Array.isArray(progress.requirements) ? progress.requirements : [];
    const completedCount = requirements.filter((item) => item && item.completed).length;
    const progressText = requirements.length ? `${completedCount}/${requirements.length}` : '';
    const sourceLabel = event.decisionSource === 'model'
      ? '模型判定'
      : event.decisionSource === 'acceptance_guard'
        ? '验收规则纠偏'
        : event.decisionSource === 'deterministic_fallback'
          ? '验收规则接管'
          : '';
    return [
      event.acceptanceGoal ? `验收标准：${compactAgentValue(event.acceptanceGoal, 140)}` : '',
      event.currentFocus ? `当前焦点：${compactAgentValue(event.currentFocus, 120)}` : '',
      progressText ? `验收进度：${progressText}${progress.acceptanceSatisfied ? '（已满足）' : '（未满足）'}` : '',
      `权威判断：${stateLabel(event.state)}${sourceLabel ? ` · ${sourceLabel}` : ''}`,
      event.evidenceGap ? `关键缺口：${compactAgentValue(event.evidenceGap, 130)}` : '',
      event.nextAction ? `下一轮动作：${compactAgentValue(event.nextAction, 150)}` : '',
      event.nextRound ? `流转：进入第 ${event.nextRound} 轮 PlanAgent` : '',
      summary ? `判定依据：${compactAgentValue(summary, 180)}` : '',
    ].filter(Boolean).join('\n');
  }
  return summary;
}

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

function rolePlaceholder(role, forRound = false) {
  if (role === 'intent') return forRound ? '等待意图识别…' : '等待意图识别…';
  if (role === 'planner') return forRound ? '等待本轮规划…' : '等待规划…';
  if (role === 'observer') return forRound ? '等待本轮执行…' : '等待执行…';
  if (role === 'reflector') return forRound ? '等待本轮验收…' : '等待验收…';
  return '等待中…';
}

function roleRunningLabel(role) {
  return ({
    intent: '识别中',
    planner: '规划中',
    observer: '执行中',
    reflector: '验收中',
  })[role] || '运行中';
}

function rolePendingText(role) {
  return ({
    intent: 'Parsing the user question and acceptance target…',
    planner: 'Drafting the tool plan for this round…',
    observer: 'Running tools and collecting evidence…',
    reflector: 'Checking evidence against the acceptance target…',
  })[role] || '处理中…';
}

function scrollThoughtStreamToCard(card) {
  if (!card || !card.closest('#agentThoughtStream')) return;
  try {
    card.scrollIntoView({block: 'nearest', behavior: 'smooth'});
  } catch (_) {
    /* ignore */
  }
}

function setAgentStreamText(card, text, {cursor = false, html = false} = {}) {
  const content = card?.querySelector('.agent-stream-text');
  if (!content) return;
  if (html) {
    content.innerHTML = text + (cursor ? '<span class="agent-stream-cursor"></span>' : '');
    return;
  }
  content.textContent = text || '';
  if (cursor) {
    const node = document.createElement('span');
    node.className = 'agent-stream-cursor';
    content.appendChild(node);
  }
  content.scrollTop = content.scrollHeight;
}

function resetAgentCards() {
  activeAgentRound = 0;
  document.querySelectorAll('.agent-workspace .agent-thought-card[data-role]').forEach((card) => {
    const role = card.dataset.role;
    card.dataset.round = '0';
    card.dataset.streamText = '';
    card.classList.remove('active', 'failed');
    card._toolLogs = [];
    card._skillReads = [];
    const round = card.querySelector('.agent-card-round');
    const state = card.querySelector('.agent-thought-head em');
    const tags = card.querySelector('.agent-thought-tags');
    if (round) round.textContent = role === 'intent' ? '全局' : '等待轮次';
    if (state) state.textContent = '等待';
    setAgentStreamText(card, '', {html: true});
    const content = card.querySelector('.agent-stream-text');
    if (content) content.innerHTML = `<span class="agent-stream-placeholder">${escapeHtml(rolePlaceholder(role))}</span>`;
    if (tags) tags.innerHTML = '';
  });
}

function prepareAgentRound(roundNumber) {
  const normalizedRound = Number(roundNumber || 1);
  if (!Number.isFinite(normalizedRound) || normalizedRound <= 0) return;
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
    const tags = card.querySelector('.agent-thought-tags');
    if (round) round.textContent = `第 ${normalizedRound} 轮`;
    if (state) state.textContent = '等待';
    const content = card.querySelector('.agent-stream-text');
    if (content) content.innerHTML = `<span class="agent-stream-placeholder">${escapeHtml(rolePlaceholder(role, true))}</span>`;
    if (tags) tags.innerHTML = '';
  });
}

function resetThoughtStream() {
  showAgentProcessView();
  resetAgentCards();
  resetPlanProgress();
  const intent = ensureAgentCard(0, 'intent');
  const intentState = document.getElementById('agentIntentState');
  const mode = document.getElementById('agentPlanMode');
  if (intent) {
    intent.classList.add('active');
    const cardState = intent.querySelector('.agent-thought-head em');
    if (cardState) cardState.textContent = '识别中';
    setAgentStreamText(intent, 'Parsing the question and acceptance target…', {cursor: true});
  }
  if (intentState) intentState.textContent = '识别中';
  if (mode) mode.textContent = 'Plan → Observe → Verify · Reflect controls the next round';
  setThinkingState('正在推理', 'active');
}

function ensureAgentCard(roundNumber, role) {
  if (role === 'intent') return document.getElementById('agentIntentResult');
  if (Number(roundNumber) > 0) prepareAgentRound(roundNumber);
  return document.querySelector(`#agentThoughtStream .agent-thought-card[data-role="${role}"]`);
}

function agentTags(event) {
  const skills = skillReadTags(event);
  if (event.role === 'intent') {
    const tags = [];
    if (event.targetKind) tags.push(`<span>${escapeHtml(targetKindLabel(event.targetKind))}</span>`);
    if (event.operation) tags.push(`<span>${escapeHtml(operationLabel(event.operation))}</span>`);
    if (event.intentSource) tags.push(`<span>${escapeHtml(intentSourceLabel(event.intentSource))}</span>`);
    if (event.hullNumber) tags.push(`<span>舷号 ${escapeHtml(event.hullNumber)}</span>`);
    if (event.description) tags.push(`<span>${escapeHtml(compactAgentValue(event.description, 24))}</span>`);
    return tags.join('') + skills;
  }
  if (event.role === 'planner') {
    const toolCount = (event.calls || []).length;
    const tools = toolCount ? `<span>Tool plan ${toolCount}</span>` : '';
    const repair = event.planRepair ? `<span class="failed" title="${escapeHtml(event.planRepair)}">计划已纠正</span>` : '';
    return skills + tools + repair;
  }
  if (event.role === 'observer') {
    const calls = event.calls || [];
    const failed = calls.filter((call) => !call.skipped && call.ok === false).length;
    return skills + (calls.length ? `<span class="${failed ? 'failed' : ''}">Tools ${calls.length}${failed ? ` · ${failed} failed` : ''}</span>` : '');
  }
  if (event.role === 'reflector') {
    return `${skills}<span>${escapeHtml(stateLabel(event.state))}</span>${event.evidenceGap ? `<span>缺口：${escapeHtml(event.evidenceGap)}</span>` : ''}`;
  }
  return '';
}

function appendSystemThought(event) {
  if (event.type === 'classification') {
    renderIntentAgentCard(event);
    return;
  }
  if (event.type === 'status' && (event.planMode || /规划模式|PlanAgent|LangGraph|Controller/.test(`${event.title || ''}${event.message || ''}`))) {
    const mode = document.getElementById('agentPlanMode');
    if (mode) mode.textContent = 'Plan → Observe → Verify · Reflect controls the next round';
    return;
  }
  if (event.type === 'status' && (/IntentAgent|意图/.test(`${event.title || ''}${event.message || ''}`) || event.role === 'intent')) {
    const state = document.getElementById('agentIntentState');
    const card = ensureAgentCard(0, 'intent');
    if (state) state.textContent = '识别中';
    if (card) {
      card.classList.add('active');
      card.classList.remove('failed');
      const head = card.querySelector('.agent-thought-head em');
      if (head) head.textContent = '识别中';
      setAgentStreamText(card, event.message || '正在解析用户意图…', {cursor: true});
    }
    setThinkingState('意图识别中', 'active');
    return;
  }
  if (event.type === 'status' && event.role) {
    const card = ensureAgentCard(event.round, event.role);
    if (card && event.message) {
      if (!card.classList.contains('active')) card.classList.add('active');
      const head = card.querySelector('.agent-thought-head em');
      if (head && head.textContent === '等待') head.textContent = roleRunningLabel(event.role);
      // 仅在仍是占位或空内容时写入状态，避免覆盖思考/工具日志
      const content = card.querySelector('.agent-stream-text');
      const isPlaceholder = content?.querySelector('.agent-stream-placeholder');
      if (isPlaceholder || !(card.dataset.streamText || '').trim()) {
        setAgentStreamText(card, event.message, {cursor: true});
      }
    }
    return;
  }
  if (event.type === 'synthesis') {
    const decision = document.getElementById('agentPlanDecision');
    if (decision) decision.textContent = `${stateLabel(event.state)}，候选轨迹 ${Number(event.trackCount || 0)} 条。`;
    setThinkingState('生成最终回答', 'active');
  }
}

function renderIntentAgentCard(event) {
  const card = ensureAgentCard(0, 'intent');
  const state = document.getElementById('agentIntentState');
  const timeScope = event.timeParseError ? `解析失败：${event.timeParseError}` : event.queryScope ? `${formatMonitorTime(event.queryScope[0])}—${formatMonitorTime(event.queryScope[1])}` : '全部监控时间';
  const targetItems = Array.isArray(event.targetItems) ? event.targetItems.filter((item) => item && item.label) : [];
  const targetText = targetItems.length > 1
    ? targetItems.map((item) => item.label).join('、')
    : event.hullNumber || event.description || targetKindLabel(event.targetKind);
  const route = `${scopeLabel(event.targetScope)} · ${operationLabel(event.operation)} · ${relationLabel(event.registryRelation)}`;
  const acceptance = event.successCriteria || event.expectedOutcome || '按查询目标返回可核验证据';
  const focus = event.nextAgentFocus || '根据当前验收缺口规划第一轮工具调用';
  if (card) {
    const summary = [
      `目标：${targetText || '—'}`,
      `路径：${route}`,
      `时间：${timeScope}`,
      `验收标准：${acceptance}`,
      `当前焦点：${focus}`,
    ].join('\n');
    card.classList.remove('active');
    card.classList.toggle('failed', Boolean(event.timeParseError));
    card.dataset.streamText = summary;
    const head = card.querySelector('.agent-thought-head em');
    if (head) head.textContent = event.timeParseError ? '需确认' : '完成';
    setAgentStreamText(card, summary);
    const tags = card.querySelector('.agent-thought-tags');
    if (tags) tags.innerHTML = agentTags({role: 'intent', ...event});
  }
  if (state) state.textContent = event.timeParseError ? '需确认' : '已识别';
  initializePlanBlueprint(event.planBlueprint);
  setThinkingState('规划中', 'active');
}

function toolResultText(call) {
  if (call.phase === 'running') return 'running';
  if (call.phase === 'skipped' || call.skipped) return `skipped: ${compactAgentValue(call.skipReason || 'condition not met', 70)}`;
  if (call.error || call.phase === 'failed' || call.ok === false) return `failed: ${compactAgentValue(call.error || 'tool failed', 70)}`;
  const summary = call.summary && typeof call.summary === 'object' ? call.summary : call;
  if (summary.trackCount != null) return `${summary.trackCount} tracks`;
  if (summary.keyframeCount != null) return `${summary.keyframeCount} keyframes`;
  if (summary.matchCount != null) return `${summary.matchCount} matches`;
  if (summary.registryItemCount != null) return `${summary.registryItemCount} registry items`;
  if (summary.registryCount != null) return `${summary.registryCount} registry items`;
  if (summary.registryReferenceCount != null) return `${summary.registryReferenceCount} references`;
  if (summary.exactMatchHullCount != null) return `${summary.exactMatchHullCount} exact hull matches`;
  if (summary.totalTrackCount != null) return `${summary.returnedTrackCount ?? summary.totalTrackCount} / ${summary.totalTrackCount} tracks`;
  if (Array.isArray(summary.trackIds)) return `${summary.trackIds.length} track ids`;
  if (Array.isArray(summary.keyframeIds)) return `${summary.keyframeIds.length} keyframe ids`;
  if (Array.isArray(summary.matchedHullNumbers)) return `${summary.matchedHullNumbers.length} hull hits`;
  if (summary.highThresholdShipCount != null) return `low ${summary.lowThresholdShipCount} · confirmed ${summary.highThresholdShipCount}`;
  if (summary.shipSegmentId) return `segment ${summary.shipSegmentId}`;
  if (Array.isArray(summary.registryReferenceIds)) return `${summary.registryReferenceIds.length} references`;
  if (summary.found === true) return 'found';
  if (summary.found === false) return 'not found';
  if (summary.decision) return String(summary.decision);
  if (call.summary != null && typeof call.summary !== 'object') return compactAgentValue(call.summary, 90);
  return 'done';
}

function formatToolArgumentValue(value) {
  if (value == null) return 'null';
  if (typeof value === 'string') return value;
  if (typeof value === 'number' || typeof value === 'boolean') return String(value);
  if (Array.isArray(value)) {
    const visible = value.slice(0, 10).map((item) => formatToolArgumentValue(item));
    return `[${visible.join(', ')}${value.length > visible.length ? `, …+${value.length - visible.length}` : ''}]`;
  }
  if (typeof value === 'object') {
    const fields = Object.entries(value).slice(0, 6)
      .map(([key, item]) => `${key}:${formatToolArgumentValue(item)}`);
    return `{${fields.join(', ')}${Object.keys(value).length > fields.length ? ', …' : ''}}`;
  }
  return String(value);
}

function formatToolArguments(argumentsValue) {
  if (!argumentsValue || typeof argumentsValue !== 'object' || Array.isArray(argumentsValue)) return 'no args';
  const entries = Object.entries(argumentsValue);
  if (!entries.length) return 'no args';
  return entries.map(([key, value]) => `${key}=${formatToolArgumentValue(value)}`).join(', ');
}

function formatToolCall(round, call) {
  const roundNumber = Math.max(1, Number(round || call.round || 1));
  if (call.legacy) return `${roundNumber}-${call.legacy}-done`;
  const tool = call.tool || 'tool';
  return `${roundNumber} · ${tool}(${formatToolArguments(call.arguments)}) · ${toolResultText(call)}`;
}

function agentActivityOpen(card, kind) {
  return Boolean(card?._activityOpen?.[kind]);
}

function isSkillReadingActive(card) {
  const until = Number(card?.dataset?.skillsRunningUntil || 0);
  return Boolean(card?._skillsRunning) || (until > Date.now());
}

function clearSkillReading(card) {
  if (!card) return;
  if (card._skillReadingTimer) {
    window.clearTimeout(card._skillReadingTimer);
    card._skillReadingTimer = null;
  }
  card._skillsRunning = false;
  card.dataset.skillsRunningUntil = '0';
}

function scheduleSkillReadingClear(card) {
  if (!card) return;
  if (card._skillReadingTimer) window.clearTimeout(card._skillReadingTimer);
  const until = Number(card.dataset.skillsRunningUntil || 0);
  if (!Number.isFinite(until) || until <= 0) return;
  const delay = Math.max(0, until - Date.now()) + 35;
  card._skillReadingTimer = window.setTimeout(() => {
    card._skillReadingTimer = null;
    const activeUntil = Number(card.dataset.skillsRunningUntil || 0);
    if (activeUntil > Date.now()) {
      scheduleSkillReadingClear(card);
      return;
    }
    card._skillsRunning = false;
    card.dataset.skillsRunningUntil = '0';
    if (!card._toolsRunning) {
      setAgentProcessStream(card, card.dataset.streamText || '', {cursor: card.classList.contains('active')});
    }
  }, delay);
}

function markSkillReading(card, durationMs = 1400) {
  if (!card || !Array.isArray(card._skillReads) || !card._skillReads.length) return;
  const nextUntil = Date.now() + Math.max(120, Number(durationMs) || 0);
  card.dataset.skillsRunningUntil = String(Math.max(Number(card.dataset.skillsRunningUntil || 0), nextUntil));
  scheduleSkillReadingClear(card);
}

function currentSkillSnapshot(card, reads) {
  const current = card?._currentSkill || {};
  const fallbackTotal = Math.max(1, reads.length);
  let index = Number(current.index || current.skillIndex || 0);
  let total = Number(current.total || current.skillTotal || fallbackTotal);
  let record = null;
  const currentId = current.skillId || current.currentSkillId;
  if (currentId) record = reads.find((item) => item.skillId === currentId) || null;
  if (!record && Number.isFinite(index) && index > 0) record = reads[index - 1] || null;
  if (!record) {
    const runningIndex = reads.findIndex((item) => item.phase === 'running');
    if (runningIndex >= 0) {
      record = reads[runningIndex];
      index = runningIndex + 1;
    }
  }
  if (!record) {
    record = reads[0] || {};
    index = Number.isFinite(index) && index > 0 ? index : 1;
  }
  if (!Number.isFinite(index) || index <= 0) index = Math.max(1, reads.indexOf(record) + 1);
  if (!Number.isFinite(total) || total <= 0) total = fallbackTotal;
  total = Math.max(total, index);
  const name = current.skillId || current.currentSkillId || record.skillId || current.title || current.currentSkillTitle || record.title || 'skill';
  const title = current.title || current.currentSkillTitle || record.title || name;
  return {index, total, name, title};
}

function renderSkillActivity(card) {
  const reads = Array.isArray(card?._skillReads) ? card._skillReads : [];
  if (!reads.length) return '';
  const running = isSkillReadingActive(card);
  const failed = reads.filter((item) => item.ok === false).length;
  const current = currentSkillSnapshot(card, reads);
  const label = running
    ? `Reading skill ${current.index}/${current.total} · ${current.name}`
    : `Read ${reads.length} skills${failed ? ` · ${failed} failed` : ''}`;
  const details = reads.map((item) => {
    const source = item.source === 'dynamic' ? 'on demand' : 'auto';
    const status = item.phase === 'running' ? 'running' : item.ok === false ? 'failed' : 'done';
    const description = item.description ? `<small>${escapeHtml(item.description)}</small>` : '';
    const itemClass = `agent-activity-item${item.ok === false ? ' failed' : ''}${item.phase === 'running' ? ' running' : ''}`;
    return `<div class="${itemClass}"><code>${escapeHtml(item.skillId || item.title || 'skill')}</code><em>${source} · ${status}</em>${description}</div>`;
  }).join('');
  return `<details class="agent-activity agent-activity-skill${running ? ' running' : ''}" data-activity-kind="skills"${agentActivityOpen(card, 'skills') ? ' open' : ''}><summary><span class="agent-activity-icon" aria-hidden="true">◇</span><span title="${escapeHtml(current.title)}">${escapeHtml(label)}</span>${running ? '<i class="agent-activity-cursor" aria-hidden="true"></i>' : ''}<b aria-hidden="true"></b></summary><div class="agent-activity-body">${details}</div></details>`;
}

function renderToolActivity(card) {
  const logs = Array.isArray(card?._toolLogs) ? card._toolLogs : [];
  if (!logs.length) return '';
  const runningIndex = logs.findIndex((item) => item.running);
  const running = runningIndex >= 0;
  const failed = logs.filter((item) => item.failed).length;
  const current = running ? logs[runningIndex] : null;
  const total = Math.max(logs.length, Number(card?._toolTotal || 0) || 0, runningIndex + 1);
  const label = running
    ? `Running tool ${runningIndex + 1}/${Math.max(1, total)} · ${(current && (current.tool || current.id)) || 'tool'}`
    : `Ran ${logs.length} tools${failed ? ` · ${failed} failed` : ''}`;
  const details = logs.map((item) => `<div class="agent-activity-item${item.failed ? ' failed' : ''}${item.running ? ' running' : ''}"><code>${escapeHtml(item.text)}</code></div>`).join('');
  return `<details class="agent-activity agent-activity-tool${running ? ' running' : ''}" data-activity-kind="tools"${agentActivityOpen(card, 'tools') ? ' open' : ''}><summary><span class="agent-activity-icon" aria-hidden="true">›_</span><span title="${escapeHtml((current && (current.tool || current.id)) || label)}">${escapeHtml(label)}</span>${running ? '<i class="agent-activity-cursor" aria-hidden="true"></i>' : ''}<b aria-hidden="true"></b></summary><div class="agent-activity-body">${details}</div></details>`;
}

function bindAgentActivityState(card) {
  card?.querySelectorAll('details.agent-activity').forEach((details) => {
    details.addEventListener('toggle', () => {
      card._activityOpen = {...(card._activityOpen || {}), [details.dataset.activityKind]: details.open};
    });
  });
}

function setAgentProcessStream(card, text, {cursor = false} = {}) {
  if (!card) return;
  const holdForSkillReading = isSkillReadingActive(card) && !card._toolsRunning;
  const activity = [renderSkillActivity(card), renderToolActivity(card)].filter(Boolean).join('');
  const body = text && !holdForSkillReading ? `<div class="agent-stream-copy">${escapeHtml(text)}</div>` : '';
  setAgentStreamText(card, `${activity}${body}`, {cursor: cursor && !holdForSkillReading, html: true});
  bindAgentActivityState(card);
}

function updateObserverToolEvent(event) {
  const card = ensureAgentCard(event.round, 'observer');
  if (!card) return;
  const logs = card._toolLogs || [];
  const eventId = String(event.id || event.tool || `tool-${logs.length + 1}`);
  const index = logs.findIndex((item) => item.id === eventId);
  const running = event.phase === 'running' || event.status === 'running';
  const toolName = event.tool || event.message || eventId;
  const item = {
    id: eventId,
    tool: toolName,
    text: formatToolCall(event.round, {...event, id: eventId, tool: toolName}),
    running,
    phase: event.phase || event.status || (running ? 'running' : 'completed'),
    failed: !running && event.ok === false && !event.skipped,
  };
  if (index >= 0) logs[index] = item; else logs.push(item);
  const runningIndex = logs.findIndex((entry) => entry.running);
  card._toolLogs = logs;
  card._currentTool = runningIndex >= 0
    ? {id: logs[runningIndex].id, tool: logs[runningIndex].tool, index: runningIndex + 1, total: logs.length}
    : null;
  card._skillsRunning = false;
  card._toolsRunning = runningIndex >= 0;
  card.classList.add('active');
  const state = card.querySelector('.agent-thought-head em');
  if (state) state.textContent = running ? '执行中' : roleRunningLabel('observer');
  const streamText = card.dataset.streamText || 'Running tools and collecting evidence…';
  setAgentProcessStream(card, streamText, {cursor: running});
  scrollThoughtStreamToCard(card);
  updatePlanProgressFromTool(event);
}

function compactAgentErrorMessage(raw) {
  const text = String(raw || '未知错误').replace(/\s+/g, ' ').trim();
  if (/GRAPH_RECURSION_LIMIT|Recursion limit/i.test(text)) {
    return '规划步骤过多已自动收敛，请重试或简化问题';
  }
  if (/allowed-local-media-path|Cannot load local files/i.test(text)) {
    return '视觉模型拒绝本地媒体路径，请确认服务端已用 data URL 传图';
  }
  if (/chat\/completions|Internal Server Error|HTTPStatusError/i.test(text)) {
    return '视觉/语言模型服务暂时不可用，请稍后重试';
  }
  return text.length > 180 ? `${text.slice(0, 180)}…` : text;
}

function appendThoughtEvent(event) {
  // 推理全程保持过程视图；complete 仅更新状态，最终结果由 askAgent 在拿到 result 后切换
  if (event.type === 'complete') {
    setThinkingState('推理完成', 'completed');
    const decision = document.getElementById('agentPlanDecision');
    if (decision && !decision.classList.contains('sufficient')) {
      decision.className = 'agent-plan-decision sufficient';
      decision.textContent = '闭环推理完成，最终回答与证据已生成。';
    }
    document.querySelectorAll('.agent-thought-card.active').forEach((card) => {
      card.classList.remove('active');
      const state = card.querySelector('.agent-thought-head em');
      if (state && ['识别中', '规划中', '执行中', '验收中', '运行中'].includes(state.textContent)) {
        state.textContent = '完成';
      }
    });
    return;
  }
  if (event.type === 'error') {
    setThinkingState('推理失败', 'failed');
    const decision = document.getElementById('agentPlanDecision');
    if (decision) {
      decision.className = 'agent-plan-decision conflict';
      decision.textContent = `执行失败：${compactAgentErrorMessage(event.message)}`;
    }
    document.querySelectorAll('.agent-thought-card.active').forEach((card) => {
      card.classList.remove('active');
      card.classList.add('failed');
      const state = card.querySelector('.agent-thought-head em');
      if (state) state.textContent = '中断';
    });
    return;
  }
  if (event.type === 'agent_start') {
    if (event.role === 'planner') prepareAgentRound(event.round);
    const card = ensureAgentCard(event.round, event.role);
    if (!card) return;
    card.classList.add('active');
    card.classList.remove('failed');
    card.dataset.streamText = '';
    card.dataset.streamThinking = '';
    card.dataset.streamToken = '';
    card.dataset.hasAgentDelta = '';
    clearSkillReading(card);
    if (event.role === 'observer') {
      card._toolLogs = [];
      card._currentTool = null;
      card._toolTotal = 0;
    }
    card._skillReads = normalizedSkillReads(event);
    card._currentSkill = card._skillReads.length
      ? {skillId: card._skillReads[0].skillId, title: card._skillReads[0].title, index: 1, total: card._skillReads.length}
      : null;
    card._skillsRunning = card._skillReads.length > 0;
    markSkillReading(card, 1600);
    card._toolsRunning = false;
    const tags = card.querySelector('.agent-thought-tags');
    if (tags) tags.innerHTML = skillReadTags({skillReads: card._skillReads});
    const state = card.querySelector('.agent-thought-head em');
    if (state) state.textContent = roleRunningLabel(event.role);
    card.dataset.streamText = rolePendingText(event.role);
    setAgentProcessStream(card, card.dataset.streamText, {cursor: true});
    scrollThoughtStreamToCard(card);
    if (event.role === 'intent') {
      setThinkingState('意图识别中', 'active');
    } else if (event.role === 'planner') {
      setThinkingState(`第 ${event.round || 1} 轮规划中`, 'active');
      const round = document.getElementById('agentPlanRound');
      const decision = document.getElementById('agentPlanDecision');
      if (round) round.textContent = `第 ${event.round || 1} 轮规划中`;
      if (decision) {
        decision.className = 'agent-plan-decision active';
        decision.textContent = '正在根据当前验收缺口生成计划。';
      }
    } else if (event.role === 'observer') {
      setThinkingState(`第 ${event.round || 1} 轮执行中`, 'active');
    } else if (event.role === 'reflector') {
      setThinkingState(`第 ${event.round || 1} 轮验收中`, 'active');
    }
  } else if (event.type === 'agent_skill') {
    const card = ensureAgentCard(event.round, event.role);
    if (!card) return;
    const reads = card._skillReads || [];
    const key = `${event.skillId || ''}:${event.source || 'auto'}`;
    const index = reads.findIndex((item) => `${item.skillId || ''}:${item.source || 'auto'}` === key);
    const eventIndex = Number(event.skillIndex || 0);
    const eventTotal = Number(event.skillTotal || reads.length || 1);
    const record = {
      skillId: event.skillId || event.currentSkillId,
      title: event.currentSkillTitle || event.title || event.skillId,
      description: event.description || '',
      source: event.source || 'auto',
      ok: event.ok !== false,
      phase: event.phase || 'completed',
      skillIndex: eventIndex,
      skillTotal: eventTotal,
    };
    if (index >= 0) reads[index] = record; else reads.push(record);
    const resolvedIndex = Number.isFinite(eventIndex) && eventIndex > 0
      ? eventIndex
      : Math.max(1, reads.findIndex((item) => item.skillId === record.skillId && item.source === record.source) + 1);
    card._skillReads = reads;
    card._currentSkill = {
      skillId: record.skillId,
      title: record.title,
      index: resolvedIndex,
      total: Math.max(Number.isFinite(eventTotal) && eventTotal > 0 ? eventTotal : reads.length, resolvedIndex),
    };
    card._skillsRunning = event.phase === 'running';
    if (card.classList.contains('active')) {
      markSkillReading(card, event.phase === 'running' ? 1400 : 720);
    }
    const tags = card.querySelector('.agent-thought-tags');
    if (tags) tags.innerHTML = skillReadTags({skillReads: reads});
    const streamText = card.dataset.streamText || rolePendingText(event.role);
    setAgentProcessStream(card, streamText, {cursor: card.classList.contains('active')});
    scrollThoughtStreamToCard(card);
  } else if (event.type === 'agent_delta') {
    const card = ensureAgentCard(event.round, event.role);
    if (!card || !event.delta) return;
    card.classList.add('active');
    card._skillsRunning = false;
    card.dataset.hasAgentDelta = '1';
    const kind = event.kind || 'thinking';
    if (kind === 'token') {
      card.dataset.streamToken = `${card.dataset.streamToken || ''}${event.delta}`.slice(-1600);
    } else {
      card.dataset.streamThinking = `${card.dataset.streamThinking || ''}${event.delta}`.slice(-2400);
    }
    const thinking = (card.dataset.streamThinking || '').trim();
    const token = (card.dataset.streamToken || '').trim();
    const live = [thinking ? `思考：\n${thinking}` : '', token ? `草稿：\n${token}` : '']
      .filter(Boolean)
      .join('\n\n')
      .slice(-2800);
    card.dataset.streamText = live;
    setAgentProcessStream(card, live || '…', {cursor: true});
    scrollThoughtStreamToCard(card);
  } else if (event.type === 'agent_tool') {
    updateObserverToolEvent(event);
  } else if (event.type === 'agent_end') {
    const card = ensureAgentCard(event.round, event.role);
    if (!card) return;
    if (event.role === 'planner') syncPlanProgress(event);
    if (event.role === 'observer') updatePlanFromObservation(event);
    if (event.role === 'reflector') updatePlanReflection(event);
    const isPlanCard = event.role === 'planner' || event.planOnly;
    card._skillReads = normalizedSkillReads(event);
    clearSkillReading(card);
    card._currentSkill = null;
    card._toolsRunning = false;
    card._currentTool = null;
    if (event.role === 'observer' && !card._toolLogs?.length && Array.isArray(event.calls)) {
      card._toolLogs = event.calls.map((call, index) => ({
        id: call.id || `${call.tool || 'tool'}-${index + 1}`,
        tool: call.tool || call.id || 'tool',
        text: formatToolCall(event.round, call),
        running: false,
        phase: call.phase || call.status || 'completed',
        failed: !call.skipped && call.ok === false,
      }));
    } else if (card._toolLogs?.length) {
      card._toolLogs = card._toolLogs.map((item) => ({...item, running: false, phase: item.phase || 'completed'}));
    }
    const summaryText = compactAgentText(event) || '';
    const thinkingTrail = event.role === 'reflector' ? '' : String(
      event.thinking
      || (event.modelSummary && event.modelSummary.thinking)
      || card.dataset.streamThinking
      || ''
    ).trim();
    // 结论优先用结构化摘要；fallback 仅作补充说明，避免盖掉真实计划
    const conclusion = summaryText || event.message || event.fallback || '本轮没有可展示信息';
    const finalText = thinkingTrail
      ? `思考过程：\n${compactAgentValue(thinkingTrail, 900)}\n\n结论：\n${conclusion}`
      : conclusion;
    setAgentProcessStream(card, finalText);
    card.dataset.streamText = finalText;
    card.dataset.streamThinking = '';
    card.dataset.streamToken = '';
    card.classList.remove('active');
    const calls = event.calls || [];
    const hasHardFail = event.role === 'observer' && calls.some((call) => !call.skipped && call.ok === false);
    const onlySkipped = event.role === 'observer'
      && calls.length > 0
      && calls.every((call) => call.skipped || call.ok !== false)
      && calls.some((call) => call.skipped)
      && !hasHardFail;
    const markFailed = isPlanCard ? false : Boolean((!isPlanCard && event.fallback) || hasHardFail);
    card.classList.toggle('failed', markFailed);
    let statusLabel = '完成';
    if (isPlanCard && event.planRepair) statusLabel = '已修正';
    else if (isPlanCard && event.fallback && String(event.fallback).includes('安全回退')) statusLabel = '安全回退';
    else if (isPlanCard && event.fallback) statusLabel = '已用默认计划';
    else if (!isPlanCard && event.fallback) statusLabel = '摘要失败';
    else if (hasHardFail) statusLabel = '部分失败';
    else if (onlySkipped) statusLabel = '部分跳过';
    else if (event.role === 'reflector') statusLabel = stateLabel(event.state);
    else if (event.role === 'intent') statusLabel = '完成';
    const state = card.querySelector('.agent-thought-head em');
    if (state) state.textContent = statusLabel;
    const tags = card.querySelector('.agent-thought-tags');
    if (tags) tags.innerHTML = agentTags(event);
    scrollThoughtStreamToCard(card);
  } else {
    appendSystemThought(event);
  }
}

async function streamAgentQuery(question, topK = null) {
  const response = await fetch('/api/agent/query/stream', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({question, top_k: topK}),
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
      let event;
      try {
        event = JSON.parse(line);
      } catch (_) {
        continue;
      }
      appendThoughtEvent(event);
      if (event.type === 'complete') result = event.result;
      if (event.type === 'error') throw new Error(event.message || '闭环推理失败');
    }
    if (done) break;
  }
  if (buffer.trim()) {
    let event;
    try {
      event = JSON.parse(buffer);
    } catch (_) {
      event = null;
    }
    if (event) {
      appendThoughtEvent(event);
      if (event.type === 'complete') result = event.result;
      if (event.type === 'error') throw new Error(event.message || '闭环推理失败');
    }
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
    // 只展示过程流，最终回答区延后到 complete
    resetThoughtStream();
    const answer = document.getElementById('agentAnswer');
    if (answer) {
      answer.className = 'agent-answer empty-state';
      answer.textContent = '推理完成后在此展示最终回答…';
    }
    renderEvidence(null, null, '推理中，证据将在完成后展示…');
    const result = await streamAgentQuery(question, selectedTopK());
    // 过程流结束后再切换最终结果
    renderAgentAnswer(result);
    renderEvidence(result.evidence, result.displayGroups, 'No evidence available', result.registryItems || [], result);
  } catch (error) {
    showAgentFinalView();
    setThinkingState('推理失败', 'failed');
    const answer = document.getElementById('agentAnswer');
    if (answer) {
      answer.className = 'agent-answer empty-state';
      answer.textContent = `执行失败：${compactAgentErrorMessage(error.message)}`;
    }
    renderEvidence(null, null);
    showToast(compactAgentErrorMessage(error.message), 'error');
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
  bindTopKControl();
  initializeEvidenceDisplayControls();
  bindEvidenceDisplayControls();
});

window.useQuestion = useQuestion;
window.askAgent = askAgent;
window.loadAgentMemorySummary = loadAgentMemorySummary;
window.clearAgentMemory = clearAgentMemory;
