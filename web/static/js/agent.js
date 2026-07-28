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
  return ({sufficient: 'Evidence Sufficient', replan: 'Replan', conflict: 'Evidence Conflict', uncertain: 'Uncertain'})[state] || valueText(state);
}


function scopeLabel(scope) {
  return ({track_memory: 'Trajectory Memory', registry: 'Registry', both: 'Cross-Memory Comparison'})[scope] || valueText(scope);
}

function targetKindLabel(kind) {
  return ({hull: 'Hull Number Target', description: 'Appearance Target', all: 'All Targets'})[kind] || valueText(kind);
}

function operationLabel(op) {
  return ({existence: 'Existence Check', list: 'List Query', time: 'Time Localization', count: 'Count Query', explain: 'Evidence Explanation'})[op] || valueText(op);
}

function relationLabel(rel) {
  return ({any: 'Any Registry Relation', in: 'In Registry', out: 'Out of Registry'})[rel] || valueText(rel);
}

function intentSourceLabel(source) {
  return ({
    model: 'Rule Table + Model',
    heuristic: 'Heuristic Fallback',
    langgraph_react: 'LangGraph/ReAct',
    langgraph_fallback: 'LangGraph Fallback',
  })[source] || valueText(source);
}

function questionTypeLabel(type) {
  return ({
    hull: 'Hull Number Query',
    registry_hull: 'Registry Hull Query',
    description: 'Appearance Query',
    registry_description: 'Registry Appearance Query',
    cross_reference: 'Cross-Memory Query',
    track_list: 'Trajectory List Query',
    registry_list: 'Registry List Query',
    relation_description: 'Appearance and Registry Query',
    out_of_registry: 'Out-of-Registry Query',
    in_registry: 'In-Registry Query',
    count: 'Count Query',
    description_count: 'Appearance Count Query',
    registry_count: 'Registry Count Query',
    registry_description_count: 'Registry Appearance Count Query'
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
  if (maximumLabel && maximum) maximumLabel.textContent = Number(maximum.value) > 0 ? `${maximum.value} items` : 'All';
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
  return shown === total ? `${total} ${unit}` : `Showing ${shown} / ${total} ${unit}`;
}

async function loadAgentMemorySummary(showNotice = false) {
  const refreshButton = document.getElementById('agentMemoryRefreshButton');
  if (refreshButton) {
    refreshButton.disabled = true;
    refreshButton.textContent = 'Refreshing…';
  }
  try {
    const summary = await apiFetch('/api/agent/memory-summary');
    initializeTopKControl(summary.retrievalTopK || 3);
    const limit = document.getElementById('agentRoundLimit');
    if (limit) {
      const maxRounds = summary.maxRounds || 3;
      limit.textContent = `Up to ${maxRounds} rounds`;
    }
    if (showNotice && typeof showToast === 'function') showToast('Memory status refreshed');
  } catch (error) {
    if (showNotice && typeof showToast === 'function') showToast(`Memory refresh failed: ${error.message || 'service unavailable'}`, 'error');
  } finally {
    if (refreshButton) {
      refreshButton.disabled = false;
      refreshButton.textContent = 'Refresh';
    }
  }
}

async function clearAgentMemory() {
  const askButton = document.getElementById('btnAskAgent');
  const clearButton = document.getElementById('agentMemoryClearButton');
  if (askButton?.disabled) return showToast('Reasoning is in progress. Wait for the current run to finish before clearing memory.', 'error');
  if (!window.confirm('Clear all QA sessions, reasoning rounds, and QA evidence? Trajectory memory and registry data will be retained.')) return;
  try {
    if (clearButton) {
      clearButton.disabled = true;
      clearButton.textContent = 'Clearing…';
    }
    const result = await apiFetch('/api/agent/memory', {method: 'DELETE'});
    resetThoughtStream();
    const answer = document.getElementById('agentAnswer');
    if (answer) {
      answer.className = 'agent-answer empty-state';
      answer.textContent = 'Query memory cleared. You can start a new query.';
    }
    renderEvidence(null, null, 'No evidence available');
    await loadAgentMemorySummary(false);
    showToast(result.message || 'QA memory cleared');
  } catch (error) {
    showToast(`Failed to clear QA memory: ${error.message || 'service unavailable'}`, 'error');
  } finally {
    if (clearButton) {
      clearButton.disabled = false;
      clearButton.textContent = 'Clear Memory';
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
    const hull = track.finalHullNumber || track.hullNumber || 'No stable hull number';
    const start = Number(track.startTime ?? track.start_time);
    const end = Number(track.endTime ?? track.end_time);
    const time = Number.isFinite(start) && Number.isFinite(end) ? `${formatMonitorTime(start)}—${formatMonitorTime(end)}` : 'Unknown time';
    const score = Number(track.embeddingScore);
    return `<div class="track-summary"><strong>${escapeHtml(track.trackId || 'Unknown Track')}</strong><span>${escapeHtml(hull)} · ${time}${Number.isFinite(score) ? ` · Similarity ${score.toFixed(3)}` : ''}</span></div>`;
  }).join('');
  const more = (!rangeList && tracks.length > 3)
    ? `<div class="track-summary-more">Showing the top 3 point matches; ${tracks.length - 3} more are available in Evidence.</div>`
    : (rangeList ? `<div class="track-summary-more">${tracks.length} range matches</div>` : '');
  return `<div class="answer-tracks ${rangeList ? 'range-list' : ''}">${rows}${more}</div>`;
}


function renderRegistryHits(result) {
  const items = result?.registryItems || result?.registryMatches || [];
  if (!items.length) return '';
  const rows = items.slice(0, 5).map((item, index) => {
    const hull = item.hullNumber || item.hull || 'Unknown Hull';
    const registryId = item.registryId || item.matchedRegistryId || 'Unknown Registry Item';
    const score = Number(item.embeddingScore);
    const band = item.scoreBand || item.verifyDecision || '';
    const desc = item.description || '';
    return `<div class="track-summary"><strong>${escapeHtml(hull)}</strong><span>Registry ${escapeHtml(String(registryId))}${Number.isFinite(score) ? ` · Similarity ${score.toFixed(3)}` : ''}${band ? ` · ${escapeHtml(band)}` : ''}${desc ? ` · ${escapeHtml(desc)}` : ''}</span></div>`;
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
    const hull = item?.hullNumber || item?.hull || 'Unknown Hull';
    const registryId = item?.registryId || item?.matchedRegistryId || 'Unknown Registry Item';
    const description = item?.description || '';
    return `<div class="classified-result-row"><strong>${escapeHtml(hull)}</strong><span>Registry ${escapeHtml(String(registryId))}${Number.isFinite(score) ? ` · Similarity ${score.toFixed(3)}` : ''}${description ? ` · ${escapeHtml(description)}` : ''}</span></div>`;
  }
  const trackId = item?.trackId || item?.matchedTrackId || 'Unknown Track';
  const hull = item?.finalHullNumber || item?.hullNumber || 'No stable hull number';
  const start = Number(item?.startTime ?? item?.start_time);
  const end = Number(item?.endTime ?? item?.end_time);
  const time = Number.isFinite(start) && Number.isFinite(end) ? `${formatMonitorTime(start)}—${formatMonitorTime(end)}` : 'Unknown time';
  return `<div class="classified-result-row"><strong>Track ${escapeHtml(String(trackId))}</strong><span>${escapeHtml(hull)} · ${escapeHtml(time)}${Number.isFinite(score) ? ` · Similarity ${score.toFixed(3)}` : ''}</span></div>`;
}

function renderClassifiedResults(result) {
  const grouped = classifiedResultItems(result);
  if (!grouped) return '';
  const confirmedRows = grouped.confirmed.map((item) => classifiedResultRow(item, grouped.kind)).join('');
  const pendingRows = grouped.pending.map((item) => classifiedResultRow(item, grouped.kind)).join('');
  return `<section class="answer-classification">
    <div class="answer-result-groups">
      <section class="answer-result-group confirmed"><header><div><strong>Confirmed Results</strong><span>Above the confirmation threshold</span></div><em>${grouped.confirmed.length}</em></header><div class="answer-result-list">${confirmedRows || '<div class="answer-result-empty">No confirmed results</div>'}</div></section>
      <section class="answer-result-group pending"><header><div><strong>Pending Review</strong><span>Gray-zone matches requiring review</span></div><em>${grouped.pending.length}</em></header><div class="answer-result-list">${pendingRows || '<div class="answer-result-empty">No pending results</div>'}</div></section>
    </div>
  </section>`;
}

function dedupTrackIds(group) {
  return (group?.trackIds || group?.mergedTrackIds || []).map((item) => String(item));
}

function dedupResultRow(group, kind) {
  const trackIds = dedupTrackIds(group);
  const score = Number(group?.minimumScore);
  const title = trackIds.length ? `Tracks ${trackIds.join(' + ')}` : 'Track Group';
  let detail = kind === 'confirmed'
    ? 'Confirmed as the same vessel above the high threshold'
    : 'Candidate merge above the low threshold; included in the minimum-count estimate';
  if (kind === 'pending' && Array.isArray(group?.currentGroups) && group.currentGroups.length) {
    const current = group.currentGroups.map((item) => `[${(item || []).join(' + ')}]`).join(' ↔ ');
    detail += ` · Current groups ${current}`;
  }
  if (Number.isFinite(score)) detail += ` · Minimum within-group similarity ${score.toFixed(3)}`;
  return `<div class="classified-result-row dedup-result-row"><strong>${escapeHtml(title)}</strong><span>${escapeHtml(detail)}</span></div>`;
}

function renderDedupResults(result) {
  const summary = result?.dedupSummary;
  if (!summary || result?.operation !== 'count') return '';
  const confirmed = Array.isArray(summary.confirmedMergeGroups) ? summary.confirmedMergeGroups : [];
  const pending = Array.isArray(summary.pendingMergeGroups) ? summary.pendingMergeGroups : [];
  const confirmedRows = confirmed.map((item) => dedupResultRow(item, 'confirmed')).join('');
  const pendingRows = pending.map((item) => dedupResultRow(item, 'pending')).join('');
  return `<section class="answer-classification dedup-classification">
    <div class="answer-result-groups">
      <section class="answer-result-group confirmed"><header><div><strong>Confirmed Merge Groups</strong><span>High-threshold groups representing the same vessel</span></div><em>${confirmed.length} groups</em></header><div class="answer-result-list">${confirmedRows || '<div class="answer-result-empty">No confirmed duplicate trajectories</div>'}</div></section>
      <section class="answer-result-group pending"><header><div><strong>Pending Merge Groups</strong><span>Requires keyframe review before minimum-count confirmation</span></div><em>${pending.length} groups</em></header><div class="answer-result-list">${pendingRows || '<div class="answer-result-empty">No pending merge groups</div>'}</div></section>
    </div>
  </section>`;
}

function setAgentResultState(label, state = 'running') {
  const panel = document.getElementById('agentResultPanel');
  const badge = document.getElementById('agentIntentState');
  if (panel) panel.dataset.state = state;
  if (badge) badge.textContent = label;
}

function setAgentInitSummary(label) {
  const summary = document.getElementById('agentInitSummary');
  if (summary) summary.textContent = label;
}

function showAgentProcessView() {
  const processView = document.getElementById('agentProcessView');
  const finalView = document.getElementById('agentFinalView');
  const initDisclosure = document.getElementById('agentInitDisclosure');
  const planDisclosure = document.getElementById('agentPlanDisclosure');
  const toolDisclosure = document.getElementById('agentToolDisclosure');
  const toolSummary = document.getElementById('agentToolSummary');
  const toolBody = document.getElementById('agentResultTools');
  const hint = document.getElementById('agentResultHint');
  if (processView) processView.hidden = false;
  if (finalView) finalView.hidden = true;
  if (initDisclosure) initDisclosure.open = true;
  if (planDisclosure) planDisclosure.open = true;
  if (toolDisclosure) toolDisclosure.open = false;
  setAgentInitSummary('Initializing');
  if (toolSummary) toolSummary.textContent = 'Waiting for completion';
  if (toolBody) toolBody.innerHTML = '<div class="agent-result-empty">Tool records will appear here after execution.</div>';
  if (hint) hint.textContent = 'Init and plan updates appear here in real time';
  setAgentResultState('Initializing', 'running');
}

function showAgentFinalView(state = 'completed') {
  const processView = document.getElementById('agentProcessView');
  const finalView = document.getElementById('agentFinalView');
  const initDisclosure = document.getElementById('agentInitDisclosure');
  const planDisclosure = document.getElementById('agentPlanDisclosure');
  const toolDisclosure = document.getElementById('agentToolDisclosure');
  const round = document.getElementById('agentPlanRound');
  const hint = document.getElementById('agentResultHint');
  const toolSummary = document.getElementById('agentToolSummary');
  const toolBody = document.getElementById('agentResultTools');
  const total = agentPlanItems.length;
  const completed = agentPlanItems.filter((item) => ['completed', 'skipped'].includes(item.status)).length;
  if (processView) processView.hidden = false;
  if (finalView) finalView.hidden = false;
  if (initDisclosure) initDisclosure.open = false;
  if (planDisclosure) planDisclosure.open = false;
  if (toolDisclosure) toolDisclosure.open = false;
  if (round) round.textContent = total
    ? `${completed}/${total} steps · ${state === 'failed' ? 'Interrupted' : 'Completed'}`
    : state === 'failed' ? 'Plan interrupted' : 'Plan completed';
  if (hint) hint.textContent = state === 'failed'
    ? 'Init, plan, and tools are collapsed; expand them to inspect the failure.'
    : 'Init, plan, and tools are collapsed for a concise final result.';
  if (state === 'failed') {
    if (toolSummary) toolSummary.textContent = 'Incomplete calls';
    if (toolBody) toolBody.innerHTML = '<div class="agent-result-empty">Execution stopped before a complete tool summary was generated.</div>';
  }
  setAgentResultState(state === 'failed' ? 'Failed' : 'Completed', state);
}

function renderAgentAnswer(result) {
  showAgentFinalView('completed');
  const scope = Array.isArray(result.queryScope) ? `${formatMonitorTime(result.queryScope[0])}—${formatMonitorTime(result.queryScope[1])}` : 'All monitoring time';
  const records = Array.isArray(result.toolRecords) && result.toolRecords.length
    ? result.toolRecords
    : (result.toolChain || []).map((item, index) => ({round: index + 1, legacy: item}));
  const chain = records.map((item) => `<div class="answer-tool-item"><code>${escapeHtml(formatToolCall(item.round, item))}</code></div>`).join('');
  const toolSummary = document.getElementById('agentToolSummary');
  const toolBody = document.getElementById('agentResultTools');
  if (toolSummary) toolSummary.textContent = records.length ? `${records.length} call records` : 'No call records';
  if (toolBody) toolBody.innerHTML = chain || '<div class="agent-result-empty">No tool calls were produced.</div>';
  const dedupResults = renderDedupResults(result);
  const classified = renderClassifiedResults(result);
  const fallbackResults = `${renderRegistryHits(result)}${renderTracks(result.tracks, result.questionType)}`;
  const rawCount = Number(result.count);
  const rawMatchCount = Number(result.matchCount);
  const hitCount = Number.isFinite(rawCount) ? rawCount : Number.isFinite(rawMatchCount) ? rawMatchCount : Number((result.tracks || []).length);
  const hitLabel = result?.dedupSummary ? 'Minimum Vessel Count' : 'Matches';
  document.getElementById('agentAnswer').className = 'agent-answer';
  document.getElementById('agentAnswer').innerHTML = `
    <div class="answer-overview">
      <div class="answer-head"><strong>${escapeHtml(result.conclusion || 'Query Completed')}</strong><span class="status-tag ${result.uncertainty === 'sufficient' ? 'ok' : 'off'}">${escapeHtml(stateLabel(result.uncertainty))}</span></div>
      <p>${escapeHtml(result.answerText || 'No answer generated')}</p>
      <div class="answer-meta"><span>Query Type: ${escapeHtml(questionTypeLabel(result.questionType))}</span><span>Scope: ${escapeHtml(scope)}</span><span>${hitLabel}: ${hitCount}</span></div>
    </div>
    <div class="answer-results-scroll">${dedupResults || classified || fallbackResults}</div>`;
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
  return '<div class="evidence-track-table-head" role="row"><span role="columnheader">Track Evidence</span><span role="columnheader">Clip</span><span role="columnheader">Keyframe</span><span role="columnheader">Registry</span></div>';
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
  const badges = `${hull ? `<em>Hull ${escapeHtml(hull)}</em>` : '<em>Hull Unknown</em>'}${score === null ? '' : `<em>Similarity ${score.toFixed(3)}</em>`}`;
  return `<article class="evidence-track-row" role="row"><div class="evidence-track-meta" role="rowheader"><span>Track</span><strong>${escapeHtml(trackId)}</strong><div class="evidence-track-badges">${badges}</div></div>${evidenceTrackCell('Clip', clip)}${evidenceTrackCell('Keyframe', keyframe)}${evidenceTrackCell('Registry', database)}</article>`;
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
      ? evidenceItem('keyframe', member.keyframeId, trackId, {label: `Track ${trackId} · Merge Decision Keyframe`})
      : missingEvidence(trackId, 'keyframe');
    return evidenceCard(item);
  }).join('');
  const current = pending && Array.isArray(group?.currentGroups) && group.currentGroups.length
    ? `<span>Current Groups: ${escapeHtml(group.currentGroups.map((item) => `[${(item || []).join(' + ')}]`).join(' ↔ '))}</span>`
    : '';
  const scoreText = Number.isFinite(score) ? `<em>Minimum Similarity ${score.toFixed(3)}</em>` : '';
  return `<article class="dedup-evidence-group ${pending ? 'pending' : 'confirmed'}">
    <header><div><span>${pending ? 'Pending Merge Group' : 'Confirmed Merge Group'} ${index + 1}</span><strong>Tracks ${escapeHtml(trackIds.join(' + ') || '-')}</strong>${current}</div>${scoreText}</header>
    <div class="dedup-evidence-members">${cards || '<div class="evidence-track-empty">No keyframes available for this group</div>'}</div>
  </article>`;
}

function renderDedupEvidence(container, resultCount, groups, emptyText) {
  const confirmedAll = groups.filter((group) => group?.groupType === 'confirmed');
  const pendingAll = groups.filter((group) => group?.groupType === 'pending');
  const confirmed = limitEvidenceItems(confirmedAll);
  const pending = limitEvidenceItems(pendingAll);
  if (resultCount) {
    resultCount.textContent = `${confirmedAll.length} confirmed groups · ${pendingAll.length} pending groups`;
  }
  const section = (kind, title, subtitle, rows, total) => `<section class="dedup-evidence-section ${kind}">
    <header><div><strong>${title}</strong><span>${subtitle}</span></div><em>${rows.length === total ? total : `${rows.length}/${total}`} groups</em></header>
    <div class="dedup-evidence-section-body">${rows.length ? rows.map(dedupEvidenceGroup).join('') : `<div class="dedup-evidence-empty">${escapeHtml(emptyText)}</div>`}</div>
  </section>`;
  container.className = 'evidence-dedup-list';
  container.innerHTML = `<div class="dedup-evidence-sections">
    ${section('confirmed', 'Confirmed Merge Evidence', 'Keyframes exceed the high threshold and represent the same vessel', confirmed, confirmedAll.length)}
    ${section('pending', 'Pending Merge Evidence', 'Gray-zone keyframes require manual review', pending, pendingAll.length)}
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
    renderDedupEvidence(container, resultCount, groups, 'No trajectory merge relation is available.');
    return;
  }
  const registryOnly = result?.targetScope === 'registry';
  if (registryOnly) {
    let database = representativeRegistryEvidence(registryItems);
    if (!database.length) {
      database = [...new Set(evidence?.registryReferenceIds || [])].map((id) => evidenceItem('registry', id));
    }
    const visibleDatabase = limitEvidenceItems(database);
    if (resultCount) resultCount.textContent = evidenceCountText(visibleDatabase.length, database.length, 'registry items');
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
  if (resultCount) resultCount.textContent = totalRows ? evidenceCountText(rows.length, totalRows, 'result tracks') : 'No matching tracks';
  container.className = 'evidence-track-list';
  container.innerHTML = rows.length
    ? `<div class="evidence-track-table" role="table" aria-label="Multimodal trajectory evidence">${evidenceTrackTableHead()}<div class="evidence-track-table-body" role="rowgroup">${rows.map((group, index) => evidenceTrackRow(group, index)).join('')}</div></div>`
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
function appendStreamDelta(previous, incoming, maxLength) {
  const base = String(previous || '');
  const delta = String(incoming || '');
  if (!delta) return base;
  let piece = delta;
  if (base && delta.startsWith(base)) {
    piece = delta.slice(base.length);
  } else if (base && (base.includes(delta) || base.endsWith(delta))) {
    piece = '';
  } else if (base && delta.includes(base)) {
    piece = delta.slice(delta.lastIndexOf(base) + base.length);
  } else if (base) {
    const limit = Math.min(base.length, delta.length);
    for (let size = limit; size > 0; size -= 1) {
      if (base.endsWith(delta.slice(0, size))) {
        piece = delta.slice(size);
        break;
      }
    }
  }
  if (!piece) return base;
  return `${base}${piece}`.slice(-maxLength);
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
  const tool = call.tool || call.id || 'tool';
  const status = call.skipped ? 'skipped' : call.ok === false ? 'failed' : 'completed';
  const details = [];
  if (Number.isFinite(Number(call.trackCount))) details.push(`${call.trackCount} tracks`);
  if (Number.isFinite(Number(call.keyframeCount))) details.push(`${call.keyframeCount} keyframes`);
  if (Number.isFinite(Number(call.matchCount))) details.push(`${call.matchCount} matches`);
  if (Number.isFinite(Number(call.registryCount))) details.push(`${call.registryCount} registry items`);
  if (call.decision) details.push(String(call.decision));
  if (call.hasMore) details.push('more pages');
  if (call.error) details.push(compactAgentValue(call.error, 70));
  return `${tool} · ${status}${details.length ? ` · ${details.join(' · ')}` : ''}`;
}

function compactAgentText(event) {
  const summary = modelSummaryText(event.modelSummary).trim();
  const calls = Array.isArray(event.calls) ? event.calls : [];
  if (event.role === 'planner') {
    const model = event.modelSummary || {};
    const lines = [];
    if (model.goal) lines.push(`Goal: ${compactAgentValue(model.goal, 100)}`);
    if (calls.length) lines.push(`Plan: ${calls.map((call) => call.tool || 'tool').join(' → ')}`);
    if (model.reason) lines.push(`Reason: ${compactAgentValue(model.reason, 140)}`);
    if (event.planRepair) lines.push('Validation: previous plan was invalid and has been repaired');
    if (event.fallback) lines.push(`Note: ${compactAgentValue(event.fallback, 120)}`);
    return lines.join('\n') || 'No executable plan was generated for this round';
  }
  if (event.role === 'intent') {
    const model = event.modelSummary || {};
    return [
      model.goal ? `Handoff: ${compactAgentValue(model.goal, 120)}` : '',
      model.reason ? `Reason: ${compactAgentValue(model.reason, 140)}` : '',
      summary ? compactAgentValue(summary, 200) : '',
    ].filter(Boolean).join('\n') || 'Intent parsing completed';
  }
  if (event.role === 'observer') {
    return summary ? `Observation: ${compactAgentValue(summary, 260)}` : calls.length ? `Completed ${calls.length} tool steps in this round` : 'No tool results in this round';
  }
  if (event.role === 'reflector') {
    const progress = event.acceptanceProgress || {};
    const requirements = Array.isArray(progress.requirements) ? progress.requirements : [];
    const completedCount = requirements.filter((item) => item && item.completed).length;
    const progressText = requirements.length ? `${completedCount}/${requirements.length}` : '';
    const sourceLabel = event.decisionSource === 'model'
      ? 'Model Decision'
      : event.decisionSource === 'acceptance_guard'
        ? 'Acceptance Guard Correction'
        : event.decisionSource === 'deterministic_fallback'
          ? 'Deterministic Acceptance Guard'
          : '';
    return [
      event.acceptanceGoal ? `Acceptance: ${compactAgentValue(event.acceptanceGoal, 140)}` : '',
      event.currentFocus ? `Focus: ${compactAgentValue(event.currentFocus, 120)}` : '',
      progressText ? `Progress: ${progressText}${progress.acceptanceSatisfied ? ' (Satisfied)' : ' (Unmet)'}` : '',
      `Decision: ${stateLabel(event.state)}${sourceLabel ? ` · ${sourceLabel}` : ''}`,
      event.evidenceGap ? `Evidence Gap: ${compactAgentValue(event.evidenceGap, 130)}` : '',
      event.nextAction ? `Next Action: ${compactAgentValue(event.nextAction, 150)}` : '',
      event.nextRound ? `Transition: Round ${event.nextRound} → PlanAgent` : '',
      summary ? `Rationale: ${compactAgentValue(summary, 180)}` : '',
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
  getTrack: 'Read Target Track',
  getFrames: 'Read Keyframes',
  getClip: 'Generate Vessel Clip',
  getRegistry: 'Read Registry Item',
  matchHull: 'Match Hull Number',
  listRegistry: 'List Registry',
  matchText: 'Match Text Features',
  matchImage: 'Match Image Features',
  verifyTarget: 'Verify Gray-Zone Evidence',
  showEvidence: 'Organize Evidence',
  dedupTracks: 'Deduplicate Trajectories',
};

function planStatusLabel(status) {
  return ({pending: 'Pending', running: 'Running', completed: 'Completed', failed: 'Failed', skipped: 'Skipped'})[status] || 'Pending';
}

function initializePlanBlueprint(blueprint) {
  if (agentPlanLocked || !Array.isArray(blueprint) || !blueprint.length) return;
  agentPlanItems = blueprint.map((step, index) => {
    const tools = Array.isArray(step.tools) ? step.tools.map(String) : [];
    return {
      key: String(step.stepId || `plan-step-${index + 1}`),
      stepId: String(step.stepId || `plan-step-${index + 1}`),
      id: '',
      title: String(step.title || toolActionLabels[tools[0]] || 'Execute Plan Step'),
      tools,
      tool: tools[0] || '',
      round: 0,
      status: 'pending',
      optional: Boolean(step.optional),
      detail: step.optional ? 'Conditional step for gray-zone evidence' : 'Pending execution',
    };
  });
  agentPlanLocked = true;
  const round = document.getElementById('agentPlanRound');
  const decision = document.getElementById('agentPlanDecision');
  if (round) round.textContent = `Full plan · ${agentPlanItems.length} steps`;
  if (decision) {
    decision.className = 'agent-plan-decision active';
    decision.textContent = 'Full plan generated; later rounds update existing step states.';
  }
  renderPlanProgress();
}
function renderPlanProgress() {
  const container = document.getElementById('agentPlanProgress');
  const meter = document.getElementById('agentPlanMeter');
  if (!container) return;
  const total = agentPlanItems.length;
  const completed = agentPlanItems.filter((item) => ['completed', 'skipped'].includes(item.status)).length;
  const hasRunning = agentPlanItems.some((item) => item.status === 'running');
  const hasFailed = agentPlanItems.some((item) => item.status === 'failed');
  const round = document.getElementById('agentPlanRound');
  if (meter) meter.style.width = total ? `${Math.round((completed / total) * 100)}%` : '0%';
  if (!total) {
    container.innerHTML = '<div class="agent-plan-empty">Plan steps will update here.</div>';
    return;
  }
  if (round) {
    const progress = `${completed}/${total} steps`;
    round.textContent = hasFailed ? `${progress} · Failed` : hasRunning ? `${progress} · Running` : completed === total ? `${progress} · Completed` : `${progress} · Pending`;
  }
  const visibleItems = agentPlanItems;
  container.innerHTML = visibleItems.map((item, index) => {
    const symbol = item.status === 'completed' ? '✓' : item.status === 'failed' ? '!' : item.status === 'skipped' ? '—' : '';
    const action = item.title || toolActionLabels[item.tool] || 'Execute Tool Step';
    const detail = item.detail ? `<small>${escapeHtml(compactAgentValue(item.detail, 90))}</small>` : `<small>Round ${item.round}</small>`;
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
  setAgentInitSummary('Waiting for intent');
  const round = document.getElementById('agentPlanRound');
  const decision = document.getElementById('agentPlanDecision');
  if (round) round.textContent = 'Waiting for plan';
  if (decision) {
    decision.className = 'agent-plan-decision';
    decision.textContent = 'Waiting for ReflectAgent verification.';
  }
  renderPlanProgress();
}

function syncPlanProgress(event) {
  initializePlanBlueprint(event.planBlueprint);
  const roundNumber = Number(event.round || 1);
  const calls = Array.isArray(event.calls) ? event.calls : [];
  calls.forEach((call, index) => {
    const stepId = String(call.planStepId || '');
    const tool = String(call.tool || call.id || 'tool');
    let item = agentPlanItems.find((entry) => stepId && entry.stepId === stepId);
    if (!item) item = agentPlanItems.find((entry) => (entry.tools || []).includes(tool));
    if (!item && !agentPlanLocked) {
      item = {
        key: stepId || `fallback-step-${index + 1}`,
        stepId: stepId || `fallback-step-${index + 1}`,
        id: '',
        title: toolActionLabels[tool] || 'Execute Tool Step',
        tools: [tool],
        tool,
        round: 0,
        status: 'pending',
        optional: false,
        detail: 'Pending execution',
      };
      agentPlanItems.push(item);
    }
    if (!item) return;
    item.id = String(call.id || item.id || '');
    item.round = roundNumber;
    item.status = 'pending';
    item.detail = `Round ${roundNumber} · Pending`;
  });
  const round = document.getElementById('agentPlanRound');
  const decision = document.getElementById('agentPlanDecision');
  if (round) round.textContent = `Full plan · Round ${roundNumber}`;
  if (decision) {
    decision.className = 'agent-plan-decision active';
    decision.textContent = calls.length ? `ObserveAgent is executing round ${roundNumber}.` : 'No executable step in this round; waiting for ReflectAgent.';
  }
  renderPlanProgress();
}

function updatePlanProgressFromTool(event) {
  const roundNumber = Number(event.round || activeAgentRound || 1);
  const id = String(event.id || '');
  const stepId = String(event.planStepId || '');
  const tool = String(event.tool || event.id || 'tool');
  let item = agentPlanItems.find((entry) => stepId && entry.stepId === stepId);
  if (!item) item = agentPlanItems.find((entry) => (entry.tools || []).includes(tool));
  if (!item && !agentPlanLocked) {
    item = {
      key: stepId || `fallback-step-${agentPlanItems.length + 1}`,
      stepId: stepId || `fallback-step-${agentPlanItems.length + 1}`,
      id,
      title: toolActionLabels[tool] || 'Execute Tool Step',
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
  item.detail = event.error || event.summary || `Round ${roundNumber} · ${planStatusLabel(item.status)}`;
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
    decision.textContent = `Round ${event.round || activeAgentRound} verified; generating the final answer.`;
    setAgentResultState('Synthesizing', 'running');
  } else if (state === 'replan') {
    decision.textContent = `Replan: ${compactAgentValue(event.nextAction || event.evidenceGap || 'additional evidence is required', 150)}`;
    setAgentResultState('Waiting for Replan', 'running');
  } else if (state === 'conflict') {
    decision.textContent = `Evidence Conflict: ${compactAgentValue(event.evidenceGap || event.nextAction || 'evidence must be rechecked', 150)}`;
    setAgentResultState('Evidence Conflict', 'running');
  } else {
    decision.textContent = `Uncertain: ${compactAgentValue(event.evidenceGap || event.nextAction || 'insufficient evidence', 150)}`;
    setAgentResultState('Insufficient Evidence', 'running');
  }
}

function rolePlaceholder(role, forRound = false) {
  if (role === 'intent') return 'Waiting for intent parsing…';
  if (role === 'planner') return forRound ? 'Waiting for this round plan…' : 'Waiting for planning…';
  if (role === 'observer') return forRound ? 'Waiting for this round execution…' : 'Waiting for tool execution…';
  if (role === 'reflector') return forRound ? 'Waiting for this round verification…' : 'Waiting for verification…';
  return 'Waiting…';
}

function roleRunningLabel(role) {
  return ({
    intent: 'Initializing',
    planner: 'Planning',
    observer: 'Running',
    reflector: 'Verifying',
  })[role] || 'Running';
}

function rolePendingText(role) {
  return ({
    intent: 'Parsing the user question and acceptance target…',
    planner: 'Drafting the tool plan for this round…',
    observer: 'Running tools and collecting evidence…',
    reflector: 'Checking evidence against the acceptance target…',
  })[role] || 'Processing…';
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
    if (round) round.textContent = role === 'intent' ? 'Global' : 'Waiting for round';
    if (state) state.textContent = 'Waiting';
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
    if (round) round.textContent = `Round ${normalizedRound}`;
    if (state) state.textContent = 'Waiting';
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
  const mode = document.getElementById('agentPlanMode');
  if (intent) {
    intent.classList.add('active');
    const cardState = intent.querySelector('.agent-thought-head em');
    if (cardState) cardState.textContent = 'Initializing';
    setAgentStreamText(intent, 'Parsing the question and acceptance target…', {cursor: true});
    setAgentInitSummary('Parsing intent');
  }
  setAgentResultState('Initializing', 'running');
  if (mode) mode.textContent = 'Plan → Observe → Verify · Reflect controls the next round';
  setThinkingState('Reasoning', 'active');
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
    if (event.hullNumber) tags.push(`<span>Hull ${escapeHtml(event.hullNumber)}</span>`);
    if (event.description) tags.push(`<span>${escapeHtml(compactAgentValue(event.description, 24))}</span>`);
    return tags.join('') + skills;
  }
  if (event.role === 'planner') {
    const toolCount = (event.calls || []).length;
    const tools = toolCount ? `<span>Tool plan ${toolCount}</span>` : '';
    const repair = event.planRepair ? `<span class="failed" title="${escapeHtml(event.planRepair)}">Plan Repaired</span>` : '';
    return skills + tools + repair;
  }
  if (event.role === 'observer') {
    const calls = event.calls || [];
    const failed = calls.filter((call) => !call.skipped && call.ok === false).length;
    return skills + (calls.length ? `<span class="${failed ? 'failed' : ''}">Tools ${calls.length}${failed ? ` · ${failed} failed` : ''}</span>` : '');
  }
  if (event.role === 'reflector') {
    return `${skills}<span>${escapeHtml(stateLabel(event.state))}</span>${event.evidenceGap ? `<span>Gap: ${escapeHtml(event.evidenceGap)}</span>` : ''}`;
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
    const card = ensureAgentCard(0, 'intent');
    setAgentResultState('Initializing', 'running');
    if (card) {
      card.classList.add('active');
      card.classList.remove('failed');
      const head = card.querySelector('.agent-thought-head em');
      if (head) head.textContent = 'Initializing';
      setAgentStreamText(card, event.message || 'Parsing user intent…', {cursor: true});
      setAgentInitSummary('Parsing intent');
    }
    setThinkingState('Initializing', 'active');
    return;
  }
  if (event.type === 'status' && event.role) {
    const card = ensureAgentCard(event.round, event.role);
    if (card && event.message) {
      if (!card.classList.contains('active')) card.classList.add('active');
      const head = card.querySelector('.agent-thought-head em');
      if (head && head.textContent === 'Waiting') head.textContent = roleRunningLabel(event.role);
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
    if (decision) decision.textContent = `${stateLabel(event.state)} · ${Number(event.trackCount || 0)} candidate tracks.`;
    setAgentResultState('Synthesizing', 'running');
    setThinkingState('Synthesizing Result', 'active');
  }
}

function renderIntentAgentCard(event) {
  const card = ensureAgentCard(0, 'intent');
  const timeScope = event.timeParseError ? `Parse Error: ${event.timeParseError}` : event.queryScope ? `${formatMonitorTime(event.queryScope[0])}—${formatMonitorTime(event.queryScope[1])}` : 'All monitoring time';
  const targetItems = Array.isArray(event.targetItems) ? event.targetItems.filter((item) => item && item.label) : [];
  const targetText = targetItems.length > 1
    ? targetItems.map((item) => item.label).join('、')
    : event.hullNumber || event.description || targetKindLabel(event.targetKind);
  const route = `${scopeLabel(event.targetScope)} · ${operationLabel(event.operation)} · ${relationLabel(event.registryRelation)}`;
  const acceptance = event.successCriteria || event.expectedOutcome || '按查询目标返回可核验证据';
  const focus = event.nextAgentFocus || '根据当前验收缺口规划第一轮工具调用';
  if (card) {
    const summary = [
      `Target: ${targetText || '—'}`,
      `Route: ${route}`,
      `Time: ${timeScope}`,
      `Acceptance: ${acceptance}`,
      `Focus: ${focus}`,
    ].join('\n');
    card.classList.remove('active');
    card.classList.toggle('failed', Boolean(event.timeParseError));
    card.dataset.streamText = summary;
    const head = card.querySelector('.agent-thought-head em');
    if (head) head.textContent = event.timeParseError ? 'Needs Review' : 'Completed';
    setAgentInitSummary(event.timeParseError ? 'Needs review' : `Initialized · ${questionTypeLabel(event.questionType)}`);
    setAgentStreamText(card, summary);
    const tags = card.querySelector('.agent-thought-tags');
    if (tags) tags.innerHTML = agentTags({role: 'intent', ...event});
  }
  setAgentResultState(event.timeParseError ? 'Needs Review' : 'Planning', 'running');
  initializePlanBlueprint(event.planBlueprint);
  setThinkingState('Planning', 'active');
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
  return `<details class="agent-activity agent-activity-tool${running ? ' running' : ''}" data-activity-kind="tools"${agentActivityOpen(card, 'tools') ? ' open' : ''}><summary><span class="agent-activity-icon" aria-hidden="true">◇</span><span title="${escapeHtml((current && (current.tool || current.id)) || label)}">${escapeHtml(label)}</span>${running ? '<i class="agent-activity-cursor" aria-hidden="true"></i>' : ''}<b aria-hidden="true"></b></summary><div class="agent-activity-body">${details}</div></details>`;
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
  if (state) state.textContent = running ? 'Running' : roleRunningLabel('observer');
  const streamText = card.dataset.streamText || 'Running tools and collecting evidence…';
  setAgentProcessStream(card, streamText, {cursor: running});
  scrollThoughtStreamToCard(card);
  updatePlanProgressFromTool(event);
}

function compactAgentErrorMessage(raw) {
  const text = String(raw || 'Unknown error').replace(/\s+/g, ' ').trim();
  if (/GRAPH_RECURSION_LIMIT|Recursion limit/i.test(text)) {
    return 'The plan exceeded the recursion limit and was stopped. Retry or simplify the question.';
  }
  if (/allowed-local-media-path|Cannot load local files/i.test(text)) {
    return 'The vision model rejected a local media path. Verify that the server sends images as data URLs.';
  }
  if (/chat\/completions|Internal Server Error|HTTPStatusError/i.test(text)) {
    return 'The vision or language model service is temporarily unavailable. Try again later.';
  }
  return text.length > 180 ? `${text.slice(0, 180)}…` : text;
}

function appendThoughtEvent(event) {
  // 推理期间保持计划展开；最终结果生成后自动折叠计划与工具。
  if (event.type === 'complete') {
    setAgentResultState('Finalizing', 'running');
    setThinkingState('Reasoning Completed', 'completed');
    const decision = document.getElementById('agentPlanDecision');
    if (decision && !decision.classList.contains('sufficient')) {
      decision.className = 'agent-plan-decision sufficient';
      decision.textContent = 'Closed-loop reasoning completed; final answer and evidence are ready.';
    }
    document.querySelectorAll('.agent-thought-card.active').forEach((card) => {
      card.classList.remove('active');
      const state = card.querySelector('.agent-thought-head em');
      if (state && ['Initializing', 'Planning', 'Running', 'Verifying'].includes(state.textContent)) {
        state.textContent = 'Completed';
      }
    });
    return;
  }
  if (event.type === 'error') {
    setAgentResultState('Failed', 'failed');
    setThinkingState('Reasoning Failed', 'failed');
    const decision = document.getElementById('agentPlanDecision');
    if (decision) {
      decision.className = 'agent-plan-decision conflict';
      decision.textContent = `Execution Failed: ${compactAgentErrorMessage(event.message)}`;
    }
    document.querySelectorAll('.agent-thought-card.active').forEach((card) => {
      card.classList.remove('active');
      card.classList.add('failed');
      const state = card.querySelector('.agent-thought-head em');
      if (state) state.textContent = 'Interrupted';
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
      setAgentResultState('Initializing', 'running');
      setThinkingState('Initializing', 'active');
    } else if (event.role === 'planner') {
      setAgentResultState(`Round ${event.round || 1} · Planning`, 'running');
      setThinkingState(`Round ${event.round || 1} · Planning`, 'active');
      const round = document.getElementById('agentPlanRound');
      const decision = document.getElementById('agentPlanDecision');
      if (round) round.textContent = `Round ${event.round || 1} · Planning`;
      if (decision) {
        decision.className = 'agent-plan-decision active';
        decision.textContent = 'Generating a plan from the current acceptance gap.';
      }
    } else if (event.role === 'observer') {
      setAgentResultState(`Round ${event.round || 1} · Running`, 'running');
      setThinkingState(`Round ${event.round || 1} · Running`, 'active');
    } else if (event.role === 'reflector') {
      setAgentResultState(`Round ${event.round || 1} · Verifying`, 'running');
      setThinkingState(`Round ${event.round || 1} · Verifying`, 'active');
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
      card.dataset.streamToken = appendStreamDelta(card.dataset.streamToken || '', event.delta, 1600);
    } else {
      card.dataset.streamThinking = appendStreamDelta(card.dataset.streamThinking || '', event.delta, 2400);
    }
    const thinking = (card.dataset.streamThinking || '').trim();
    const token = (card.dataset.streamToken || '').trim();
    const live = [thinking ? `Thinking:\n${thinking}` : '', token ? `Draft:\n${token}` : '']
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
    const conclusion = summaryText || event.message || event.fallback || 'No displayable information for this round';
    const finalText = thinkingTrail
      ? `Reasoning Trail:\n${compactAgentValue(thinkingTrail, 900)}\n\nConclusion:\n${conclusion}`
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
    let statusLabel = 'Completed';
    if (isPlanCard && event.planRepair) statusLabel = 'Repaired';
    else if (isPlanCard && event.fallback && String(event.fallback).includes('安全回退')) statusLabel = 'Safe Fallback';
    else if (isPlanCard && event.fallback) statusLabel = 'Default Plan';
    else if (!isPlanCard && event.fallback) statusLabel = 'Summary Failed';
    else if (hasHardFail) statusLabel = 'Partially Failed';
    else if (onlySkipped) statusLabel = 'Partially Skipped';
    else if (event.role === 'reflector') statusLabel = stateLabel(event.state);
    else if (event.role === 'intent') statusLabel = 'Completed';
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
    throw new Error(body.detail || `Request failed: ${response.status}`);
  }
  if (!response.body) throw new Error('This browser does not support streaming responses');

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
      if (event.type === 'error') throw new Error(event.message || 'Closed-loop reasoning failed');
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
      if (event.type === 'error') throw new Error(event.message || 'Closed-loop reasoning failed');
    }
  }
  if (!result) throw new Error('No final answer was received');
  return result;
}

async function askAgent() {
  const question = document.getElementById('agentQuestion').value.trim();
  const button = document.getElementById('btnAskAgent');
  const clearButton = document.getElementById('agentMemoryClearButton');
  if (!question) return showToast('Enter a question', 'error');
  try {
    button.disabled = true;
    if (clearButton) clearButton.disabled = true;
    button.textContent = 'Reasoning…';
    // 只展示过程流，最终回答区延后到 complete
    resetThoughtStream();
    const answer = document.getElementById('agentAnswer');
    if (answer) {
      answer.className = 'agent-answer empty-state';
      answer.textContent = 'The final answer will appear here after reasoning.';
    }
    renderEvidence(null, null, 'Reasoning in progress. Evidence will appear when complete.');
    const result = await streamAgentQuery(question, selectedTopK());
    // 过程流结束后再切换最终结果
    renderAgentAnswer(result);
    renderEvidence(result.evidence, result.displayGroups, 'No evidence available', result.registryItems || [], result);
  } catch (error) {
    showAgentFinalView('failed');
    setThinkingState('Reasoning Failed', 'failed');
    const answer = document.getElementById('agentAnswer');
    if (answer) {
      answer.className = 'agent-answer empty-state';
      answer.textContent = `Execution Failed: ${compactAgentErrorMessage(error.message)}`;
    }
    renderEvidence(null, null);
    showToast(compactAgentErrorMessage(error.message), 'error');
  } finally {
    button.disabled = false;
    button.textContent = 'Run Closed-Loop Reasoning';
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
