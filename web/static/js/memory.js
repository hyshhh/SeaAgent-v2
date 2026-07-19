let memoryRefreshTimer = null;
let memorySettingsLoaded = false;

function formatMemoryTime(timestamp) {
  return formatMonitorTime(timestamp);
}

function escapeMemoryText(value) {
  return String(value ?? '').replace(/[&<>'"]/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[char]));
}

function memoryMatchLabel(value) {
  const labels = {confirmed:'已确认', candidate:'候选', conflict:'冲突', unknown:'未知'};
  return labels[value] || value || '未知';
}

function renderTrackMemory(payload) {
  document.getElementById('memoryTrackCount').textContent = payload.trackCount ?? 0;
  document.getElementById('memoryKeyframeCount').textContent = payload.keyframeCount ?? 0;
  document.getElementById('memoryEmbeddedCount').textContent = payload.embeddedKeyframeCount ?? 0;
  if (!memorySettingsLoaded) {
    const retentionSeconds = Number(payload.settings?.retentionSeconds ?? 0);
    document.getElementById('memoryRetentionHours').value = Math.round(retentionSeconds / 3600 * 10) / 10;
    memorySettingsLoaded = true;
  }
  const body = document.getElementById('trackMemoryTable');
  const tracks = payload.tracks || [];
  if (!tracks.length) {
    body.innerHTML = '<tr><td colspan="7" class="empty-msg memory-empty">当前轨迹表为空，请先在海域监控页面处理视频。</td></tr>';
    return;
  }
  body.innerHTML = tracks.map(track => {
    const hull = track.finalHullNumber || '无稳定舷号';
    const frames = `${track.embeddedKeyframeCount || 0}/${track.keyframeCount || 0}`;
    const stateClass = track.memoryState === '已完成' ? 'done' : 'building';
    return `<tr>
      <td><strong class="memory-track-id">${escapeMemoryText(track.trackId)}</strong></td>
      <td>${formatMemoryTime(track.startTime)} — ${formatMemoryTime(track.endTime)}</td>
      <td>${escapeMemoryText(hull)}</td>
      <td><span class="memory-state ${escapeMemoryText(track.finalMatchType)}">${escapeMemoryText(memoryMatchLabel(track.finalMatchType))}</span></td>
      <td><span class="memory-frame-count">${frames}</span><small>已向量/总数</small></td>
      <td><span class="memory-state ${stateClass}">${escapeMemoryText(track.memoryState)}</span></td>
      <td class="memory-description">${escapeMemoryText(track.finalDescription || '暂无稳定描述')}</td>
    </tr>`;
  }).join('');
}

async function loadTrackMemory(silent = false) {
  const state = document.getElementById('memoryRefreshState');
  try {
    if (!silent) state.textContent = '正在刷新…';
    const payload = await apiFetch('/api/memory/tracks');
    renderTrackMemory(payload);
    state.textContent = `已更新 ${new Date().toLocaleTimeString('zh-CN', {hour12:false})}`;
  } catch (error) {
    state.textContent = '刷新失败';
    if (!silent) showToast(error.message, 'error');
  }
}

function startMemoryAutoRefresh() {
  stopMemoryAutoRefresh();
  memoryRefreshTimer = setInterval(() => loadTrackMemory(true), 2000);
}

function stopMemoryAutoRefresh() {
  if (memoryRefreshTimer) clearInterval(memoryRefreshTimer);
  memoryRefreshTimer = null;
}

async function saveMemorySettings() {
  const input = document.getElementById('memoryRetentionHours');
  const hours = Number(input.value);
  if (!Number.isFinite(hours) || hours < 0 || hours > 24) {
    showToast('请输入 0 至 24 小时之间的维护时间', 'error');
    return;
  }
  const retentionSeconds = Math.round(hours * 3600);
  try {
    const response = await apiFetch('/api/memory/settings', {
      method: 'PUT',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({retention_seconds: retentionSeconds})
    });
    memorySettingsLoaded = false;
    showToast(response.message || '记忆维护时间已更新');
    await loadTrackMemory(true);
  } catch (error) {
    showToast(error.message, 'error');
  }
}

async function clearAllMemory() {
  if (!confirm('确定清除全部轨迹记忆吗？该操作无法撤销，先验库不会受到影响。')) return;
  try {
    const response = await apiFetch('/api/memory', {method: 'DELETE'});
    showToast(response.message || '轨迹记忆已清除');
    await loadTrackMemory(true);
    if (typeof loadAgentMemorySummary === 'function') loadAgentMemorySummary();
  } catch (error) {
    showToast(error.message, 'error');
  }
}
