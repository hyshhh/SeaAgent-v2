const SHIP_API = '/api/ships';
const MAX_REFERENCE_IMAGES = 6;

let ships = [];
let editingHull = null;
let editingReferences = [];
let uploadFiles = [];
let recognizedUpload = null;
let toastTimer = null;

function escapeHtml(value) {
  return String(value ?? '')
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#39;');
}

function formatVideoTime(seconds) {
  const value = Number(seconds);
  if (!Number.isFinite(value)) return '时间未知';
  const tenths = Math.max(0, Math.round(value * 10));
  const hours = Math.floor(tenths / 36000);
  const minutes = Math.floor((tenths % 36000) / 600);
  const wholeSeconds = Math.floor((tenths % 600) / 10);
  const fraction = tenths % 10;
  return `${String(hours).padStart(2, '0')}:${String(minutes).padStart(2, '0')}:${String(wholeSeconds).padStart(2, '0')}.${fraction}`;
}

function formatMonitorTime(timestamp) {
  const value = Number(timestamp);
  if (!Number.isFinite(value)) return '时间未知';
  if (value < 946684800) return `历史视频 ${formatVideoTime(value)}`;
  const date = new Date(value * 1000);
  const pad = number => String(number).padStart(2, '0');
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())} ${pad(date.getHours())}:${pad(date.getMinutes())}:${pad(date.getSeconds())}`;
}

function showToast(message, type = 'success') {
  const toast = document.getElementById('toast');
  if (!toast) return;
  clearTimeout(toastTimer);
  toast.textContent = message;
  toast.className = `toast show${type === 'error' ? ' error' : ''}`;
  toastTimer = setTimeout(() => { toast.className = 'toast'; }, 3200);
}

async function apiFetch(url, options = {}) {
  const response = await fetch(url, options);
  const contentType = response.headers.get('content-type') || '';
  const body = contentType.includes('application/json') ? await response.json() : null;
  if (!response.ok) throw new Error(body?.detail || body?.message || `请求失败：${response.status}`);
  return body;
}

function imageFiles(files) {
  return Array.from(files || []).filter((file) => file.type.startsWith('image/'));
}

function validateFiles(files, available = MAX_REFERENCE_IMAGES) {
  const selected = imageFiles(files);
  if (selected.length !== Array.from(files || []).length) throw new Error('只能选择图片文件');
  if (selected.length > available) throw new Error(`最多还能选择 ${available} 张参考图`);
  if (selected.some((file) => file.size > 20 * 1024 * 1024)) throw new Error('单张图片不能超过 20MB');
  return selected;
}

function renderLocalPreviews(containerId, files) {
  const container = document.getElementById(containerId);
  if (!container) return;
  container.innerHTML = '';
  files.forEach((file) => {
    const card = document.createElement('div');
    card.className = 'reference-thumb';
    const image = document.createElement('img');
    image.alt = file.name;
    image.src = URL.createObjectURL(file);
    image.onload = () => URL.revokeObjectURL(image.src);
    card.appendChild(image);
    container.appendChild(card);
  });
}

function referenceHtml(reference, hullNumber, allowDelete = true) {
  const referenceId = encodeURIComponent(reference.referenceId);
  const hull = encodeURIComponent(hullNumber);
  const state = reference.isEmbedded ? '已写入向量库' : '不可检索';
  const button = allowDelete
    ? `<button title="删除参考图" onclick="deleteReference('${hull}','${referenceId}')">×</button>`
    : '';
  return `<div class="reference-thumb" title="${state}"><img loading="lazy" src="/api/evidence/registry/${referenceId}" alt="先验库参考图">${button}</div>`;
}

async function loadShips(query = '') {
  const table = document.getElementById('shipTable');
  if (table) table.innerHTML = '<tr><td colspan="4" class="empty-msg">正在加载…</td></tr>';
  try {
    const url = query.trim() ? `${SHIP_API}/search?q=${encodeURIComponent(query.trim())}` : SHIP_API;
    const data = await apiFetch(url);
    ships = data.ships || data.results || [];
    renderShips();
    await loadStats();
  } catch (error) {
    if (table) table.innerHTML = `<tr><td colspan="4" class="empty-msg">${escapeHtml(error.message)}</td></tr>`;
    showToast(error.message, 'error');
  }
}

async function loadStats() {
  const stats = await apiFetch(`${SHIP_API}/stats`);
  document.getElementById('totalCount').textContent = stats.total_ships;
  document.getElementById('totalImages').textContent = stats.total_reference_images;
  document.getElementById('backendType').textContent = stats.backend;
}

function renderShips() {
  const table = document.getElementById('shipTable');
  if (!table) return;
  if (!ships.length) {
    table.innerHTML = '<tr><td colspan="4" class="empty-msg">暂无先验库项</td></tr>';
    return;
  }
  table.innerHTML = ships.map((ship) => {
    const references = ship.references || [];
    const state = ship.searchable ? '<span class="status-tag ok">可检索</span>' : '<span class="status-tag off">缺少有效向量</span>';
    const images = references.length
      ? `<div class="reference-grid">${references.map((item) => referenceHtml(item, ship.hull_number)).join('')}</div>`
      : '<span class="hint">暂无参考图</span>';
    const hull = encodeURIComponent(ship.hull_number);
    return `<tr>
      <td><span class="hull-num">${escapeHtml(ship.hull_number)}</span></td>
      <td>${escapeHtml(ship.description || '—')}</td>
      <td>${state}<span class="hint"> ${references.length}/${MAX_REFERENCE_IMAGES}</span>${images}</td>
      <td><div class="actions"><button class="btn ghost btn-sm" onclick="openEditModal('${hull}')">编辑</button><button class="btn danger btn-sm" onclick="deleteShip('${hull}')">删除</button></div></td>
    </tr>`;
  }).join('');
}

function resetShipModal() {
  editingHull = null;
  editingReferences = [];
  document.getElementById('modalHullNumber').value = '';
  document.getElementById('modalHullNumber').disabled = false;
  document.getElementById('modalDescription').value = '';
  document.getElementById('modalReferenceFiles').value = '';
  document.getElementById('modalReferenceList').innerHTML = '';
}

function openAddModal() {
  resetShipModal();
  document.getElementById('modalTitle').textContent = '新建先验库项';
  document.getElementById('shipModal').classList.add('active');
}

async function openEditModal(encodedHull) {
  const hull = decodeURIComponent(encodedHull);
  try {
    const ship = await apiFetch(`${SHIP_API}/${encodeURIComponent(hull)}`);
    editingHull = ship.hull_number;
    editingReferences = ship.references || [];
    document.getElementById('modalTitle').textContent = `编辑舷号 ${ship.hull_number}`;
    document.getElementById('modalHullNumber').value = ship.hull_number;
    document.getElementById('modalHullNumber').disabled = true;
    document.getElementById('modalDescription').value = ship.description || '';
    document.getElementById('modalReferenceFiles').value = '';
    document.getElementById('modalReferenceList').innerHTML = editingReferences.length
      ? editingReferences.map((item) => referenceHtml(item, ship.hull_number)).join('')
      : '<span class="hint">暂无参考图</span>';
    document.getElementById('shipModal').classList.add('active');
  } catch (error) {
    showToast(error.message, 'error');
  }
}

function closeModal() {
  document.getElementById('shipModal').classList.remove('active');
  resetShipModal();
}

async function submitShip() {
  const button = document.getElementById('modalSubmitBtn');
  const hull = document.getElementById('modalHullNumber').value.trim();
  const description = document.getElementById('modalDescription').value.trim();
  let files;
  try {
    files = validateFiles(document.getElementById('modalReferenceFiles').files, MAX_REFERENCE_IMAGES - editingReferences.length);
    if (!hull) throw new Error('舷号不能为空');
    button.disabled = true;
    button.textContent = '保存中…';
    if (editingHull) {
      await apiFetch(`${SHIP_API}/${encodeURIComponent(editingHull)}`, {
        method: 'PUT', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({description})
      });
      if (files.length) await addReferenceImages(editingHull, files);
    } else if (files.length) {
      const form = new FormData();
      files.forEach((file) => form.append('files', file));
      form.append('hull_number', hull);
      form.append('description', description);
      form.append('aliases', '[]');
      await apiFetch(`${SHIP_API}/upload`, {method: 'POST', body: form});
    } else {
      await apiFetch(SHIP_API, {
        method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({hull_number: hull, description, aliases: []})
      });
    }
    showToast('先验库项已保存');
    closeModal();
    await loadShips(document.getElementById('searchInput').value);
  } catch (error) {
    showToast(error.message, 'error');
  } finally {
    button.disabled = false;
    button.textContent = '保存';
  }
}

async function addReferenceImages(hull, files) {
  const form = new FormData();
  files.forEach((file) => form.append('files', file));
  return apiFetch(`${SHIP_API}/${encodeURIComponent(hull)}/images`, {method: 'POST', body: form});
}

async function deleteReference(encodedHull, encodedReferenceId) {
  const hull = decodeURIComponent(encodedHull);
  const referenceId = decodeURIComponent(encodedReferenceId);
  if (!confirm(`确认删除舷号 ${hull} 的这张参考图？`)) return;
  try {
    await apiFetch(`${SHIP_API}/${encodeURIComponent(hull)}/images/${encodeURIComponent(referenceId)}`, {method: 'DELETE'});
    showToast('参考图已删除，向量库已重建');
    if (editingHull === hull) await openEditModal(encodeURIComponent(hull));
    await loadShips(document.getElementById('searchInput').value);
  } catch (error) {
    showToast(error.message, 'error');
  }
}

async function deleteShip(encodedHull) {
  const hull = decodeURIComponent(encodedHull);
  if (!confirm(`确认删除舷号 ${hull} 及其全部参考图？`)) return;
  try {
    await apiFetch(`${SHIP_API}/${encodeURIComponent(hull)}`, {method: 'DELETE'});
    showToast(`已删除舷号 ${hull}`);
    await loadShips(document.getElementById('searchInput').value);
  } catch (error) {
    showToast(error.message, 'error');
  }
}

function openBulkModal() {
  document.getElementById('bulkInput').value = '';
  document.getElementById('bulkModal').classList.add('active');
}

function closeBulkModal() {
  document.getElementById('bulkModal').classList.remove('active');
}

async function submitBulk() {
  try {
    const shipsObject = JSON.parse(document.getElementById('bulkInput').value);
    if (!shipsObject || Array.isArray(shipsObject) || typeof shipsObject !== 'object') throw new Error('请输入舷号到描述的对象');
    const result = await apiFetch(`${SHIP_API}/bulk`, {
      method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({ships: shipsObject})
    });
    showToast(result.message);
    closeBulkModal();
    await loadShips();
  } catch (error) {
    showToast(error.message, 'error');
  }
}

function openUploadModal() {
  uploadFiles = [];
  recognizedUpload = null;
  document.getElementById('uploadFilename').textContent = '';
  document.getElementById('uploadFileInput').value = '';
  document.getElementById('uploadPreviewList').innerHTML = '';
  document.getElementById('recognizeResult').classList.remove('show');
  document.getElementById('recHullNumber').value = '';
  document.getElementById('recDescription').value = '';
  document.getElementById('recExistsWarn').style.display = 'none';
  document.getElementById('btnConfirmAdd').style.display = 'none';
  document.getElementById('uploadModal').classList.add('active');
}

function closeUploadModal() {
  document.getElementById('uploadModal').classList.remove('active');
  uploadFiles = [];
  recognizedUpload = null;
}

function setUploadFiles(files) {
  try {
    uploadFiles = validateFiles(files);
    document.getElementById('uploadFilename').textContent = uploadFiles.map((file) => file.name).join('、');
    renderLocalPreviews('uploadPreviewList', uploadFiles);
    document.getElementById('recognizeResult').classList.remove('show');
    document.getElementById('btnConfirmAdd').style.display = 'none';
  } catch (error) {
    showToast(error.message, 'error');
  }
}

async function doRecognize() {
  if (!uploadFiles.length) return showToast('请先选择参考图', 'error');
  const button = document.getElementById('btnRecognize');
  try {
    button.disabled = true;
    button.textContent = '识别中…';
    const form = new FormData();
    form.append('file', uploadFiles[0]);
    const result = await apiFetch(`${SHIP_API}/recognize`, {method: 'POST', body: form});
    recognizedUpload = result.data;
    document.getElementById('recHullNumber').value = recognizedUpload.hull_number || '';
    document.getElementById('recDescription').value = recognizedUpload.description || '';
    const warning = document.getElementById('recExistsWarn');
    warning.style.display = recognizedUpload.already_exists ? 'block' : 'none';
    warning.textContent = recognizedUpload.already_exists ? '该舷号已存在，确认后将追加参考图并更新描述。' : '';
    document.getElementById('recognizeResult').classList.add('show');
    document.getElementById('btnConfirmAdd').style.display = '';
  } catch (error) {
    showToast(error.message, 'error');
  } finally {
    button.disabled = false;
    button.textContent = '识别首图';
  }
}

async function doConfirmAdd() {
  const hull = document.getElementById('recHullNumber').value.trim();
  const description = document.getElementById('recDescription').value.trim();
  const button = document.getElementById('btnConfirmAdd');
  if (!hull) return showToast('舷号不能为空', 'error');
  if (!uploadFiles.length) return showToast('请先选择参考图', 'error');
  try {
    button.disabled = true;
    button.textContent = '写入中…';
    let existing = recognizedUpload?.already_exists;
    if (recognizedUpload?.hull_number !== hull) {
      try { await apiFetch(`${SHIP_API}/${encodeURIComponent(hull)}`); existing = true; }
      catch (error) { if (!error.message.includes('未找到')) throw error; existing = false; }
    }
    if (existing) {
      const current = await apiFetch(`${SHIP_API}/${encodeURIComponent(hull)}`);
      validateFiles(uploadFiles, MAX_REFERENCE_IMAGES - (current.references || []).length);
      await apiFetch(`${SHIP_API}/${encodeURIComponent(hull)}`, {
        method: 'PUT', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({description})
      });
      await addReferenceImages(hull, uploadFiles);
    } else {
      const form = new FormData();
      uploadFiles.forEach((file) => form.append('files', file));
      form.append('hull_number', hull);
      form.append('description', description);
      form.append('aliases', '[]');
      await apiFetch(`${SHIP_API}/upload`, {method: 'POST', body: form});
    }
    showToast(`舷号 ${hull} 已写入先验库`);
    closeUploadModal();
    await loadShips();
  } catch (error) {
    showToast(error.message, 'error');
  } finally {
    button.disabled = false;
    button.textContent = '确认入库';
  }
}

function bindRegistryEvents() {
  document.getElementById('searchInput')?.addEventListener('input', (event) => loadShips(event.target.value));
  document.getElementById('modalReferenceFiles')?.addEventListener('change', (event) => {
    try {
      const files = validateFiles(event.target.files, MAX_REFERENCE_IMAGES - editingReferences.length);
      const existing = editingReferences.map((item) => referenceHtml(item, editingHull || '', Boolean(editingHull))).join('');
      document.getElementById('modalReferenceList').innerHTML = existing;
      const preview = document.createElement('div');
      preview.className = 'reference-grid';
      document.getElementById('modalReferenceList').appendChild(preview);
      files.forEach((file) => {
        const card = document.createElement('div');
        card.className = 'reference-thumb';
        const image = document.createElement('img');
        image.alt = file.name;
        image.src = URL.createObjectURL(file);
        image.onload = () => URL.revokeObjectURL(image.src);
        card.appendChild(image);
        preview.appendChild(card);
      });
    } catch (error) {
      event.target.value = '';
      showToast(error.message, 'error');
    }
  });
  document.getElementById('uploadFileInput')?.addEventListener('change', (event) => setUploadFiles(event.target.files));
  const zone = document.getElementById('uploadZone');
  zone?.addEventListener('dragover', (event) => { event.preventDefault(); zone.classList.add('dragover'); });
  zone?.addEventListener('dragleave', () => zone.classList.remove('dragover'));
  zone?.addEventListener('drop', (event) => {
    event.preventDefault();
    zone.classList.remove('dragover');
    setUploadFiles(event.dataTransfer.files);
  });
}

window.showToast = showToast;
window.apiFetch = apiFetch;
window.loadShips = loadShips;
window.openAddModal = openAddModal;
window.openEditModal = openEditModal;
window.closeModal = closeModal;
window.submitShip = submitShip;
window.deleteReference = deleteReference;
window.deleteShip = deleteShip;
window.openBulkModal = openBulkModal;
window.closeBulkModal = closeBulkModal;
window.submitBulk = submitBulk;
window.openUploadModal = openUploadModal;
window.closeUploadModal = closeUploadModal;
window.doRecognize = doRecognize;
window.doConfirmAdd = doConfirmAdd;

document.addEventListener('DOMContentLoaded', () => {
  bindRegistryEvents();
  loadShips();
});
