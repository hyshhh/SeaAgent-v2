const SETTINGS_API = '/api/settings';

let systemSettings = {};
let systemSettingSpecs = {};

function settingValue(values, path) {
  return path.split('.').reduce((current, key) => current?.[key], values);
}

function setSettingValue(values, path, value) {
  const keys = path.split('.');
  let current = values;
  keys.slice(0, -1).forEach((key) => {
    if (!current[key] || typeof current[key] !== 'object') current[key] = {};
    current = current[key];
  });
  current[keys[keys.length - 1]] = value;
}

function settingsStatus(message, type = '') {
  const status = document.getElementById('settingsStatus');
  if (!status) return;
  status.textContent = message;
  status.className = `settings-status${type ? ` ${type}` : ''}`;
}

function updateSettingInput(input, value, spec = {}) {
  if (spec.type === 'text' || input.tagName === 'TEXTAREA') {
    input.value = value ?? '';
    if (spec.max_chars) input.maxLength = spec.max_chars;
    return;
  }
  if (spec.type === 'enum' || input.tagName === 'SELECT') {
    const next = value == null ? '' : String(value);
    input.value = next;
    if (spec.choices && !spec.choices.map(String).includes(next) && next) {
      const option = document.createElement('option');
      option.value = next;
      option.textContent = next;
      input.appendChild(option);
      input.value = next;
    }
    return;
  }
  if (spec.min !== undefined) input.min = spec.min;
  if (spec.max !== undefined) input.max = spec.max;
  if (spec.step !== undefined) input.step = spec.step;
  input.value = value ?? '';
}

function fillSystemSettings(data) {
  systemSettings = data.settings || {};
  systemSettingSpecs = data.specs || {};
  document.querySelectorAll('[data-setting]').forEach((input) => {
    const path = input.dataset.setting;
    updateSettingInput(input, settingValue(systemSettings, path), systemSettingSpecs[path]);
  });
  syncMonitoringOptions(systemSettings);
}

function syncMonitoringOptions(settings) {
  const links = [
    ['optConf', 'yolo.confidence'],
    ['optIou', 'yolo.iou'],
    ['optDetectEvery', 'yolo.detect_every_n_frames'],
  ];
  links.forEach(([elementId, path]) => {
    const input = document.getElementById(elementId);
    const value = settingValue(settings, path);
    if (input && value !== undefined && value !== null) input.value = value;
  });
}

function collectSystemSettings() {
  const values = {};
  for (const input of document.querySelectorAll('[data-setting]')) {
    const path = input.dataset.setting;
    const spec = systemSettingSpecs[path] || {};
    const raw = input.value;
    if (spec.type === 'text' || input.tagName === 'TEXTAREA') {
      const text = String(raw || '').trim();
      if (!text) throw new Error(`提示词“${spec.label || path}”不能为空`);
      if (spec.max_chars && text.length > spec.max_chars) {
        throw new Error(`提示词“${spec.label || path}”过长，最多 ${spec.max_chars} 字符`);
      }
      setSettingValue(values, path, text);
      continue;
    }
    if (spec.type === 'enum' || input.tagName === 'SELECT') {
      const choice = String(raw || '').trim();
      if (!choice) throw new Error(`参数「${path}」不能为空`);
      setSettingValue(values, path, choice);
      continue;
    }
    if (String(raw).trim() === '') throw new Error(`参数“${path}”不能为空`);
    const value = spec.type === 'int' ? Number.parseInt(raw, 10) : Number.parseFloat(raw);
    if (!Number.isFinite(value)) throw new Error(`参数“${path}”必须是数字`);
    setSettingValue(values, path, value);
  }
  return values;
}

async function loadSystemSettings(showMessage = false) {
  settingsStatus('正在读取…');
  try {
    const data = await apiFetch(SETTINGS_API);
    fillSystemSettings(data);
    settingsStatus('已读取当前配置', 'success');
    if (showMessage) showToast('设置已重新读取');
  } catch (error) {
    settingsStatus('读取失败', 'error');
    if (showMessage) showToast(error.message, 'error');
  }
}

async function saveSystemSettings() {
  const button = document.getElementById('saveSettingsButton');
  try {
    const settings = collectSystemSettings();
    if (button) {
      button.disabled = true;
      button.textContent = '保存中…';
    }
    settingsStatus('正在保存…');
    const response = await apiFetch(SETTINGS_API, {
      method: 'PUT',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({settings}),
    });
    fillSystemSettings(response.data || response);
    settingsStatus('已保存，新任务与问答生效', 'success');
    showToast('设置已保存，下一轮问答将使用最新提示词与阈值');
  } catch (error) {
    settingsStatus('保存失败', 'error');
    showToast(error.message, 'error');
  } finally {
    if (button) {
      button.disabled = false;
      button.textContent = '保存设置';
    }
  }
}

async function resetSystemSettings() {
  if (!window.confirm('确定恢复全部运行参数和提示词的默认值吗？')) return;
  const button = document.getElementById('saveSettingsButton');
  try {
    if (button) button.disabled = true;
    settingsStatus('正在恢复默认…');
    const response = await apiFetch(`${SETTINGS_API}/reset`, {method: 'POST'});
    fillSystemSettings(response.data || response);
    settingsStatus('已恢复默认值', 'success');
    showToast('运行参数与提示词已恢复默认值');
  } catch (error) {
    settingsStatus('恢复失败', 'error');
    showToast(error.message, 'error');
  } finally {
    if (button) button.disabled = false;
  }
}

window.loadSystemSettings = loadSystemSettings;
window.saveSystemSettings = saveSystemSettings;
window.resetSystemSettings = resetSystemSettings;

document.addEventListener('DOMContentLoaded', () => loadSystemSettings(false));
