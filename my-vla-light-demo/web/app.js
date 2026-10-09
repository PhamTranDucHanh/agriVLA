'use strict';
const el = id => document.getElementById(id);
let ready = false;
let busy = false;
let hardware = false;
const history = [];
async function api(path, options = {}) {
  const response = await fetch(path, { ...options, cache: 'no-store' });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
  return data;
}
function device(state, detail) {
  el('lamp').classList.toggle('on', state === true);
  el('light-status').textContent = state === true ? 'Đèn đang BẬT' : state === false ? 'Đèn đang TẮT' : 'Chưa xác định';
  el('device-detail').textContent = detail;
}
async function refreshState() {
  el('refresh').disabled = true;
  try {
    const data = await api('/api/state');
    device(data.light_on, hardware ? 'Trạng thái GPIO do ESP32 báo về' : 'Chế độ inference · Không điều khiển phần cứng');
  } catch (error) {
    device(null, `Không đọc được ESP32: ${error.message}`);
  } finally { el('refresh').disabled = false; }
}
async function status() {
  try {
    const data = await api('/api/status');
    hardware = data.hardware_enabled;
    ready = data.model === 'ready';
    el('model-status').textContent = ready ? '● Mô hình sẵn sàng' : data.model === 'error' ? 'Lỗi tải mô hình' : 'Đang tải mô hình…';
    el('model-status').classList.toggle('error', data.model === 'error');
    el('submit').disabled = !ready || busy;
    if (!busy) el('message').textContent = data.error || (ready ? (hardware ? 'Sẵn sàng dự đoán và gửi lệnh tới ESP32.' : 'Sẵn sàng dự đoán · Phần cứng chưa được bật.') : 'Model đang được tải lên GPU. Vui lòng chờ…');
    if (ready || data.model === 'error') {
      await refreshState();
      return;
    }
  } catch (error) { el('message').textContent = `Backend chưa kết nối: ${error.message}`; }
  setTimeout(status, 2000);
}
document.querySelectorAll('[data-prompt]').forEach(button => button.addEventListener('click', () => {
  el('prompt').value = button.dataset.prompt;
  el('prompt').focus();
}));
el('refresh').addEventListener('click', refreshState);
el('prompt-form').addEventListener('submit', async event => {
  event.preventDefault();
  const prompt = el('prompt').value.trim();
  if (!prompt || !ready || busy) return;
  busy = true;
  el('submit').disabled = true;
  el('submit').firstChild.textContent = 'Đang dự đoán… ';
  el('message').textContent = 'SmolVLA đang xử lý prompt. Vui lòng chờ kết quả.';
  try {
    const result = await api('/api/infer', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ prompt })
    });
    el('action-value').textContent = `${result.light_value >= 0 ? '+' : ''}${result.light_value.toFixed(4)}`;
    el('semantic').textContent = result.semantic_action;
    el('final-action').textContent = result.hardware_enabled ? result.final_action : 'INFERENCE_ONLY';
    el('elapsed').textContent = `${result.elapsed_seconds.toFixed(2)} giây`;
    el('vector').textContent = JSON.stringify(result.action, null, 2);
    device(result.hardware_state, result.hardware_error ? 'Gửi lệnh thất bại · Trạng thái chưa xác nhận' : result.hardware_enabled ? 'Trạng thái do ESP32 xác nhận' : 'Phần cứng chưa được bật');
    el('message').textContent = result.hardware_error ? `Dự đoán hoàn tất, nhưng ESP32 gặp lỗi: ${result.hardware_error}` : result.final_action === 'NO_ACTION' ? 'NO_ACTION: đèn đã ở trạng thái mong muốn (kiểm tra bằng Python).' : result.hardware_enabled ? 'Dự đoán hoàn tất. ESP32 đã xác nhận lệnh.' : 'Dự đoán hoàn tất. Không gửi lệnh phần cứng.';
    history.unshift({ prompt, action: result.hardware_enabled ? result.final_action : result.semantic_action });
    history.splice(8);
    el('history').replaceChildren(...history.map(item => {
      const row = document.createElement('li');
      for (const text of [item.prompt, item.action]) {
        const span = document.createElement('span'); span.textContent = text; row.append(span);
      }
      return row;
    }));
  } catch (error) { el('message').textContent = `Không thực hiện được yêu cầu: ${error.message}`; }
  finally {
    busy = false; el('submit').disabled = !ready;
    el('submit').firstChild.textContent = 'Chạy SmolVLA ';
  }
});
status();
