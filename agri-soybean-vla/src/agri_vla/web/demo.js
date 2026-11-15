const state = { modelReady: false, images: [null, null, null], expected: null };
const $ = (id) => document.getElementById(id);

function createZones() {
  const root = $("zoneInputs");
  for (let index = 0; index < 3; index += 1) {
    const card = document.createElement("article");
    card.className = "zone-card";
    card.innerHTML = `
      <label class="drop-zone" for="zone-${index}">
        <input id="zone-${index}" type="file" accept="image/png,image/jpeg,image/webp">
        <span class="placeholder"><strong>Chọn ảnh khu vực ${index + 1}</strong>PNG, JPEG hoặc WebP · tối đa 8 MiB</span>
      </label>
      <div class="zone-meta">
        <strong>ZONE 0${index + 1}</strong>
        <label>Mức héo
          <select id="wilt-${index}">
            <option value="">Không rõ</option>
            <option value="0">0 · khỏe</option><option value="1">1 · nhẹ</option>
            <option value="2">2 · vừa</option><option value="3">3 · nặng</option>
            <option value="4">4 · rất nặng</option>
          </select>
        </label>
      </div>`;
    root.appendChild(card);
    const input = card.querySelector("input");
    input.addEventListener("change", () => readImage(index, input.files[0], card));
  }
}

function readImage(index, file, card) {
  clearExpected();
  const error = $("formError");
  error.hidden = true;
  if (!file) return;
  if (file.size > 8 * 1024 * 1024) {
    error.textContent = `Ảnh khu vực ${index + 1} lớn hơn 8 MiB.`;
    error.hidden = false;
    return;
  }
  const reader = new FileReader();
  reader.onload = () => {
    state.images[index] = reader.result;
    let image = card.querySelector("img");
    if (!image) {
      image = document.createElement("img");
      image.alt = `Ảnh khu vực ${index + 1}`;
      card.querySelector(".drop-zone").appendChild(image);
    }
    image.src = reader.result;
    card.querySelector(".placeholder").hidden = true;
  };
  reader.readAsDataURL(file);
}

async function pollStatus() {
  const badge = $("modelStatus");
  try {
    const response = await fetch("/api/status", { cache: "no-store" });
    const data = await response.json();
    badge.className = `status ${data.status}`;
    badge.querySelector("span").textContent = data.status === "ready" ? "Model sẵn sàng" :
      data.status === "error" ? "Model tải lỗi" : "Đang tải model";
    $("qualityWarning").textContent = data.error || data.quality_warning;
    $("qualityWarning").hidden = false;
    $("modelIdentity").textContent = `Checkpoint đang dùng: ${data.checkpoint}`;
    state.modelReady = data.status === "ready";
    $("runButton").disabled = !state.modelReady;
    if (data.status === "loading") setTimeout(pollStatus, 1500);
  } catch (error) {
    badge.className = "status error";
    badge.querySelector("span").textContent = "Backend mất kết nối";
    $("qualityWarning").textContent = error.message;
    $("qualityWarning").hidden = false;
  }
}

function showResults(data, elapsed, expected) {
  $("resultPanel").hidden = false;
  $("latency").textContent = `HTTP round trip ${(elapsed / 1000).toFixed(2)} s`;
  $("continuousAction").textContent = `[${data.continuous_action.map(x => x.toFixed(3)).join(", ")}]`;
  $("quantizedAction").textContent = `[${data.quantized_action.join(", ")}]`;
  $("targetComparison").textContent = expected ? `Target [${expected.join(', ')}] · ${expected.every((x,i) => x === data.quantized_action[i]) ? 'ĐÚNG cả ba vùng' : 'SAI — xem từng vùng bên dưới'}` : 'Ảnh/câu tự nhập: chưa có target để chấm.';
  $("valveResults").innerHTML = data.zones.map(zone => `
    <article class="valve ${zone.valve_open ? "open" : ""}">
      <div class="valve-head"><h3>Khu vực ${zone.zone}</h3><div class="valve-icon">✦</div></div>
      <div class="level">${zone.level} · ${zone.action}</div>
      <p>Van ${zone.valve_open ? "MỞ" : "ĐÓNG"}${zone.wilt_note === null ? "" : ` · chú thích héo ${zone.wilt_note}`}</p>
      ${zone.out_of_range ? '<p class="range-warning">Raw output nằm ngoài [0,3] và đã bị clip.</p>' : ""}
    </article>`).join("");
  $("resultPanel").scrollIntoView({ behavior: "smooth", block: "start" });
}

async function runInference() {
  const error = $("formError");
  error.hidden = true;
  const instruction = $("instruction").value.trim();
  if (!instruction || state.images.some(value => !value)) {
    error.textContent = "Hãy chọn đủ ba ảnh và nhập instruction.";
    error.hidden = false;
    return;
  }
  const button = $("runButton");
  button.disabled = true;
  button.firstChild.textContent = "Đang suy luận… ";
  const started = performance.now();
  const expected = state.expected ? [...state.expected] : null;
  try {
    const wiltNotes = [0, 1, 2].map(index => {
      const value = $(`wilt-${index}`).value;
      return value === "" ? null : Number(value);
    });
    const response = await fetch("/api/infer", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ instruction, images: state.images, wilt_notes: wiltNotes }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
    showResults(data, performance.now() - started, expected);
  } catch (caught) {
    error.textContent = caught.message;
    error.hidden = false;
  } finally {
    button.disabled = !state.modelReady;
    button.firstChild.textContent = "Chạy SmolVLA ";
  }
}

createZones();
function clearExpected() {
  state.expected = null;
  $("expectedTarget").textContent = 'Input đã thay đổi; chưa có target để chấm.';
  $("samplePicker").value = '';
  $("resultPanel").hidden = true;
}
async function loadExamples() {
  try {
    const response = await fetch('/api/examples');
    const data = await response.json();
    data.examples.forEach((row, i) => {
      const option = document.createElement('option'); option.value = String(i); option.textContent = row.label;
      $("samplePicker").appendChild(option);
    });
    $("samplePicker").addEventListener('change', () => {
      if ($("samplePicker").value === '') { clearExpected(); return; }
      const row = data.examples[Number($("samplePicker").value)];
      state.images = [...row.images]; state.expected = [...row.target_action];
      $("instruction").value = row.instruction;
      row.images.forEach((src, i) => {
        const card = $("zoneInputs").children[i]; let img = card.querySelector('img');
        if (!img) { img = document.createElement('img'); img.alt = `Ảnh vùng ${i+1}`; card.querySelector('.drop-zone').appendChild(img); }
        img.src = src; card.querySelector('.placeholder').hidden = true;
        $(`zone-${i}`).value = ''; $(`wilt-${i}`).value = String(row.wilt_notes[i]);
      });
      $("expectedTarget").textContent = `Đáp án mẫu train: [${row.target_action.join(', ')}]. Target chỉ hiển thị để đối chiếu, không gửi vào model.`;
      $("resultPanel").hidden = true;
    });
  } catch (error) { $("expectedTarget").textContent = 'Chưa tải được mẫu; bạn vẫn có thể upload ảnh.'; }
}
$("instruction").addEventListener('input', clearExpected);
loadExamples();
document.querySelectorAll("[data-prompt]").forEach(button => {
  button.addEventListener("click", () => { clearExpected(); $("instruction").value = button.dataset.prompt; });
});
$("runButton").addEventListener("click", runInference);
pollStatus();
