"use strict";
const $ = (id) => document.getElementById(id);
const fields = ["q", "split", "language", "wording", "intent", "wilt", "group", "action", "kind", "eligibility", "page_size"];
const names = {AUTO: "Tưới theo ngưỡng héo", WORST_ONLY: "Chỉ vùng héo nhất · giữ mọi vùng hòa max", ZONE_ONLY: "Chỉ vùng chỉ định", CAP_LEVEL: "Giới hạn mức tưới", EXCLUDE_ZONE: "Loại một vùng", STOP_ALL: "Dừng tất cả", WATER_ALL: "Tưới cả 3 theo mức chỉ định · không xét ngưỡng"};
const kindNames = {balanced: "Tổ hợp bổ sung", contrastive_base: "Cùng ảnh · khác lệnh", permutation: "Hoán vị zone", all_off: "Ba vùng wilt 0", all_high: "Ba vùng wilt 4", tie: "Hòa mức héo cao nhất", below_threshold: "Dưới ngưỡng tưới", operator_override_all: "Người dùng chỉ định mức cho cả 3 vùng"};
const folderNames = {manifests: "Dataset JSONL & báo cáo QA", images: "Ảnh nguồn nguyên gốc", raw: "Nguồn tải về · CSV & ZIP", "source-splits": "Nguồn đã chia train / val / test", "source-report": "Audit nguồn & quarantine"};
const levelNames = ["OFF", "LOW", "MEDIUM", "HIGH"];
const esc = (value) => String(value ?? "").replace(/[&<>"']/g, (c) => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const fmt = (value) => Number(value).toLocaleString("vi-VN");
const encoded = (value) => encodeURIComponent(value);
const emptyState = (view = "samples") => ({view, page: "1", page_size: "12"});
let state = {...emptyState(), ...Object.fromEntries(new URLSearchParams(location.search))};
let summary, pages = 1, controller, searchTimer, toastTimer, detailRequest = 0;
if (!["samples", "sources", "files", "guide"].includes(state.view)) state.view = "samples";

async function api(path, signal) {
  const response = await fetch(path, {signal});
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || `Lỗi HTTP ${response.status}`);
  return data;
}
function toast(message) {
  $("toast").textContent = message;
  $("toast").hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { $("toast").hidden = true; }, 3000);
}
async function copy(value) {
  try {
    await navigator.clipboard.writeText(value);
    toast("Đã copy. Có thể dán vào thanh đường dẫn hoặc terminal.");
  } catch {
    // Text remains selectable when a browser does not grant clipboard access.
    window.prompt("Copy đường dẫn / link dưới đây:", value);
  }
}
function writeURL(push = false) {
  const params = new URLSearchParams();
  for (const key of ["view", ...fields, "page", "scene_id", "case_id", "source_id", "sample"]) {
    if (state[key] && !(key === "page" && state[key] === "1") && !(key === "page_size" && state[key] === "12")) params.set(key, state[key]);
  }
  const url = "/?" + params.toString();
  if (push && url !== location.pathname + location.search) history.pushState(null, "", url);
  else history.replaceState(null, "", url);
}
function syncControls() {
  for (const key of fields) $(key).value = state[key] || (key === "page_size" ? "12" : "");
  $("page").value = state.page || "1";
  const isSource = state.view === "sources";
  document.querySelectorAll("[data-samples-only]").forEach((node) => { node.hidden = isSource; });
  document.querySelectorAll("[data-sources-only]").forEach((node) => { node.hidden = !isSource; });
  $("wilt5").hidden = !isSource;
  $("jumpForm").hidden = isSource;
  document.querySelectorAll("[data-view]").forEach((button) => { button.classList.toggle("active", button.dataset.view === state.view); });
  $("browserPanel").hidden = !["samples", "sources"].includes(state.view);
  $("filesPanel").hidden = state.view !== "files";
  $("guidePanel").hidden = state.view !== "guide";
  const relation = [state.scene_id && `Cùng bộ ba ảnh: ${state.scene_id}`, state.case_id && `Cùng case (bản dịch / wording): ${state.case_id}`, state.source_id && `Dùng ảnh gốc: ${state.source_id}`].filter(Boolean);
  $("relation").hidden = relation.length === 0;
  $("relation").textContent = relation.join(" · ");
  $("cards").classList.toggle("source-grid", isSource);
}
function zone(row, index) {
  const source = row.source_image_ids[index], level = row.target_action[index];
  return `<div class="zone"><div class="zone-top"><span>VÙNG ${index + 1}</span><span class="level level-${level}">${level} · ${levelNames[level]}</span></div><a class="zone-image" href="/original/${encoded(source)}" target="_blank" rel="noopener" title="Mở ảnh gốc ${esc(source)}"><img loading="lazy" src="/thumbnail/${encoded(source)}" alt="Vùng ${index + 1}: ${esc(source)}"></a><div class="zone-meta"><span class="source-name" title="${esc(source)}">${esc(source)}</span><span>Teacher wilt ${row.teacher_annotations.wilt_scores[index]} · ${esc(row.source_groups[index])}</span><br><a href="/files/images/?q=${encoded(source)}" target="_blank" rel="noopener">Thư mục gốc ↗</a></div></div>`;
}
function sampleCard(row) {
  return `<article class="sample-card" data-sample="${esc(row.sample_id)}"><div class="card-head"><span class="ordinal">#${fmt(row.ordinal)} · dòng ${row.manifest_line} / ${row.split}.jsonl</span><span class="badge ${row.split}">${row.split.toUpperCase()}</span><span class="badge">${row.language.toUpperCase()}</span><span class="badge ${row.wording_partition}">${row.wording_partition}</span></div><div class="card-content"><p class="instruction">${esc(row.instruction)}</p><div class="intent-label">${esc(names[row.intent])} · ${esc(kindNames[row.generation_metadata.kind])}${Object.keys(row.intent_params).length ? " · " + esc(JSON.stringify(row.intent_params)) : ""}</div><div class="zones">${[0,1,2].map((i) => zone(row,i)).join("")}</div><div class="card-actions"><span class="target-vector">Target [${row.target_action.join(", ")}]</span><button data-detail="${esc(row.sample_id)}">Chi tiết</button><button class="secondary" data-related-case="${esc(row.case_id)}">Bản dịch & wording</button><button class="secondary" data-related-scene="${esc(row.scene_id)}">Cùng ảnh · lệnh khác</button></div></div></article>`;
}
function sourceCard(row) {
  const status = !row.eligible ? "Quarantine · ngoài v0" : row.record_uses ? "Có trong dataset" : "Eligible · chưa được chọn";
  return `<article class="source-card"><a href="${row.original_url}" target="_blank" rel="noopener"><img loading="lazy" src="${row.thumbnail_url}" alt="${esc(row.source_image_id)}"></a><div class="source-content"><span class="badge ${row.split || "test"}">${esc(row.split || "EXCLUDED")}</span> <span class="badge">Nhãn ${row.annotation}</span><code>${esc(row.source_image_id)}</code><p>${esc(status)}<br>Nhóm ${esc(row.group)} · ${row.width}×${row.height}<br>Dùng trong ${fmt(row.record_uses)} record</p><div class="card-actions"><a class="button secondary" href="${row.original_url}" target="_blank" rel="noopener">Ảnh gốc ↗</a><a class="button secondary" href="${row.folder_url}" target="_blank" rel="noopener">Thư mục ↗</a>${row.record_uses ? `<button data-source="${esc(row.source_image_id)}">Xem các mẫu dùng ảnh</button>` : ""}</div></div></article>`;
}
async function load(push = false) {
  controller?.abort();
  controller = new AbortController();
  const signal = controller.signal;
  syncControls();
  writeURL(push);
  if (!["samples", "sources"].includes(state.view)) return;
  $("error").hidden = true;
  $("cards").classList.add("loading");
  $("resultCount").textContent = "Đang đọc mẫu…";
  const query = new URLSearchParams({page: state.page || "1", page_size: state.page_size || "12"});
  const keys = state.view === "samples" ? ["q","split","language","wording","intent","wilt","group","action","kind","scene_id","case_id","source_id"] : ["q","split","wilt","group","eligibility"];
  for (const key of keys) if (state[key]) query.set(key,state[key]);
  try {
    const result = await api(`/api/${state.view}?${query}`,signal);
    if (signal.aborted) return;
    state.page = String(result.page);
    pages = result.pages;
    $("page").value = result.page;
    $("page").max = pages;
    $("pageTotal").textContent = "/ " + fmt(pages);
    const start = result.total ? (result.page - 1) * result.page_size + 1 : 0;
    const end = Math.min(result.page * result.page_size,result.total);
    $("resultCount").textContent = `${fmt(start)}–${fmt(end)} / ${fmt(result.total)} ${state.view === "samples" ? "mẫu" : "ảnh nguồn"}`;
    $("cards").innerHTML = result.items.map(state.view === "samples" ? sampleCard : sourceCard).join("");
    $("empty").hidden = result.total !== 0;
    for (const key of ["first","prev"]) $(key).disabled = result.page <= 1;
    for (const key of ["next","last"]) $(key).disabled = result.page >= pages;
    writeURL(false);
  } catch (error) {
    if (error.name !== "AbortError") {
      $("error").textContent = `Không tải được dữ liệu: ${error.message}. Kiểm tra server còn chạy, hoặc nhấn Xóa bộ lọc.`;
      $("error").hidden = false;
      $("cards").innerHTML = "";
      $("resultCount").textContent = "Chưa tải được kết quả";
    }
  } finally {
    if (!signal.aborted) $("cards").classList.remove("loading");
  }
}
function related(key,value) {
  if ($("detail").open) $("detail").close();
  state = {...emptyState(), page_size: state.page_size, [key]: value};
  load(true);
  $("filters").scrollIntoView({behavior:"smooth",block:"start"});
}
async function showDetail(id, index = null) {
  const current = ++detailRequest;
  try {
    const data = await api("/api/sample?" + (index ? `index=${encoded(index)}` : `id=${encoded(id)}`));
    if (current !== detailRequest) return;
    const row = data.record;
    $("detailTitle").textContent = `#${fmt(data.ordinal)} · ${row.split.toUpperCase()} · ${row.language.toUpperCase()} · ${row.wording_partition}`;
    const sourceInfo = data.sources.map((source, i) => `<section class="detail-source"><h3>Vùng ${i+1} · teacher wilt ${row.teacher_annotations.wilt_scores[i]}</h3><p>${esc(source.source_image_id)}<br>Camera ${esc(source.metadata.camera_id)} / plot ${esc(source.metadata.plot_id)} · ${esc(source.timestamp || "Không có timestamp")} (timezone nguồn chưa xác định)</p><code class="path">${esc(source.path)}</code><a class="button secondary" href="${source.original_url}" target="_blank" rel="noopener">Mở ảnh gốc ↗</a> <a class="button secondary" href="${source.folder_url}" target="_blank" rel="noopener">Thư mục gốc ↗</a> <button class="secondary" data-copy="${esc(source.path)}">Copy đường dẫn</button><p>SHA256: ${esc(source.sha256)}</p></section>`).join("");
    const translationRows = data.related.map((r) => `<tr><td>${r.language.toUpperCase()} · ${r.wording_partition}</td><td>${esc(r.instruction)}</td><td>[${r.target_action.join(",")}]</td><td><button class="secondary" data-detail="${esc(r.sample_id)}">Xem</button></td></tr>`).join("");
    $("detailBody").innerHTML = `<p class="detail-instruction">${esc(row.instruction)}</p><p>${esc(names[row.intent])} · <code>${esc(JSON.stringify(row.intent_params))}</code> · <b>Target [${row.target_action.join(", ")}]</b></p><div class="zones">${[0,1,2].map((i)=>zone(row,i)).join("")}</div><div class="card-actions"><button data-related-case="${esc(row.case_id)}">Bản dịch & wording</button><button class="secondary" data-related-scene="${esc(row.scene_id)}">Cùng ảnh · lệnh khác</button><button class="secondary" data-copy-link="${esc(row.sample_id)}">Copy link mẫu</button><a class="button secondary" href="/api/sample?id=${encoded(row.sample_id)}&download=1">Tải JSON mẫu</a></div><h3 class="detail-source">Các bản dịch / cách diễn đạt trong cùng case</h3><table class="detail-table"><thead><tr><th>Ngôn ngữ</th><th>Instruction</th><th>Target</th><th></th></tr></thead><tbody>${translationRows}</tbody></table><details><summary>Ảnh gốc, đường dẫn local & checksum</summary>${sourceInfo}</details><details><summary>Toàn bộ record JSON & provenance</summary><pre>${esc(JSON.stringify(row,null,2))}</pre></details><p class="note">Teacher wilt và intent/params là metadata để sinh nhãn. Model chỉ nhận ba ảnh, instruction và state [0,0,0] giả định mọi van ban đầu đóng. Target là proxy, chưa phải lượng nước hiệu chuẩn.</p>`;
    state.sample = row.sample_id;
    writeURL(false);
    if (!$("detail").open) $("detail").showModal();
    $("detail").scrollTop = 0;
  } catch (error) {
    toast(`Không mở được mẫu: ${error.message}`);
  }
}
function renderSummary() {
  $("metrics").innerHTML = [[summary.records,"Mẫu tổng hợp (record)","Train / val / test: " + Object.values(summary.splits).map(fmt).join(" / ")],[summary.scenes,"Bộ ba ảnh có thứ tự","Có thể dùng chung cho nhiều câu lệnh"],[summary.used_source_images,"Ảnh gốc dùng trong dataset","Tổ hợp / bản dịch không tăng ảnh độc lập"],[summary.raw_source_images,"Ảnh nguồn để đối chiếu",`${summary.quarantined} ảnh nhãn 5 nằm ngoài v0`]].map(([n,title,note])=>`<div class="metric"><strong>${fmt(n)}</strong><span>${esc(title)}</span><small>${esc(note)}</small></div>`).join("");
  $("samplesCount").textContent = fmt(summary.records);
  $("sourcesCount").textContent = fmt(summary.raw_source_images);
  $("verified").textContent = `✓ ${summary.dataset_revision || "Dataset"} · QA PASS · đã kiểm tra lock & ảnh`;
  $("jump").max = summary.records;
  for (const [id, values] of Object.entries(summary.options)) {
    for (const value of values) {
      const option = document.createElement("option");
      option.value = value;
      option.textContent = id === "intent" ? `${value} · ${names[value]}` : id === "kind" ? kindNames[value] : value;
      $(id).appendChild(option);
    }
  }
  $("folderCards").innerHTML = summary.folders.map((folder)=>`<article class="folder-card"><h3>${esc(folderNames[folder.key])}</h3><p>Danh sách file thật, đọc từ đường dẫn local sau:</p><code class="path">${esc(folder.path)}</code><a class="button" href="${folder.url}" target="_blank" rel="noopener">Mở thư mục ↗</a> <button class="secondary" data-copy="${esc(folder.path)}">Copy đường dẫn</button></article>`).join("");
}

document.addEventListener("click", (event) => {
  const button = event.target.closest("button");
  if (!button) return;
  if (button.dataset.view) { state = emptyState(button.dataset.view); load(true); }
  if (button.dataset.detail) showDetail(button.dataset.detail);
  if (button.dataset.relatedCase) related("case_id",button.dataset.relatedCase);
  if (button.dataset.relatedScene) related("scene_id",button.dataset.relatedScene);
  if (button.dataset.source) related("source_id",button.dataset.source);
  if (button.dataset.copy) copy(button.dataset.copy);
  if (button.dataset.copyLink) copy(location.origin + "/?view=samples&sample=" + encoded(button.dataset.copyLink));
});
$("filters").addEventListener("submit",(event)=>event.preventDefault());
$("q").addEventListener("input",()=>{clearTimeout(searchTimer);searchTimer=setTimeout(()=>{state.q=$("q").value;state.page="1";load(true);},250);});
for (const key of fields.filter((key)=>key!=="q")) $(key).addEventListener("change",()=>{clearTimeout(searchTimer);state[key]=$(key).value;state.page="1";delete state.sample;load(true);});
$("reset").addEventListener("click",()=>{clearTimeout(searchTimer);state={...emptyState(state.view),page_size:state.page_size};load(true);});
function go(page) { state.page=String(page);load(true);$("resultCount").scrollIntoView({behavior:"smooth",block:"start"}); }
$("first").addEventListener("click",()=>go(1));
$("prev").addEventListener("click",()=>go(Math.max(1,Number(state.page)-1)));
$("next").addEventListener("click",()=>go(Math.min(pages,Number(state.page)+1)));
$("last").addEventListener("click",()=>go(pages));
$("pageForm").addEventListener("submit",(event)=>{event.preventDefault();go(Math.max(1,Math.min(pages,Number($("page").value)||1)));});
$("jumpForm").addEventListener("submit",(event)=>{event.preventDefault();showDetail(null,$("jump").value);});
$("closeDetail").addEventListener("click",()=>{detailRequest++;$("detail").close();});
$("detail").addEventListener("close",()=>{delete state.sample;writeURL(false);});
window.addEventListener("popstate",()=>{state={...emptyState(),...Object.fromEntries(new URLSearchParams(location.search))};load(false);if(state.sample)showDetail(state.sample);});
window.addEventListener("error",(event)=>{if(event.target instanceof HTMLImageElement) event.target.alt="Ảnh không tải được — kiểm tra server và file gốc";},true);
(async()=>{
  try {
    summary=await api("/api/summary");
    renderSummary();
    await load(false);
    if(state.sample) await showDetail(state.sample);
  } catch(error) {
    $("verified").textContent="Không kết nối được server";
    $("error").hidden=false;
    $("error").textContent=`${error.message}. Hãy chạy lại scripts/browse.sh và tải lại trang.`;
  }
})();
