"use strict";

const ROOT = "/outputs/eval";
const PATHS = {
  val: `${ROOT}/step9-final-val-full-20261109`,
  test: `${ROOT}/step9-final-test-full-20261109`,
  baseline: `${ROOT}/step9-baselines-20261109/metrics.json`,
};
const state = { metrics: {}, predictions: {}, baseline: null };
const $ = (id) => document.getElementById(id);
const percent = (value) => `${(100 * value).toFixed(2).replace(".", ",")}%`;
const number = (value, digits = 4) => Number(value).toFixed(digits).replace(".", ",");
const vector = (values, digits = null) => `[${values.map((v) => digits === null ? v : Number(v).toFixed(digits)).join(", ")}]`;

async function getJson(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${response.status} ${url}`);
  return response.json();
}

async function getJsonl(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${response.status} ${url}`);
  return (await response.text()).trim().split("\n").filter(Boolean).map(JSON.parse);
}

function renderCards() {
  const definitions = [
    ["Exact-action accuracy", "exact_action_accuracy", true],
    ["Zone accuracy", "zone_accuracy", true],
    ["Continuous MAE", "continuous_mae", false],
  ];
  $("metricCards").innerHTML = definitions.map(([label, key, higher]) => {
    const test = state.metrics.test.metrics.overall[key];
    const base = state.baseline.baselines.test.always_off[key === "continuous_mae" ? "mae" : key];
    const better = higher ? test >= base : test <= base;
    const format = key.includes("accuracy") ? percent : number;
    return `<article class="metric-card"><h3>${label}</h3><div class="metric-pair">
      <div><span>LoRA step 6.000</span><strong class="${better ? "good" : "bad"}">${format(test)}</strong></div>
      <div><span>Always OFF</span><strong>${format(base)}</strong></div>
    </div></article>`;
  }).join("");
}

function renderIntents() {
  const groups = state.metrics[$("intentSplit").value].metrics.by_intent;
  $("intentRows").innerHTML = Object.entries(groups).map(([intent, row]) => `<tr>
    <td><strong>${intent}</strong></td><td>${row.records}</td><td>${percent(row.exact_action_accuracy)}</td>
    <td>${percent(row.zone_accuracy)}</td><td>${number(row.continuous_mae)}</td><td>${row.out_of_range_components}</td>
  </tr>`).join("");
}

function renderPredictions() {
  const split = $("splitFilter").value;
  const language = $("languageFilter").value;
  const result = $("resultFilter").value;
  const filtered = state.predictions[split].filter((row) => {
    const exact = row.quantized_action.every((v, i) => v === row.target_action[i]);
    const outside = row.out_of_range.some(Boolean);
    return (language === "all" || row.language === language)
      && (result === "all" || result === "correct" && exact || result === "wrong" && !exact || result === "range" && outside);
  });
  $("predictionCount").textContent = `${filtered.length} / ${state.predictions[split].length} records phù hợp bộ lọc.`;
  $("predictionRows").innerHTML = filtered.slice(0, 100).map((row) => {
    const exact = row.quantized_action.every((v, i) => v === row.target_action[i]);
    const outside = row.out_of_range.some(Boolean);
    const status = outside ? ["range", "Ngoài miền"] : exact ? ["correct", "Đúng"] : ["wrong", "Sai"];
    return `<tr><td><code>${row.sample_id}</code></td><td>${row.language.toUpperCase()}</td><td>${row.intent}</td>
      <td><code>${vector(row.target_action)}</code></td><td><code>${vector(row.quantized_action)}</code></td>
      <td><code>${vector(row.continuous_action, 3)}</code></td><td><span class="pill ${status[0]}">${status[1]}</span></td></tr>`;
  }).join("");
}

async function load() {
  try {
    const [valMetrics, testMetrics, valPredictions, testPredictions, baseline] = await Promise.all([
      getJson(`${PATHS.val}/metrics.json`), getJson(`${PATHS.test}/metrics.json`),
      getJsonl(`${PATHS.val}/predictions.jsonl`), getJsonl(`${PATHS.test}/predictions.jsonl`), getJson(PATHS.baseline),
    ]);
    state.metrics = { val: valMetrics, test: testMetrics };
    state.predictions = { val: valPredictions, test: testPredictions };
    state.baseline = baseline;
    renderCards(); renderIntents(); renderPredictions();
    const model = testMetrics.metrics.overall;
    const base = baseline.baselines.test.always_off;
    if (model.exact_action_accuracy < base.exact_action_accuracy) {
      $("warning").hidden = false;
      $("warning").textContent = "Checkpoint 6.000 thấp hơn baseline always-OFF trên test. Viewer này trình bày kết quả thật, không chứng nhận model sẵn sàng triển khai.";
    }
    $("loadStatus").className = "status ready";
    $("loadStatus").textContent = "Đã tải 1.440 predictions";
  } catch (error) {
    $("loadStatus").className = "status error";
    $("loadStatus").textContent = "Không đọc được artifacts";
    $("warning").hidden = false;
    $("warning").textContent = `Hãy chạy scripts/view_step9.sh từ repo root. Chi tiết: ${error.message}`;
  }
}

$("intentSplit").addEventListener("change", renderIntents);
for (const id of ["splitFilter", "languageFilter", "resultFilter"]) $(id).addEventListener("change", renderPredictions);
load();
