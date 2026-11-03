import { createPredictor } from "./model.js";
import { barChart, histogram, hideTip } from "./charts.js";

const FIELDS = [
  { key: "length", label: "Length", hint: "longest shell measurement", kind: "mm", slider: [0.05, 0.85], step: 0.005, max: 1.0 },
  { key: "diameter", label: "Diameter", hint: "perpendicular to length", kind: "mm", slider: [0.04, 0.7], step: 0.005, max: 1.0 },
  { key: "height", label: "Height", hint: "with meat in shell", kind: "mm", slider: [0, 0.3], step: 0.005, max: 1.2 },
  { key: "whole_weight", label: "Whole weight", hint: "whole abalone", kind: "g", slider: [0, 3], step: 0.0005, max: 4 },
  { key: "shucked_weight", label: "Shucked weight", hint: "weight of meat", kind: "g", slider: [0, 1.6], step: 0.0005, max: 2 },
  { key: "viscera_weight", label: "Viscera weight", hint: "gut weight after bleeding", kind: "g", slider: [0, 0.8], step: 0.0005, max: 1 },
  { key: "shell_weight", label: "Shell weight", hint: "after drying", kind: "g", slider: [0, 1.1], step: 0.0005, max: 1.5 },
];
const RAW = new Set(["sex", ...FIELDS.map((f) => f.key)]);
const RAW_FEATURE = (name) => RAW.has(name) || name.startsWith("sex_");

const $ = (sel) => document.querySelector(sel);
const fmt = (v, d) => Number(v).toFixed(d);

async function loadJSON(path) {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`Failed to load ${path}: ${res.status}`);
  return res.json();
}

// ---------- theme ----------
function initTheme() {
  const root = document.documentElement;
  $("#theme-toggle").addEventListener("click", () => {
    const current = root.dataset.theme ||
      (window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
    const next = current === "dark" ? "light" : "dark";
    root.dataset.theme = next;
    try { localStorage.setItem("abalone-theme", next); } catch (e) { /* ignore */ }
  });
}

// ---------- predictor ----------
function buildFields(state, onChange) {
  const host = $("#fields");
  for (const f of FIELDS) {
    const id = `in-${f.key}`;
    const max = f.max ?? f.slider[1];
    const wrap = document.createElement("div");
    wrap.className = "field";
    wrap.innerHTML = `
      <div class="field-top">
        <label for="${id}">${f.label} <span class="muted tiny">${f.hint}</span></label>
        <span class="real" id="${id}-real"></span>
      </div>
      <div class="field-row">
        <input type="range" id="${id}-range" min="${f.slider[0]}" max="${f.slider[1]}" step="${f.step}"
               aria-label="${f.label} slider">
        <input type="number" id="${id}" min="0" max="${max}" step="${f.step}" inputmode="decimal" required>
      </div>`;
    host.appendChild(wrap);
    const range = wrap.querySelector(`#${id}-range`);
    const num = wrap.querySelector(`#${id}`);
    range.addEventListener("input", () => {
      num.value = range.value;
      num.removeAttribute("aria-invalid");
      state.values[f.key] = Number(range.value);
      onChange(true);
    });
    num.addEventListener("input", () => {
      const v = num.value.trim() === "" ? NaN : Number(num.value);
      if (!Number.isFinite(v) || v < 0 || v > max) {
        num.setAttribute("aria-invalid", "true");
        return;
      }
      num.removeAttribute("aria-invalid");
      range.value = String(Math.min(Math.max(v, f.slider[0]), f.slider[1]));
      state.values[f.key] = v;
      onChange(true);
    });
  }
  document.querySelectorAll('input[name="sex"]').forEach((r) =>
    r.addEventListener("change", () => {
      state.values.sex = r.value;
      onChange(true);
    }));
}

function setInputs(values) {
  for (const f of FIELDS) {
    const v = values[f.key];
    $(`#in-${f.key}`).value = String(v);
    $(`#in-${f.key}`).removeAttribute("aria-invalid");
    $(`#in-${f.key}-range`).value = String(Math.min(Math.max(v, f.slider[0]), f.slider[1]));
  }
  const radio = document.querySelector(`input[name="sex"][value="${values.sex}"]`);
  if (radio) radio.checked = true;
}

function warnings(v) {
  const out = [];
  if (v.height <= 0) out.push("Height is 0; the model clips it to the training range.");
  for (const k of ["shucked_weight", "viscera_weight", "shell_weight"]) {
    if (v[k] > v.whole_weight) {
      out.push(`${FIELDS.find((f) => f.key === k).label} exceeds the whole weight, which is physically implausible.`);
    }
  }
  if (v.diameter > v.length) out.push("Diameter is larger than length; real shells are longer than they are wide.");
  return out;
}

function initPredictor(model, insights) {
  const state = { values: {}, preset: null };
  const presets = insights.presets;
  const bins = [];
  const hist = insights.ring_histogram;
  const maxRing = Math.max(...Object.keys(hist).map(Number));
  for (let r = 1; r <= maxRing; r++) bins.push({ x: r, count: hist[String(r)] ?? 0 });

  const chipHost = $("#presets");
  presets.forEach((p, i) => {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "chip";
    b.textContent = p.label;
    b.title = `UCI row ${p.uci_row}: ${p.rings} rings`;
    b.setAttribute("aria-pressed", "false");
    b.addEventListener("click", () => applyPreset(i));
    chipHost.appendChild(b);
  });

  function matchesPreset(p) {
    return p && p.sex === state.values.sex && FIELDS.every((f) => Math.abs(p[f.key] - state.values[f.key]) < 1e-12);
  }

  function render() {
    const v = state.values;
    const t0 = performance.now();
    const out = model.predict(v);
    const ms = performance.now() - t0;

    const ringsEl = $("#pred-rings");
    ringsEl.textContent = fmt(out.rings, 1);
    ringsEl.dataset.rings = String(out.rings);
    $("#pred-age").textContent = fmt(out.age, 1);

    const p = state.preset !== null ? presets[state.preset] : null;
    const actual = $("#pred-actual");
    if (matchesPreset(p)) {
      const err = out.rings - p.rings;
      actual.hidden = false;
      actual.innerHTML = `Actual: <strong>${p.rings} rings</strong> (${fmt(p.rings + 1.5, 1)} years) &middot; ` +
        `error ${err >= 0 ? "+" : "&minus;"}${fmt(Math.abs(err), 1)} rings &middot; UCI row ${p.uci_row}, hold-out`;
    } else {
      actual.hidden = true;
    }
    document.querySelectorAll(".chip").forEach((c, i) =>
      c.setAttribute("aria-pressed", String(matchesPreset(presets[i]))));

    for (const f of FIELDS) {
      const real = v[f.key] * 200;
      $(`#in-${f.key}-real`).textContent = f.kind === "mm" ? `${fmt(real, 0)} mm` : `${fmt(real, 1)} g`;
    }

    const w = warnings(v);
    const warn = $("#input-warning");
    warn.hidden = w.length === 0;
    warn.textContent = w.join(" ");

    const highlight = Math.min(Math.max(Math.round(out.rings), 1), maxRing);
    histogram($("#hist"), bins, highlight, { ariaLabel: `Ring count distribution; prediction ${fmt(out.rings, 1)} rings` });

    const rows = model.featureNames.map((name) => {
      const val = out.features[name];
      return `<tr class="${RAW_FEATURE(name) ? "" : "eng"}"><td>${name}</td><td>${
        Number.isInteger(val) ? val : val.toPrecision(5)}</td></tr>`;
    });
    $("#feat-table").innerHTML = `<tbody>${rows.join("")}</tbody>`;
    $("#pred-meta").textContent =
      `${model.nTrees} trees evaluated in ${ms < 1 ? "<1" : fmt(ms, 1)} ms on this device. * = engineered feature.`;
  }

  function applyPreset(i) {
    const p = presets[i];
    state.preset = i;
    state.values = { sex: p.sex };
    for (const f of FIELDS) state.values[f.key] = p[f.key];
    setInputs(state.values);
    render();
  }

  buildFields(state, () => render());
  applyPreset(Math.min(2, presets.length - 1));
  window.addEventListener("resize", () => render(), { passive: true });
}

// ---------- results ----------
function initResults(results, insights) {
  const ours = {
    ...results.tuned_xgboost,
    rmse: insights.cv?.rmse_mean,
    mae: insights.cv?.mae_mean,
    r2: insights.cv?.r2_mean,
    ours: true,
  };
  const base = results.cv.find((r) => r.model === "Baseline").rmsle;
  ours.vs_baseline = -100 * (base - ours.rmsle) / base;
  const all = [...results.cv, ours].sort((a, b) => a.rmsle - b.rmsle);
  const ensembles = new Set(["Blend", "Stacking"]);
  const short = { "Linear Regression": "Linear", "Random Forest": "Rand. Forest", "XGBoost (tuned)": "XGB tuned" };

  const color = (r) => (r.ours ? "var(--series-1)" : ensembles.has(r.model) ? "var(--series-ens)" : "var(--neutral-bar)");
  const draw = () => barChart($("#cmp-chart"), all.map((r) => ({
    label: r.model,
    shortLabel: short[r.model] ?? r.model,
    value: r.rmsle,
    err: r.std,
    color: color(r),
    strong: r.ours,
    tip: `<strong>${r.model}</strong>${r.detail ? `${r.detail}<br>` : ""}CV RMSLE ${fmt(r.rmsle, 4)} &plusmn; ${fmt(r.std, 4)}` +
      (r.vs_baseline != null ? `<br>${fmt(r.vs_baseline, 1)}% vs. baseline` : ""),
  })), {
    format: (v) => fmt(v, 4),
    labelWidth: 150,
    ariaLabel: "Cross-validated RMSLE by model; lower is better",
  });
  draw();
  window.addEventListener("resize", draw, { passive: true });

  const best = Math.min(...results.cv.map((r) => r.rmsle));
  const num = (v, d) => (v == null ? "&ndash;" : fmt(v, d));
  const body = all.map((r) => `
    <tr class="${r.ours ? "ours" : ""} ${r.rmsle === best ? "best" : ""}">
      <td>${r.model}${r.ours ? '<span class="tag">this page</span>' : ""}${r.detail ? `<span class="detail">${r.detail}</span>` : ""}</td>
      <td>${fmt(r.rmsle, 4)} &plusmn; ${fmt(r.std, 4)}</td>
      <td>${num(r.rmse, 3)}</td><td>${num(r.mae, 3)}</td><td>${num(r.r2, 3)}</td>
      <td>${r.vs_baseline == null ? "&mdash;" : `${r.vs_baseline < 0 ? "&minus;" : "+"}${fmt(Math.abs(r.vs_baseline), 1)}%`}</td>
    </tr>`).join("");
  $("#cv-table").insertAdjacentHTML("beforeend", `
    <thead><tr><th scope="col">Model</th><th scope="col">CV RMSLE (mean &plusmn; std)</th><th scope="col">RMSE (rings)</th>
      <th scope="col">MAE</th><th scope="col">R&sup2;</th><th scope="col">vs. baseline</th></tr></thead>
    <tbody>${body}</tbody>`);

  const h = insights.holdout;
  const hb = results.holdout_blend;
  $("#holdout-note").innerHTML =
    `<strong>Hold-out (836 unseen rows).</strong> Blend: RMSLE ${fmt(hb.rmsle, 4)}, RMSE ${fmt(hb.rmse, 2)}, MAE ${fmt(hb.mae, 2)}, R&sup2; ${fmt(hb.r2, 3)}. ` +
    `Tuned XGBoost on this page: RMSLE ${fmt(h.rmsle, 4)}, RMSE ${fmt(h.rmse, 2)}, MAE ${fmt(h.mae, 2)}, R&sup2; ${fmt(h.r2, 3)} ` +
    `(mean baseline ${fmt(h.baseline_rmsle, 4)}). It matches the ensembles with a single 300-tree model, which is why it was chosen for the browser. ` +
    `RMSE/MAE/R&sup2; for the tuned row come from the export script's own 5&times;2 CV run.`;
}

function initShap(insights) {
  const vals = insights.shap.values;
  const draw = () => barChart($("#shap-chart"), vals.map((r) => ({
    label: r.feature,
    value: r.mean_abs_shap,
    color: RAW_FEATURE(r.feature) ? "var(--series-1)" : "var(--series-2)",
    tip: `<strong>${r.feature}</strong>${RAW_FEATURE(r.feature) ? "raw measurement" : "engineered feature"}<br>` +
      `mean |SHAP| ${fmt(r.mean_abs_shap, 4)} log-rings<br>&asymp; &times;${fmt(Math.exp(r.mean_abs_shap), 3)} on the ring count`,
  })), {
    rowHeight: 26,
    barHeight: 16,
    labelWidth: 160,
    format: (v) => fmt(v, 3),
    ariaLabel: "Mean absolute SHAP value per feature",
  });
  draw();
  window.addEventListener("resize", draw, { passive: true });
}

function initStats(insights, results) {
  const h = insights.holdout;
  const set = (k, v) => { $(`[data-stat="${k}"]`).innerHTML = v; };
  set("holdout", fmt(h.rmsle, 4));
  set("improvement", `&minus;${fmt(100 * (h.baseline_rmsle - h.rmsle) / h.baseline_rmsle, 1)}%`);
  set("cv", `${fmt(results.tuned_xgboost.rmsle, 4)}`);
  set("size", `${Math.round(insights.model.model_json_bytes / 1024)} kB`);
  $("#parity-text").insertAdjacentHTML("beforeend",
    ` Last export: ${insights.model.n_trees} trees, ${insights.model.n_features} features, trained on ` +
    `${insights.model.trained_on_rows.toLocaleString()} rows.`);
}

async function main() {
  initTheme();
  document.addEventListener("scroll", hideTip, { passive: true });
  const [spec, insights, results] = await Promise.all([
    loadJSON("data/model.json"),
    loadJSON("data/insights.json"),
    loadJSON("data/results.json"),
  ]);
  const model = createPredictor(spec);
  initStats(insights, results);
  initPredictor(model, insights);
  initResults(results, insights);
  initShap(insights);
  document.body.dataset.ready = "true";
}

main().catch((err) => {
  console.error(err);
  const box = document.createElement("p");
  box.className = "warn wrap";
  box.textContent = `Could not load the model: ${err.message}`;
  document.querySelector("main").prepend(box);
});
