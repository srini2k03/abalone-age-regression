// Small hand-drawn SVG charts (no dependencies): horizontal bars and a column histogram.

const NS = "http://www.w3.org/2000/svg";
const tooltip = () => document.getElementById("tooltip");

function el(name, attrs = {}, parent) {
  const node = document.createElementNS(NS, name);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  if (parent) parent.appendChild(node);
  return node;
}

function niceStep(max, target) {
  const raw = max / target;
  const pow = 10 ** Math.floor(Math.log10(raw));
  const n = raw / pow;
  return (n < 1.5 ? 1 : n < 3 ? 2 : n < 7 ? 5 : 10) * pow;
}

/** Bar with a 4px rounded data end and a square baseline end (horizontal). */
function hBarPath(x0, y, w, h, r = 4) {
  if (w <= 0) return "";
  const rr = Math.min(r, w, h / 2);
  return `M${x0},${y}H${x0 + w - rr}Q${x0 + w},${y} ${x0 + w},${y + rr}V${y + h - rr}` +
    `Q${x0 + w},${y + h} ${x0 + w - rr},${y + h}H${x0}Z`;
}

/** Column with a rounded top and a square baseline. */
function vBarPath(x, y0, w, h, r = 3) {
  if (h <= 0) return "";
  const rr = Math.min(r, h, w / 2);
  const top = y0 - h;
  return `M${x},${y0}V${top + rr}Q${x},${top} ${x + rr},${top}H${x + w - rr}` +
    `Q${x + w},${top} ${x + w},${top + rr}V${y0}Z`;
}

export function showTip(evt, html) {
  const tip = tooltip();
  tip.innerHTML = html;
  tip.hidden = false;
  const pad = 14;
  const { innerWidth: W, innerHeight: H } = window;
  const r = tip.getBoundingClientRect();
  let x = evt.clientX + pad;
  let y = evt.clientY + pad;
  if (x + r.width > W - 8) x = evt.clientX - r.width - pad;
  if (y + r.height > H - 8) y = evt.clientY - r.height - pad;
  tip.style.left = `${Math.max(8, x)}px`;
  tip.style.top = `${Math.max(8, y)}px`;
}
export function hideTip() {
  tooltip().hidden = true;
}

/**
 * Horizontal bar chart.
 * rows: [{label, value, err?, color, tip, strong?}]
 */
export function barChart(container, rows, opts = {}) {
  const width = Math.max(container.clientWidth, 280);
  const narrow = width < 520;
  const rowH = opts.rowHeight ?? 30;
  const barH = Math.min(opts.barHeight ?? 18, 24);
  const labelW = Math.min(opts.labelWidth ?? 170, narrow ? 118 : 220);
  const right = narrow ? 52 : 64;
  const top = 6;
  const axisH = 26;
  const plotW = width - labelW - right;
  const height = top + rows.length * rowH + axisH;
  const max = opts.max ?? Math.max(...rows.map((r) => r.value + (r.err ?? 0)));
  const step = niceStep(max, narrow ? 3 : 5);
  const domain = Math.ceil(max / step) * step;
  const sx = (v) => labelW + (v / domain) * plotW;
  const fmt = opts.format ?? ((v) => v.toFixed(3));

  container.innerHTML = "";
  const svg = el("svg", {
    viewBox: `0 0 ${width} ${height}`,
    role: "img",
    "aria-label": opts.ariaLabel ?? "Bar chart",
  }, container);

  // grid + axis ticks
  for (let v = 0; v <= domain + 1e-12; v += step) {
    const x = sx(v);
    el("line", { x1: x, x2: x, y1: top - 2, y2: height - axisH + 4, class: v === 0 ? "axis-line" : "grid-line" }, svg);
    const t = el("text", { x, y: height - 6, "text-anchor": "middle" }, svg);
    t.textContent = opts.tickFormat ? opts.tickFormat(v) : String(+v.toFixed(4));
  }

  const g = el("g", {}, svg);
  rows.forEach((r, i) => {
    const y = top + i * rowH;
    const row = el("g", { class: "row" }, g);
    const lbl = el("text", {
      x: labelW - 10, y: y + rowH / 2 + 4, "text-anchor": "end", class: r.strong ? "lbl-strong" : "",
    }, row);
    lbl.textContent = narrow && r.shortLabel ? r.shortLabel : r.label;
    const by = y + (rowH - barH) / 2;
    el("path", { d: hBarPath(labelW, by, sx(r.value) - labelW, barH), fill: r.color, class: "bar" }, row);
    let tipX = sx(r.value);
    if (r.err) {
      const x1 = sx(r.value - r.err);
      const x2 = sx(r.value + r.err);
      const cy = by + barH / 2;
      el("line", { x1, x2, y1: cy, y2: cy, class: "whisker" }, row);
      el("line", { x1: x2, x2, y1: cy - 4, y2: cy + 4, class: "whisker" }, row);
      el("line", { x1, x2: x1, y1: cy - 4, y2: cy + 4, class: "whisker" }, row);
      tipX = x2;
    }
    const val = el("text", { x: tipX + 6, y: by + barH / 2 + 4, class: "val" }, row);
    val.textContent = fmt(r.value);
    const hit = el("rect", { x: 0, y, width, height: rowH, class: "hit" }, row);
    hit.addEventListener("pointermove", (e) => {
      g.classList.add("row-hover");
      row.classList.add("active");
      showTip(e, r.tip);
    });
    hit.addEventListener("pointerleave", () => {
      g.classList.remove("row-hover");
      row.classList.remove("active");
      hideTip();
    });
  });
  return svg;
}

/**
 * Column histogram of integer bins with one highlighted bin.
 * bins: [{x, count}], highlight: x value or null
 */
export function histogram(container, bins, highlight, opts = {}) {
  const width = Math.max(container.clientWidth, 260);
  const height = container.clientHeight || 150;
  const left = 4;
  const right = 4;
  const top = 18;
  const bottom = 20;
  const xs = bins.map((b) => b.x);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const n = maxX - minX + 1;
  const slot = (width - left - right) / n;
  const gap = Math.min(2, slot * 0.25);
  const maxC = Math.max(...bins.map((b) => b.count));
  const plotH = height - top - bottom;
  const base = height - bottom;

  container.innerHTML = "";
  const svg = el("svg", {
    viewBox: `0 0 ${width} ${height}`,
    role: "img",
    "aria-label": opts.ariaLabel ?? "Histogram",
  }, container);
  el("line", { x1: left, x2: width - right, y1: base + 0.5, y2: base + 0.5, class: "axis-line" }, svg);

  for (const b of bins) {
    const x = left + (b.x - minX) * slot + gap / 2;
    const h = (b.count / maxC) * plotH;
    const on = highlight !== null && b.x === highlight;
    const bar = el("path", {
      d: vBarPath(x, base, slot - gap, h),
      fill: on ? "var(--series-1)" : "var(--neutral-bar)",
      class: "bar",
    }, svg);
    if (on) bar.setAttribute("data-highlight", "true");
    const hit = el("rect", { x: x - gap / 2, y: top - 10, width: slot, height: plotH + 10, class: "hit" }, svg);
    hit.addEventListener("pointermove", (e) =>
      showTip(e, `<strong>${b.x} rings</strong>${b.count.toLocaleString()} abalones`));
    hit.addEventListener("pointerleave", hideTip);
    if (on) {
      const t = el("text", { x: x + (slot - gap) / 2, y: base - h - 5, "text-anchor": "middle", class: "val" }, svg);
      t.textContent = "▼";
      t.setAttribute("style", "fill: var(--series-1); font-size: 10px");
    }
  }
  for (let v = Math.ceil(minX / 5) * 5; v <= maxX; v += 5) {
    const t = el("text", { x: left + (v - minX) * slot + slot / 2, y: height - 4, "text-anchor": "middle" }, svg);
    t.textContent = String(v);
  }
  return svg;
}
