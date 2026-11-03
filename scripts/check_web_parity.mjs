// Check that the browser model (web/js/model.js) reproduces the Python pipeline's predictions.
//
//   node scripts/check_web_parity.mjs [web/data]
//
// Reads model.json and parity-fixture.json (written by scripts/export_web_model.py) and fails
// if any |JS - Python| difference in predicted rings exceeds the fixture tolerance.
import { readFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const dataDir = resolve(process.argv[2] ?? join(here, "..", "web", "data"));
const { createPredictor } = await import(join(here, "..", "web", "js", "model.js"));

const spec = JSON.parse(readFileSync(join(dataDir, "model.json"), "utf8"));
const fixture = JSON.parse(readFileSync(join(dataDir, "parity-fixture.json"), "utf8"));
const model = createPredictor(spec);

let maxRings = 0;
let maxMargin = 0;
let worst = -1;
fixture.rows.forEach((row, i) => {
  const out = model.predict(row);
  const dRings = Math.abs(out.rings - row.python_rings);
  const dMargin = Math.abs(out.margin - row.python_margin);
  if (dRings > maxRings) {
    maxRings = dRings;
    worst = i;
  }
  maxMargin = Math.max(maxMargin, dMargin);
});

const ok = maxRings < fixture.tolerance;
console.log(
  `parity: ${fixture.rows.length} rows, ${model.nTrees} trees | ` +
    `max |dRings| = ${maxRings.toExponential(3)}, max |dLog1p| = ${maxMargin.toExponential(3)} ` +
    `(tolerance ${fixture.tolerance}) -> ${ok ? "OK" : "FAIL"}`,
);
if (!ok) {
  console.error("worst row:", JSON.stringify(fixture.rows[worst]));
  process.exit(1);
}
