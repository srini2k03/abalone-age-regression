// Browser smoke test for the static site in web/ (requires the `playwright` package + Chromium).
//
//   node scripts/web_smoke_test.cjs [--screenshots <dir>]
//
// Serves web/ on a local port, then checks that the page loads without console errors, that
// predictions update when inputs change, that preset and fixture rows match the Python
// predictions, and that there is no horizontal overflow at desktop and phone widths.
const http = require("node:http");
const fs = require("node:fs");
const path = require("node:path");
const { chromium } = require("playwright");

const ROOT = path.resolve(__dirname, "..", "web");
const TYPES = { ".html": "text/html", ".css": "text/css", ".js": "text/javascript", ".json": "application/json", ".svg": "image/svg+xml" };
const args = process.argv.slice(2);
const shotDir = args.includes("--screenshots") ? path.resolve(args[args.indexOf("--screenshots") + 1]) : null;

function serve() {
  const server = http.createServer((req, res) => {
    const url = decodeURIComponent(new URL(req.url, "http://x").pathname);
    const file = path.join(ROOT, url.endsWith("/") ? `${url}index.html` : url);
    if (!file.startsWith(ROOT) || !fs.existsSync(file)) {
      res.writeHead(404).end("not found");
      return;
    }
    res.writeHead(200, { "content-type": TYPES[path.extname(file)] ?? "application/octet-stream" });
    fs.createReadStream(file).pipe(res);
  });
  return new Promise((resolve) => server.listen(0, "127.0.0.1", () => resolve(server)));
}

const failures = [];
const check = (ok, msg) => {
  console.log(`${ok ? "PASS" : "FAIL"}  ${msg}`);
  if (!ok) failures.push(msg);
};

async function rings(page) {
  return Number(await page.getAttribute("#pred-rings", "data-rings"));
}

async function setRow(page, row) {
  await page.check(`input[name="sex"][value="${row.sex}"]`);
  for (const k of ["length", "diameter", "height", "whole_weight", "shucked_weight", "viscera_weight", "shell_weight"]) {
    await page.fill(`#in-${k}`, String(row[k]));
  }
}

(async () => {
  const server = await serve();
  const base = `http://127.0.0.1:${server.address().port}/`;
  const browser = await chromium.launch();
  const insights = JSON.parse(fs.readFileSync(path.join(ROOT, "data", "insights.json"), "utf8"));
  const fixture = JSON.parse(fs.readFileSync(path.join(ROOT, "data", "parity-fixture.json"), "utf8"));

  for (const vp of [
    { name: "desktop", width: 1280, height: 860, scheme: "light" },
    { name: "phone", width: 390, height: 844, scheme: "light", mobile: true },
    { name: "desktop-dark", width: 1280, height: 860, scheme: "dark" },
    { name: "phone-dark", width: 390, height: 844, scheme: "dark", mobile: true },
  ]) {
    const ctx = await browser.newContext({
      viewport: { width: vp.width, height: vp.height },
      colorScheme: vp.scheme,
      isMobile: !!vp.mobile,
      hasTouch: !!vp.mobile,
      deviceScaleFactor: vp.mobile ? 2 : 1,
    });
    const page = await ctx.newPage();
    const errors = [];
    page.on("console", (m) => { if (m.type() === "error" || m.type() === "warning") errors.push(`console.${m.type()}: ${m.text()}`); });
    page.on("pageerror", (e) => errors.push(`pageerror: ${e.message}`));
    page.on("requestfailed", (r) => errors.push(`requestfailed: ${r.url()}`));
    page.on("response", (r) => { if (r.status() >= 400) errors.push(`HTTP ${r.status()}: ${r.url()}`); });

    await page.goto(base, { waitUntil: "networkidle" });
    await page.waitForSelector("body[data-ready=true]", { timeout: 10000 });

    const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
    check(overflow <= 0, `[${vp.name}] no horizontal overflow (scrollWidth - innerWidth = ${overflow})`);

    if (vp.name === "desktop") {
      // default preset matches Python
      const p0 = insights.presets[2];
      const r0 = await rings(page);
      check(Math.abs(r0 - p0.python_pred_rings) < 1e-4,
        `default preset "${p0.label}": JS ${r0.toFixed(6)} vs Python ${p0.python_pred_rings} rings`);

      // every preset button
      const chips = await page.$$(".chip");
      for (let i = 0; i < chips.length; i++) {
        await chips[i].click();
        const r = await rings(page);
        const p = insights.presets[i];
        check(Math.abs(r - p.python_pred_rings) < 1e-4,
          `preset "${p.label}" (UCI row ${p.uci_row}): JS ${r.toFixed(6)} vs Python ${p.python_pred_rings}`);
      }
      check(!(await page.isHidden("#pred-actual")), "actual ring count shown for a preset");

      // inputs drive the prediction
      const before = await rings(page);
      await page.fill("#in-shell_weight", "0.6");
      const afterNum = await rings(page);
      check(afterNum !== before, `typing shell weight changes prediction (${before.toFixed(3)} -> ${afterNum.toFixed(3)})`);
      check(await page.isHidden("#pred-actual"), "actual ring count hidden after editing");
      const slider = page.locator("#in-length-range");
      await slider.focus();
      for (let k = 0; k < 10; k++) await page.keyboard.press("ArrowRight");
      const afterSlider = await rings(page);
      const lenVal = await page.inputValue("#in-length");
      check(afterSlider !== afterNum, `slider moves length to ${lenVal} and changes prediction (${afterNum.toFixed(3)} -> ${afterSlider.toFixed(3)})`);
      await page.check('input[name="sex"][value="I"]');
      const afterSex = await rings(page);
      check(afterSex !== afterSlider, `switching sex to infant changes prediction (${afterSlider.toFixed(3)} -> ${afterSex.toFixed(3)})`);
      const age = Number(await page.textContent("#pred-age"));
      check(Math.abs(age - (afterSex + 1.5)) < 0.051, `age = rings + 1.5 (${age})`);

      // known rows typed in by hand, including an anomalous and an out-of-range one
      const picks = [0, 1, fixture.rows.length - 4, fixture.rows.length - 5];
      for (const idx of picks) {
        const row = fixture.rows[idx];
        await setRow(page, row);
        const r = await rings(page);
        check(Math.abs(r - row.python_rings) < 1e-4,
          `fixture row ${idx} typed in: JS ${r.toFixed(6)} vs Python ${row.python_rings.toFixed(6)} (|d| = ${Math.abs(r - row.python_rings).toExponential(2)})`);
      }

      // invalid input is flagged and ignored
      const keep = await rings(page);
      await page.fill("#in-height", "");
      check((await page.getAttribute("#in-height", "aria-invalid")) === "true" && (await rings(page)) === keep,
        "empty field is flagged and does not change the prediction");

      // theme toggle flips to dark
      await page.click("#theme-toggle");
      check((await page.getAttribute("html", "data-theme")) === "dark", "theme toggle switches to dark");
      await page.click("#theme-toggle");

      check((await page.$$("#cmp-chart svg .row")).length === 10, "model comparison chart has 10 bars");
      check((await page.$$("#cv-table tbody tr")).length === 10, "CV table has 10 rows");
      check((await page.$$("#shap-chart svg .row")).length === insights.shap.values.length, "SHAP chart has one bar per feature");
      await page.reload({ waitUntil: "networkidle" });
      await page.waitForSelector("body[data-ready=true]");
    }

    if (shotDir) {
      fs.mkdirSync(shotDir, { recursive: true });
      await page.evaluate(() => window.scrollTo(0, 0));
      await page.waitForTimeout(200);
      await page.screenshot({ path: path.join(shotDir, `${vp.name}-top.png`) });
      await page.screenshot({ path: path.join(shotDir, `${vp.name}-full.png`), fullPage: true });
    }
    check(errors.length === 0, `[${vp.name}] no console errors or failed requests${errors.length ? `: ${errors.join(" | ")}` : ""}`);
    await ctx.close();
  }

  await browser.close();
  server.close();
  console.log(failures.length ? `\n${failures.length} check(s) failed` : "\nall checks passed");
  process.exit(failures.length ? 1 : 0);
})().catch((e) => {
  console.error(e);
  process.exit(1);
});
