/**
 * render-mermaid-svgs.mjs — pre-render every ```mermaid block of every blog post to a static SVG,
 * so the 精读版 pages (reads/<id>.html) can embed the board's own diagrams as figures.
 *
 *   node scripts/render-mermaid-svgs.mjs            # all posts in data/blog.json
 *   node scripts/render-mermaid-svgs.mjs <post-id>  # one post
 *
 * Output: reads/img/<post-id>/fig-<n>.svg  (n = 1-based order of appearance in the markdown)
 * Needs: npm i playwright-core (Chromium at /opt/pw-browsers or PLAYWRIGHT_CHROMIUM) and js/vendor/mermaid.min.js.
 */
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const blog = JSON.parse(fs.readFileSync(path.join(ROOT, "data/blog.json"), "utf8"));
const only = process.argv[2];
const posts = blog.posts.filter((p) => p.id && p.file && (!only || p.id === only));

function findChromium() {
  if (process.env.PLAYWRIGHT_CHROMIUM && fs.existsSync(process.env.PLAYWRIGHT_CHROMIUM)) return process.env.PLAYWRIGHT_CHROMIUM;
  const base = process.env.PLAYWRIGHT_BROWSERS_PATH || "/opt/pw-browsers";
  if (!fs.existsSync(base)) return null;
  for (const d of fs.readdirSync(base)) {
    for (const c of [path.join(base, d, "chrome-linux", "chrome"), path.join(base, d, "chrome-linux64", "chrome")]) {
      if (fs.existsSync(c)) return c;
    }
  }
  return null;
}

const { chromium } = await import(path.join(ROOT, "node_modules/playwright-core/index.mjs"));
const exe = findChromium();
if (!exe) { console.error("Chromium not found"); process.exit(1); }
const browser = await chromium.launch({ executablePath: exe, args: ["--no-sandbox"] });
const page = await (await browser.newContext({ viewport: { width: 1400, height: 900 } })).newPage();
await page.setContent(`<!doctype html><html><body style="font-family:Inter,ui-sans-serif,system-ui,'PingFang SC','Microsoft YaHei',sans-serif"><div id="host"></div></body></html>`);
await page.addScriptTag({ path: path.join(ROOT, "js/vendor/mermaid.min.js") });
await page.evaluate(() => window.mermaid.initialize({ startOnLoad: false, securityLevel: "strict", theme: "neutral",
  fontFamily: "Inter, ui-sans-serif, system-ui, 'PingFang SC', 'Microsoft YaHei', sans-serif" }));

let total = 0, failed = 0;
for (const p of posts) {
  const md = fs.readFileSync(path.join(ROOT, p.file), "utf8");
  const re = /```mermaid\n([\s\S]*?)```/g;
  let m, n = 0;
  const outDir = path.join(ROOT, "reads/img", p.id);
  while ((m = re.exec(md))) {
    n++;
    fs.mkdirSync(outDir, { recursive: true });
    try {
      const svg = await page.evaluate(async ([code, id]) => {
        const { svg } = await window.mermaid.render(id, code);
        return svg;
      }, [m[1], `fig_${n}_${Date.now()}`]);
      // make the svg self-sizing when embedded via <img>
      const clean = svg.replace(/<br>/g, "<br/>").replace(/ style="max-width:[^"]*"/, "");
      fs.writeFileSync(path.join(outDir, `fig-${n}.svg`), clean);
      total++;
    } catch (e) {
      failed++;
      console.warn(`! ${p.id} fig-${n}: ${String(e.message || e).split("\n")[0]}`);
    }
  }
  if (n) console.log(`${p.id}: ${n} diagram(s)`);
}
await browser.close();
console.log(`rendered ${total} svg(s), ${failed} failed`);
process.exit(failed ? 1 : 0);
