#!/usr/bin/env node
/**
 * build-share-pages.mjs — per-post share assets for the SPA blog.
 *
 * Generates, for every post in data/blog.json:
 *   1. assets/og/<id>.png   — 1200×630 share card (unique title per post)
 *   2. p/<id>.html          — static snapshot page with per-post <title>/description/og:*
 *                             meta (crawlers & link unfurlers read these), a static
 *                             excerpt for indexing, and an instant redirect into the SPA
 * Plus sitemap.xml listing the homepage + all snapshot pages.
 *
 * Re-runnable: run again after adding posts (only missing/changed og images are re-rendered
 * unless --force). Requires local playwright-core + the preinstalled Chromium.
 *
 *   node scripts/build-share-pages.mjs [--force]
 */
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const BASE = "https://lloo099.github.io/mission44";
const FORCE = process.argv.includes("--force");
const CHROME_CANDIDATES = [
  "/opt/pw-browsers/chromium-1194/chrome-linux/chrome",
  process.env.PLAYWRIGHT_CHROMIUM || "",
];

const blog = JSON.parse(fs.readFileSync(path.join(ROOT, "data/blog.json"), "utf8"));
const posts = blog.posts || [];
fs.mkdirSync(path.join(ROOT, "assets/og"), { recursive: true });
fs.mkdirSync(path.join(ROOT, "p"), { recursive: true });

const esc = (s) => String(s || "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");

/* ---------- og card template ---------- */
function ogHtml(post) {
  const title = esc(post.title);
  const date = esc(post.date || "");
  return `<!DOCTYPE html><html><head><meta charset="utf-8"><style>
  * { margin:0; padding:0; box-sizing:border-box; }
  body { width:1200px; height:630px; background:#0b0f17; font-family:system-ui,-apple-system,"Segoe UI",sans-serif; color:#fff; position:relative; overflow:hidden;
    background-image:repeating-linear-gradient(0deg, rgba(106,166,255,.07) 0 1px, transparent 1px 48px),
                     repeating-linear-gradient(90deg, rgba(106,166,255,.07) 0 1px, transparent 1px 48px); }
  .brand { position:absolute; top:44px; left:56px; font-size:30px; font-weight:600; }
  .brand .logo { color:#6aa6ff; margin-right:12px; }
  .date { position:absolute; top:52px; right:56px; font-size:20px; color:#5b6b85; letter-spacing:.06em; }
  .title { position:absolute; left:56px; right:120px; top:180px; font-size:52px; font-weight:700; line-height:1.28;
    display:-webkit-box; -webkit-line-clamp:4; -webkit-box-orient:vertical; overflow:hidden; }
  .series { position:absolute; left:56px; bottom:96px; font-size:22px; color:#8ea3c7; }
  .url { position:absolute; left:56px; bottom:44px; font-size:17px; color:#5b6b85; }
  .deco { position:absolute; right:-40px; bottom:-60px; font-size:300px; color:rgba(106,166,255,.10); }
  </style></head><body>
    <div class="brand"><span class="logo">◈</span>RL on NPU</div>
    <div class="date">${date}</div>
    <div class="title">${title}</div>
    <div class="series">NPU Frontier Dispatch — 深度解析 · 图解 · 对比</div>
    <div class="url">lloo099.github.io/mission44</div>
    <div class="deco">◈</div>
  </body></html>`;
}

/* ---------- snapshot page template ---------- */
function snapshotHtml(post) {
  const id = post.id;
  const title = esc(post.title);
  const desc = esc((post.subtitle || "").slice(0, 280));
  const og = `${BASE}/assets/og/${id}.png`;
  const spa = `${BASE}/#blog/${encodeURIComponent(id)}`;
  const self = `${BASE}/p/${id}.html`;
  return `<!DOCTYPE html>
<html lang="zh">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>${title} · NPU Frontier Dispatch</title>
  <meta name="description" content="${desc}" />
  <link rel="canonical" href="${self}" />
  <meta property="og:type" content="article" />
  <meta property="og:title" content="${title}" />
  <meta property="og:description" content="${desc}" />
  <meta property="og:url" content="${self}" />
  <meta property="og:image" content="${og}" />
  <meta name="twitter:card" content="summary_large_image" />
  <meta name="twitter:image" content="${og}" />
  <script>location.replace(${JSON.stringify(spa)});</script>
  <meta http-equiv="refresh" content="0; url=${spa}" />
  <style>body{font-family:system-ui,sans-serif;background:#0b0f17;color:#dfe7f5;max-width:720px;margin:80px auto;padding:0 20px;line-height:1.7}a{color:#6aa6ff}</style>
</head>
<body>
  <h1>${title}</h1>
  <p><em>${esc(post.date || "")} · NPU Frontier Dispatch · ${(post.tags || []).map(esc).join(" / ")}</em></p>
  <p>${desc}</p>
  <p><a href="${spa}">→ 阅读全文(RL-on-NPU Research Dashboard)</a></p>
</body>
</html>`;
}

/* ---------- main ---------- */
const chromePath = CHROME_CANDIDATES.find((p) => p && fs.existsSync(p));
let browser = null, page = null;
if (chromePath) {
  const { chromium } = await import(path.join(ROOT, "node_modules/playwright-core/index.mjs"));
  browser = await chromium.launch({ executablePath: chromePath, args: ["--no-sandbox"] });
  page = await browser.newPage({ viewport: { width: 1200, height: 630 } });
} else {
  console.warn("! Chromium not found — skipping og image rendering (snapshots + sitemap only)");
}

let rendered = 0, snapshots = 0;
for (const post of posts) {
  if (!post.id) continue;
  const pngPath = path.join(ROOT, "assets/og", `${post.id}.png`);
  if (page && (FORCE || !fs.existsSync(pngPath))) {
    await page.setContent(ogHtml(post), { waitUntil: "load" });
    await page.screenshot({ path: pngPath });
    rendered++;
  }
  fs.writeFileSync(path.join(ROOT, "p", `${post.id}.html`), snapshotHtml(post));
  snapshots++;
}
if (browser) await browser.close();

/* sitemap */
const urls = [`${BASE}/`, ...posts.filter((p) => p.id).map((p) => `${BASE}/p/${p.id}.html`)];
const sitemap = `<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n`
  + urls.map((u) => `  <url><loc>${u}</loc></url>`).join("\n") + `\n</urlset>\n`;
fs.writeFileSync(path.join(ROOT, "sitemap.xml"), sitemap);
fs.writeFileSync(path.join(ROOT, "robots.txt"), `User-agent: *\nAllow: /\nSitemap: ${BASE}/sitemap.xml\n`);

console.log(`og rendered: ${rendered}, snapshots: ${snapshots}, sitemap urls: ${urls.length}`);
