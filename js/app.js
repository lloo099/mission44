/* RL-on-NPU Research Dashboard — vanilla JS, no build step. */

const DATA_SOURCES = {
  rl: "data/rl.json",
  ascend: "data/ascend.json",
  modeling: "data/modeling.json",
  agentic: "data/agentic.json",
  ideas: "data/ideas.json",
  live: "data/feed.json",
};

const store = {}; // key -> array of entries
const storeUpdated = {}; // key -> ISO date string (from each file's "updated")
const activeFilters = {}; // key -> Set of active tags
let searchTerm = "";
let agenticTrends = []; // {title, body} blurbs for the Agentic RL section

/* defaults for dynamic <title> / meta description (restored on the blog index) */
const DEFAULT_TITLE = document.title;
const metaDesc = document.querySelector('meta[name="description"]');
const DEFAULT_DESC = metaDesc ? metaDesc.content : "";

/* ---------- data loading ---------- */
async function loadJSON(url) {
  try {
    const res = await fetch(url, { cache: "no-cache" });
    if (!res.ok) throw new Error(res.status);
    return await res.json();
  } catch (e) {
    console.warn("Failed to load", url, e);
    return null;
  }
}

async function init() {
  const keys = Object.keys(DATA_SOURCES);
  const results = await Promise.all(keys.map((k) => loadJSON(DATA_SOURCES[k])));
  keys.forEach((k, i) => {
    const payload = results[i];
    store[k] = (payload && payload.items) ? payload.items : (Array.isArray(payload) ? payload : []);
    storeUpdated[k] = (payload && payload.updated) ? String(payload.updated).slice(0, 10) : "";
    if (k === "agentic" && payload) agenticTrends = payload.trends || [];
    activeFilters[k] = new Set();
  });

  wireTheme();
  buildStats();
  buildLedger();
  renderAgenticTrends();
  ["rl", "ascend", "modeling", "agentic", "ideas", "live"].forEach(renderPanel);
  buildFilterbars();
  wireTabs();
  wireSearch();
  wireLive();
  wireCurves();
  wireCompare();
  wireTimeline();
  wireBlog();
  wireMermaidLightbox();
  wireTabsScrollHint();
  wireShareButtons();
  if (window.wireArch) wireArch();
  setLastUpdated();
}

/* copy a per-post share link (static snapshot page in /p/ with its own og card).
   Falls back to the SPA hash URL when running on a non-published origin. */
function wireShareButtons() {
  document.addEventListener("click", (e) => {
    const btn = e.target.closest(".share-btn");
    if (!btn) return;
    const id = btn.dataset.share;
    const onPages = location.hostname.endsWith("github.io");
    const url = onPages
      ? location.origin + location.pathname.replace(/index\.html$/, "").replace(/\/$/, "") + "/p/" + id + ".html"
      : location.origin + location.pathname + "#blog/" + id;
    const done = () => { const t = btn.textContent; btn.textContent = "✓ 已复制"; setTimeout(() => { btn.textContent = t; }, 1600); };
    if (navigator.clipboard && navigator.clipboard.writeText) navigator.clipboard.writeText(url).then(done).catch(() => { window.prompt("复制链接:", url); });
    else window.prompt("复制链接:", url);
  });
}

/* ---------- light/dark theme toggle (default: light) ---------- */
function wireTheme() {
  const btn = document.getElementById("theme-toggle");
  const root = document.documentElement;
  const apply = (t) => {
    if (t === "dark") root.setAttribute("data-theme", "dark");
    else root.removeAttribute("data-theme");
    if (btn) { btn.textContent = t === "dark" ? "☀️" : "🌙"; btn.title = t === "dark" ? "切换到浅色" : "切换到深色"; }
    window.__reRenderMermaid && window.__reRenderMermaid(); // theme-aware diagrams (no-op when none rendered)
  };
  let cur;
  try { cur = localStorage.getItem("theme") || "light"; } catch (e) { cur = "light"; }
  apply(cur);
  if (btn) btn.addEventListener("click", () => {
    cur = cur === "dark" ? "light" : "dark";
    try { localStorage.setItem("theme", cur); } catch (e) {}
    apply(cur);
  });
}

/* render any .mermaid diagrams inside a scope once mermaid.js is ready (it loads async).
   The vendor bundle is ~3.5MB: on slow networks it can take well over 5s, so we wait
   patiently (with a visible "loading" placeholder), fall back only on real script failure
   or a long timeout, and recover automatically via the 'mermaid-ready' event. */
function mermaidFallback(nodes) {
  nodes.forEach((n) => {
    n.setAttribute("data-processed", "fallback");
    n.classList.remove("mermaid-waiting");
    n.classList.add("mermaid-failed");
    n.innerHTML = '<details class="mermaid-fallback"><summary>⚠️ 图渲染引擎加载失败 — 点开查看图源码(刷新页面可重试)</summary><pre>'
      + escapeHtml(n.dataset.src || "") + "</pre></details>";
  });
}
function renderMermaidIn(scope, tries) {
  const nodes = scope.querySelectorAll(".mermaid:not([data-processed])");
  if (!nodes.length) return;
  nodes.forEach((n) => { if (!n.dataset.src) n.dataset.src = n.textContent; }); // keep the diagram source for re-render / fallback
  if (!window.mermaid) {
    const t = tries || 0;
    if (window.__mermaidFailed) { mermaidFallback(nodes); return; }   // script 404/failed — no point waiting
    if (t === 3) nodes.forEach((n) => {                                // after ~1s, swap raw source for a friendly placeholder
      n.classList.add("mermaid-waiting");
      n.textContent = "⏳ 图渲染引擎加载中…(约 1MB,首次访问或网络较慢时需几秒)";
    });
    if (t < 120) { setTimeout(() => renderMermaidIn(scope, t + 1), 300); return; }  // wait up to ~36s
    mermaidFallback(nodes);                                            // long timeout — degrade, recoverable via mermaid-ready
    return;
  }
  nodes.forEach((n) => {                                               // restore source if we showed the waiting placeholder
    if (n.classList.contains("mermaid-waiting")) { n.classList.remove("mermaid-waiting"); n.textContent = n.dataset.src; }
  });
  try { window.mermaid.run({ nodes }); } catch (e) { console.warn("mermaid", e); }
}

/* when the (slow) vendor script finally arrives, revive placeholders AND any fallback blocks */
document.addEventListener("mermaid-ready", () => {
  document.querySelectorAll('.mermaid[data-processed="fallback"], .mermaid.mermaid-waiting').forEach((n) => {
    if (!n.dataset.src) return;
    n.removeAttribute("data-processed");
    n.classList.remove("mermaid-failed", "mermaid-waiting");
    n.textContent = n.dataset.src;
  });
  const post = document.getElementById("blog-post");
  if (post && !post.hidden) renderMermaidIn(post);
});
document.addEventListener("mermaid-failed", () => {
  document.querySelectorAll(".mermaid:not([data-processed])").forEach((n) => { if (!n.dataset.src) n.dataset.src = n.textContent; });
  mermaidFallback(document.querySelectorAll(".mermaid:not([data-processed])"));
});

/* B. re-render all mermaid diagrams after a theme switch (called from wireTheme) */
window.__reRenderMermaid = function () {
  if (window.__initMermaid) window.__initMermaid();
  document.querySelectorAll(".mermaid[data-processed]").forEach((n) => {
    if (!n.dataset.src) return;
    n.removeAttribute("data-processed");
    n.classList.remove("mermaid-failed");
    n.textContent = n.dataset.src;
  });
  const post = document.getElementById("blog-post");
  if (post && !post.hidden) renderMermaidIn(post);
};

/* ---------- blog / dispatch (markdown posts, unipat-style index) ---------- */
async function wireBlog() {
  const idx = document.getElementById("blog-index");
  const post = document.getElementById("blog-post");
  if (!idx || !post) return;
  const head = document.querySelector("#blog .panel-head"); // section intro — hidden while reading a post
  let posts = [];
  try {
    const res = await fetch("data/blog.json", { cache: "no-cache" });
    if (!res.ok) throw new Error(res.status);
    posts = (await res.json()).posts || [];
  } catch (e) {
    idx.innerHTML = `<div class="empty">Couldn't load blog index (${escapeHtml(String(e.message || e))}).</div>`;
    return;
  }

  // expose posts + matcher so the global search (updateTabCounts) can count blog hits
  window.__blogPosts = posts;
  function blogMatches(p) {
    return !searchTerm || (p.title + " " + (p.subtitle || "") + " " + (p.tags || []).join(" ")).toLowerCase().includes(searchTerm);
  }
  window.__blogMatches = blogMatches;

  // knowledge map: collapsible overview diagram + per-cluster jump chips, above the index.
  // Diagram source lives in data/blog-map.mmd and renders through the shared mermaid pipeline
  // (placeholder / fallback / theme re-render all come for free). Rendered lazily on first open.
  const MAP_CLUSTERS = [
    ["系统与综述", ["survey-2026-h1-architecture", "dispatch-02-rollout-bottleneck", "dispatch-08-agentic-rl", "dispatch-10-radixattention", "dispatch-11-vllm-vs-sglang", "dispatch-23-agentic-rl-problem-map", "dispatch-25-efficiency-aware-agent-rl", "dispatch-26-inference-efficiency-map"]],
    ["模型与架构", ["dispatch-01-seed-2-1", "dispatch-04-minimax-msa", "dispatch-05-deepseek-v4", "dispatch-06-glm-5-2", "dispatch-07-mimo-v2-5", "dispatch-15-dspark", "dispatch-16-deepseek-v4-agent", "dispatch-21-longcat-2", "dispatch-24-kimi-k3"]],
    ["SWE 与训练数据", ["dispatch-12-swe-agents", "dispatch-14-scaleswe", "dispatch-17-denovoswe"]],
    ["RL 框架", ["dispatch-09-radixark-miles", "dispatch-18-prime-rl-vs-skyrl", "dispatch-19-slime", "dispatch-22-deepswe-rllm"]],
    ["昇腾落地", ["dispatch-03-ascend-950", "dispatch-13-swe-rl-on-ascend", "dispatch-20-openpangu"]],
  ];
  const mapEl = document.createElement("details");
  mapEl.className = "blog-map";
  const byId = {};
  posts.forEach((p) => { byId[p.id] = p; });
  const chipLabel = (id) => {
    const m = id.match(/^dispatch-(\d+)/);
    return m ? "D" + m[1] : "综述";
  };
  mapEl.innerHTML = '<summary>🗺️ 知识地图 — ' + posts.length + ' 篇的主题脉络与阅读路径</summary>'
    + '<div class="blog-map-diagram"><div class="mermaid"></div></div>'
    + '<div class="blog-map-chips">' + MAP_CLUSTERS.map(([name, ids]) =>
      '<div class="map-cluster"><span class="map-cluster-name">' + escapeHtml(name) + "</span>"
      + ids.filter((id) => byId[id]).map((id) =>
        '<a class="map-chip" href="#blog/' + escapeAttr(id) + '" title="' + escapeAttr((byId[id] || {}).title || id) + '">' + chipLabel(id) + "</a>").join("")
      + "</div>").join("") + "</div>";
  idx.parentNode.insertBefore(mapEl, idx);
  let mapLoaded = false;
  mapEl.addEventListener("toggle", async () => {
    if (!mapEl.open || mapLoaded) return;
    mapLoaded = true;
    try {
      const res = await fetch("data/blog-map.mmd", { cache: "no-cache" });
      if (!res.ok) throw new Error(res.status);
      mapEl.querySelector(".mermaid").textContent = await res.text();
      renderMermaidIn(mapEl);
    } catch (e) {
      mapEl.querySelector(".blog-map-diagram").innerHTML = '<div class="empty">地图加载失败。</div>';
    }
  });

  function showIndex() {
    post.hidden = true; idx.hidden = false; mapEl.hidden = false;
    if (head) head.hidden = false;
    document.title = DEFAULT_TITLE;
    if (metaDesc) metaDesc.content = DEFAULT_DESC;
    if (!posts.length) { idx.innerHTML = `<div class="empty">No posts yet.</div>`; return; }
    const visible = posts.filter(blogMatches);
    if (!visible.length) {
      idx.innerHTML = '<div class="empty">没有匹配 "' + escapeHtml(searchTerm) + '" 的 Dispatch。</div>';
      return;
    }
    idx.innerHTML = visible.map((p) => `<a class="blog-card" href="#blog/${escapeAttr(p.id)}">
      <div class="blog-card-date">${escapeHtml(p.date || "")}</div>
      <h3>${escapeHtml(p.title)}</h3>
      <p>${escapeHtml(p.subtitle || "")}</p>
      <div class="blog-card-tags">${(p.tags || []).map((t) => `<span class="tag">${escapeHtml(t)}</span>`).join("")}</div>
    </a>`).join("");
  }

  // re-render the blog index with the current search term (only when the index is visible)
  window.__refreshBlogIndex = () => { if (post.hidden) showIndex(); };

  async function openPost(id) {
    const p = posts.find((x) => x.id === id);
    if (!p) { showIndex(); return; }
    idx.hidden = true; post.hidden = false; mapEl.hidden = true;
    if (head) head.hidden = true;
    toTop();                       // jump up immediately, before the async fetch
    post.innerHTML = `<div class="empty">Loading…</div>`;
    try {
      const res = await fetch(p.file, { cache: "no-cache" });
      if (!res.ok) throw new Error(res.status);
      const md = await res.text();
      // estimated read time: CJK chars @380/min + latin words @220/min
      const cjk = (md.match(/[\u4e00-\u9fff]/g) || []).length;
      const words = (md.match(/[A-Za-z0-9]+/g) || []).length;
      const mins = Math.max(1, Math.round(cjk / 380 + words / 220));
      // prev/next: posts are sorted newest-first, so i-1 = newer, i+1 = older
      const i = posts.findIndex((x) => x.id === id);
      const newer = posts[i - 1], older = posts[i + 1];
      const postNav = '<div class="post-nav">'
        + (newer ? '<a class="prev" href="#blog/' + escapeAttr(newer.id) + '"><span class="pn-label">← 更新一篇</span><span class="pn-title">' + escapeHtml(newer.title) + "</span></a>" : "<span></span>")
        + (older ? '<a class="next" href="#blog/' + escapeAttr(older.id) + '"><span class="pn-label">更早一篇 →</span><span class="pn-title">' + escapeHtml(older.title) + "</span></a>" : "<span></span>")
        + "</div>";
      post.innerHTML = '<div class="blog-topline"><a class="blog-back" href="#blog">← 所有 Dispatch</a>'
        + '<span class="topline-right"><button class="share-btn" data-share="' + escapeAttr(p.id) + '" title="复制分享链接(带专属预览卡)">🔗 分享</button>'
        + '<span class="read-time">约 ' + mins + " 分钟读完</span></span></div>"
        + `<div class="prose blog-prose">${renderMarkdown(md)}</div>`
        + postNav
        + `<a class="blog-back foot" href="#blog">← 所有 Dispatch</a>`;
      buildBlogLayout(post);
      renderMermaidIn(post);
      document.title = p.title + " · NPU Frontier Dispatch";
      if (metaDesc) metaDesc.content = (p.subtitle || "").slice(0, 200);
    } catch (e) {
      post.innerHTML = `<a class="blog-back" href="#blog">← 返回</a><div class="empty">Couldn't load post (${escapeHtml(String(e.message || e))}).</div>`;
    }
    toTop();                       // after content renders
    requestAnimationFrame(toTop);  // and once more after layout settles
  }
  function toTop() {
    try { window.scrollTo(0, 0); } catch (e) {}
    if (document.documentElement) document.documentElement.scrollTop = 0;
    if (document.body) document.body.scrollTop = 0;
  }

  function route() {
    const parts = location.hash.slice(1).split("/");
    if (parts[0] !== "blog") return;       // only react to our own tab
    if (parts[1]) openPost(parts[1]); else showIndex();
  }
  window.addEventListener("hashchange", route);
  wireTocSpy();
  showIndex();
  route();
}

/* wrap the rendered post in a grid layout and, when long enough, add a table of contents.
   TOC links must NOT touch location.hash (it drives tab/blog routing) — no href, scrollIntoView only. */
function buildBlogLayout(post) {
  const prose = post.querySelector(".blog-prose");
  if (!prose) return;
  const layout = document.createElement("div");
  layout.className = "blog-layout";
  const article = document.createElement("div");
  article.className = "blog-article";
  prose.parentNode.insertBefore(layout, prose);
  article.appendChild(prose);
  layout.appendChild(article);

  const heads = prose.querySelectorAll("h2, h3");
  heads.forEach((h, i) => { h.id = "sec-" + i; });
  if (heads.length < 3) return; // too short for a TOC — keep the layout only

  const toc = document.createElement("nav");
  toc.className = "blog-toc";
  toc.setAttribute("aria-label", "目录");
  toc.innerHTML = '<div class="toc-title">目录</div>' + [...heads].map((h, i) => {
    let t = h.textContent.trim();
    if (t.length > 44) t = t.slice(0, 44) + "…";
    return '<a class="toc-link' + (h.tagName === "H3" ? " toc-h3" : "") + '" data-target="sec-' + i + '">' + escapeHtml(t) + "</a>";
  }).join("");
  toc.addEventListener("click", (e) => {
    const a = e.target.closest(".toc-link");
    if (!a) return;
    e.preventDefault();
    const el = document.getElementById(a.dataset.target);
    if (el) el.scrollIntoView({ behavior: "smooth", block: "start" });
  });
  layout.appendChild(toc);
}

/* highlight the TOC entry for the section currently in view (rAF-throttled, wired once) */
let tocSpyWired = false;
function wireTocSpy() {
  if (tocSpyWired) return;
  tocSpyWired = true;
  let ticking = false;
  window.addEventListener("scroll", () => {
    if (ticking) return;
    ticking = true;
    requestAnimationFrame(() => {
      ticking = false;
      const post = document.getElementById("blog-post");
      if (!post || post.hidden) return;
      const heads = post.querySelectorAll('[id^="sec-"]');
      if (!heads.length) return;
      let cur = null;
      heads.forEach((h) => { if (h.getBoundingClientRect().top <= 150) cur = h; });
      post.querySelectorAll(".toc-link").forEach((a) => {
        a.classList.toggle("active", !!cur && a.dataset.target === cur.id);
      });
    });
  }, { passive: true });
}

/* click a rendered mermaid diagram → full-screen lightbox; click anywhere / Escape closes */
function wireMermaidLightbox() {
  document.addEventListener("click", (e) => {
    const open = document.getElementById("mermaid-lightbox");
    if (open) { open.remove(); return; } // any click while open closes it
    const node = e.target.closest('.blog-prose .mermaid[data-processed]:not(.mermaid-failed)');
    if (!node) return;
    const svg = node.querySelector("svg");
    if (!svg) return;
    const lb = document.createElement("div");
    lb.id = "mermaid-lightbox";
    const inner = document.createElement("div");
    inner.className = "lb-inner";
    const clone = svg.cloneNode(true);
    clone.removeAttribute("width");
    clone.removeAttribute("height"); // let CSS control sizing
    inner.appendChild(clone);
    const close = document.createElement("button");
    close.className = "lb-close";
    close.textContent = "✕";
    lb.appendChild(inner);
    lb.appendChild(close);
    document.body.appendChild(lb);
  });
  document.addEventListener("keydown", (e) => {
    if (e.key !== "Escape") return;
    const lb = document.getElementById("mermaid-lightbox");
    if (lb) lb.remove();
  });
}

/* ---------- timeline (aggregated from curated data) ---------- */
function wireTimeline() {
  const el = document.getElementById("timeline-body");
  if (!el) return;
  const DOMAINS = { rl: "RL", agentic: "Agentic", ascend: "Ascend", modeling: "Modeling" };
  const rows = [];
  Object.keys(DOMAINS).forEach((key) => {
    (store[key] || []).forEach((e) => {
      const m = String(e.year || "").match(/(\d{4})(?:[-/](\d{2}))?/);
      if (!m) return;
      const y = +m[1], mo = m[2] ? +m[2] : 0;
      rows.push({ y, mo, key, label: DOMAINS[key], e });
    });
  });
  rows.sort((a, b) => (b.y - a.y) || (b.mo - a.mo) || a.label.localeCompare(b.label));
  if (!rows.length) { el.innerHTML = `<div class="empty">No dated entries.</div>`; return; }

  let html = "", lastKeyLabel = "";
  rows.forEach((r) => {
    const gl = r.mo ? `${r.y}-${String(r.mo).padStart(2, "0")}` : `${r.y}`;
    if (gl !== lastKeyLabel) { html += `<div class="tl-month">${gl}</div>`; lastKeyLabel = gl; }
    const e = r.e;
    const title = e.url
      ? `<a href="${escapeAttr(e.url)}" target="_blank" rel="noopener">${hl(e.title || "")}</a>`
      : hl(e.title || "");
    html += `<div class="tl-item">
      <span class="tl-dom dom-${r.key}">${escapeHtml(r.label)}</span>
      <span class="tl-title">${title}</span>
      ${e.category ? `<span class="tl-cat">${escapeHtml(e.category)}</span>` : ""}
      ${confBadge(e)}${ascendBadge(e)}
    </div>`;
  });
  el.innerHTML = `<div class="tl-count">${rows.length} dated entries</div>${html}`;
}

/* ---------- markdown renderer (no deps; used by the blog) ---------- */

function mdInline(s) {
  const SENT = "\u0000";
  s = escapeHtml(s).replace(/\\\*/g, SENT);
  s = s.replace(/`([^`]+)`/g, (m, a) => `<code>${a}</code>`);
  s = s.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  s = s.replace(/(^|[^*])\*([^*]+)\*(?!\*)/g, "$1<em>$2</em>");
  s = s.replace(/\[([^\]]+)\]\(([^)\s]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
  return s.split(SENT).join("*");
}

function renderMarkdown(md) {
  const lines = md.replace(/\r/g, "").split("\n");
  const out = [];
  const splitRow = (l) => l.trim().replace(/^\|/, "").replace(/\|$/, "").split("|").map((c) => c.trim());
  let i = 0;
  while (i < lines.length) {
    let l = lines[i];
    if (!l.trim()) { i++; continue; }

    // fenced code block ``` … ```  (```mermaid → rendered diagram; else monospace block)
    if (/^```/.test(l)) {
      const lang = l.replace(/^```/, "").trim().toLowerCase();
      const buf = [];
      i++;
      while (i < lines.length && !/^```/.test(lines[i])) { buf.push(lines[i]); i++; }
      i++; // skip closing fence
      const code = buf.join("\n");
      if (lang === "mermaid") {
        out.push(`<div class="mermaid">${code.replace(/&/g, "&amp;").replace(/</g, "&lt;")}</div>`);
      } else {
        out.push(`<pre class="codeblock"><code>${escapeHtml(code)}</code></pre>`);
      }
      continue;
    }

    // heading
    let m = l.match(/^(#{1,6})\s+(.*)$/);
    if (m) { const n = m[1].length; out.push(`<h${n}>${mdInline(m[2])}</h${n}>`); i++; continue; }

    // horizontal rule (dashes only, no pipe → not a table separator)
    if (/^-{3,}\s*$/.test(l)) { out.push("<hr>"); i++; continue; }

    // table: a row with | followed by a |---| separator
    if (l.includes("|") && i + 1 < lines.length && /^\s*\|?[\s:|-]*-[\s:|-]*\|/.test(lines[i + 1]) && lines[i + 1].includes("-")) {
      const head = splitRow(l);
      i += 2; // skip header + separator
      const body = [];
      while (i < lines.length && lines[i].includes("|") && lines[i].trim()) { body.push(splitRow(lines[i])); i++; }
      const th = head.map((c) => `<th>${mdInline(c)}</th>`).join("");
      const rows = body.map((r) => `<tr>${r.map((c) => `<td>${mdInline(c)}</td>`).join("")}</tr>`).join("");
      out.push(`<div class="prose-tablewrap"><table>${`<tr>${th}</tr>`}${rows}</table></div>`);
      continue;
    }

    // blockquote
    if (/^>\s?/.test(l)) {
      const buf = [];
      while (i < lines.length && /^>\s?/.test(lines[i])) { buf.push(lines[i].replace(/^>\s?/, "")); i++; }
      const paras = buf.join("\n").split(/\n{2,}/).map((p) => `<p>${mdInline(p.replace(/\n/g, " "))}</p>`).join("");
      out.push(`<blockquote>${paras}</blockquote>`);
      continue;
    }

    // unordered list
    if (/^[-*]\s+/.test(l)) {
      const items = [];
      while (i < lines.length && /^[-*]\s+/.test(lines[i])) { items.push(`<li>${mdInline(lines[i].replace(/^[-*]\s+/, ""))}</li>`); i++; }
      out.push(`<ul>${items.join("")}</ul>`);
      continue;
    }

    // ordered list
    if (/^\d+\.\s+/.test(l)) {
      const items = [];
      while (i < lines.length && /^\d+\.\s+/.test(lines[i])) { items.push(`<li>${mdInline(lines[i].replace(/^\d+\.\s+/, ""))}</li>`); i++; }
      out.push(`<ol>${items.join("")}</ol>`);
      continue;
    }

    // paragraph (gather until blank / block start)
    const para = [];
    while (i < lines.length && lines[i].trim() && !/^(#{1,6}\s|>|[-*]\s|\d+\.\s|-{3,}\s*$)/.test(lines[i]) && !(lines[i].includes("|") && i + 1 < lines.length && lines[i + 1].includes("-") && lines[i + 1].includes("|"))) {
      para.push(lines[i]); i++;
    }
    if (para.length) out.push(`<p>${mdInline(para.join(" "))}</p>`);
    else { i++; }
  }
  return out.join("\n");
}


/* ---------- stats ---------- */
function buildStats() {
  const grid = document.getElementById("stat-grid");
  const stats = [
    { num: store.rl.length, lbl: "RL papers & frameworks" },
    { num: store.ascend.length, lbl: "Ascend / NPU entries" },
    { num: store.modeling.length, lbl: "Modeling advances" },
    { num: store.agentic.length, lbl: "Agentic RL entries" },
    { num: store.ideas.length, lbl: "Project ideas" },
  ];
  grid.innerHTML = stats
    .map((s) => `<div class="stat"><div class="num">${s.num}</div><div class="lbl">${s.lbl}</div></div>`)
    .join("");
}

/* ---------- Data Quality Ledger (Overview) ---------- */
const PRIMARY_HOSTS = new Set([
  "arxiv.org", "github.com", "gitee.com", "huggingface.co", "hiascend.com", "docs.vllm.ai",
  "mindspore.cn", "pytorch.org", "docs.sglang.ai", "qwenlm.github.io", "ai.meta.com",
  "openai.com", "x.ai", "mimo.mi.com", "mimo.xiaomi.com", "seed.bytedance.com", "deepseek.com",
]);
function hostOf(u) { const m = String(u || "").match(/^https?:\/\/([^/]+)/i); return m ? m[1].replace(/^www\./, "") : ""; }
function isPrimarySource(e) { const h = hostOf(e.url); return PRIMARY_HOSTS.has(h) || h.endsWith(".edu") || h.endsWith(".edu.cn"); }

function buildLedger() {
  const el = document.getElementById("data-ledger");
  if (!el) return;
  const KEYS = ["rl", "ascend", "modeling"];
  const all = KEYS.flatMap((k) => (store[k] || []));
  const n = all.length || 1;
  const cnt = (f) => all.filter(f).length;
  const prim = cnt(isPrimarySource);
  const labelled = cnt((e) => e.confidence);
  const confirmedN = cnt((e) => ["confirmed", "确证"].includes(e.confidence));
  const secN = cnt((e) => ["secondary", "二手"].includes(e.confidence));
  const selfN = cnt((e) => ["self-reported", "自报"].includes(e.confidence));
  const models = all.filter((e) => ["model", "model-report"].includes(e.category));
  const md = models.length || 1;
  const mAsc = models.filter((e) => e.ascend).length;
  const aReady = models.filter((e) => e.ascend === "ready").length;
  const aPartial = models.filter((e) => e.ascend === "partial").length;
  const aNone = models.filter((e) => e.ascend === "none").length;
  const analysis = cnt((e) => e.analysis);

  const rows = [
    { lbl: "Primary sources", c: prim, d: n, hint: `${n - prim} 来自媒体/二手`,
      tip: "信源来自一手/官方渠道(arXiv · HuggingFace · GitHub · 官方文档/厂商页)的占比。越高 = 越可溯源。" },
    { lbl: "Confidence labelled", c: labelled, d: n, hint: `确证 ${confirmedN} · 二手 ${secN} · 自报 ${selfN}`,
      tip: "已标注信源可信度(确证 confirmed / 二手 secondary / 自报 self-reported)的条目占比。" },
    { lbl: "Ascend readiness", c: mAsc, d: md, hint: `就绪 ${aReady} · 部分 ${aPartial} · 无 ${aNone} · 共 ${models.length} 张模型卡`,
      tip: "在『模型卡』里已给出昇腾就绪度判断(就绪/部分/无)的占比。只对模型类条目计——算法/框架论文不适用。" },
    { lbl: "Deep analysis", c: analysis, d: n, hint: `${analysis}/${n} 带 ▸ 深析`,
      tip: "带 ▸ 深度分析段落(创新 / 性能 / 对昇腾意义)的条目占比,而非仅一句话摘要。" },
  ];
  const bar = (p) => `<span class="ledger-bar"><span style="width:${p}%"></span></span>`;
  el.innerHTML = `
    <div class="ledger-head">
      <h3>Data Quality Ledger</h3>
      <span class="dim">${all.length} curated entries (RL · Ascend · Modeling) · 悬停看含义</span>
    </div>
    <div class="ledger-grid">
      ${rows.map((r) => {
        const p = Math.round((r.c / r.d) * 100);
        return `<div class="ledger-cell" title="${escapeAttr(r.tip)}">
          <div class="ledger-top"><span>${escapeHtml(r.lbl)}</span><span class="ledger-pct">${p}%</span></div>
          ${bar(p)}
          <div class="ledger-hint dim">${r.c}/${r.d} · ${escapeHtml(r.hint)}</div>
        </div>`;
      }).join("")}
    </div>`;
}

/* ---------- Agentic RL trends block ---------- */
function renderAgenticTrends() {
  const el = document.getElementById("agentic-trends");
  if (!el) return;
  if (!agenticTrends.length) { el.innerHTML = ""; return; }
  el.innerHTML = `<div class="trend-head"><h3>本期趋势 · Trends</h3>
      <span class="dim">工业界为主,兼顾学术 · ${storeUpdated.agentic || ""}</span></div>
    <div class="trend-grid">${agenticTrends.map((t, i) => `<div class="trend-card">
      <div class="trend-num">${String(i + 1).padStart(2, "0")}</div>
      <h4>${escapeHtml(t.title)}</h4>
      <p>${escapeHtml(t.body)}</p>
    </div>`).join("")}</div>`;
}

/* industry / academic track badge */
function trackBadge(track) {
  if (!track) return "";
  const map = { industry: ["工业界", "ind"], academic: ["学术", "acad"] };
  const m = map[track];
  if (!m) return "";
  return `<span class="track ${m[1]}" title="来源:${m[0]}">${m[0]}</span>`;
}

/* ---------- filters ---------- */
function uniqueTags(key) {
  const tags = new Set();
  (store[key] || []).forEach((e) => {
    if (e.category) tags.add(e.category);
    if (e.ascend) tags.add("ascend:" + e.ascend);
    if (e.analysis || (window.hasSavedAnalysis && window.hasSavedAnalysis(e))) tags.add("analyzed");
    (e.tags || []).forEach((t) => tags.add(t));
  });
  return [...tags].sort();
}

function buildFilterbars() {
  document.querySelectorAll("[data-filterbar]").forEach((bar) => {
    const key = bar.dataset.filterbar;
    const tags = uniqueTags(key);
    bar.innerHTML =
      `<span class="chip ${activeFilters[key].size === 0 ? "active" : ""}" data-tag="__all">All</span>` +
      tags.map((t) => `<span class="chip" data-tag="${escapeAttr(t)}">${escapeHtml(t)}</span>`).join("");
    bar.querySelectorAll(".chip").forEach((chip) => {
      chip.addEventListener("click", () => {
        const tag = chip.dataset.tag;
        if (tag === "__all") {
          activeFilters[key].clear();
        } else {
          activeFilters[key].has(tag) ? activeFilters[key].delete(tag) : activeFilters[key].add(tag);
        }
        bar.querySelectorAll(".chip").forEach((c) => {
          if (c.dataset.tag === "__all") c.classList.toggle("active", activeFilters[key].size === 0);
          else c.classList.toggle("active", activeFilters[key].has(c.dataset.tag));
        });
        renderPanel(key);
      });
    });
  });
}

/* ---------- rendering ---------- */
function matchesFilter(key, entry) {
  const f = activeFilters[key];
  if (f && f.size > 0) {
    const entryTags = new Set([entry.category, ...(entry.tags || [])]);
    if (entry.ascend) entryTags.add("ascend:" + entry.ascend);
    if (entry.analysis || (window.hasSavedAnalysis && window.hasSavedAnalysis(entry))) entryTags.add("analyzed");
    let ok = false;
    f.forEach((t) => { if (entryTags.has(t)) ok = true; });
    if (!ok) return false;
  }
  if (searchTerm) {
    const hay = JSON.stringify(entry).toLowerCase();
    if (!hay.includes(searchTerm)) return false;
  }
  return true;
}

function renderPanel(key) {
  if (key === "ideas") return renderIdeas();
  const container = document.querySelector(`[data-cards="${key}"]`);
  if (!container) return;
  const items = (store[key] || []).filter((e) => matchesFilter(key, e));
  if (items.length === 0) {
    container.innerHTML = `<div class="empty">No entries match. ${store[key] && store[key].length ? "Adjust filters/search." : "Data not loaded yet."}</div>`;
    return;
  }
  const count = `<div class="result-count">${items.length} of ${store[key].length} shown${searchTerm ? ` · “${escapeHtml(searchTerm)}”` : ""}</div>`;
  container.innerHTML = count + items.map((e) => cardHTML(e, key)).join("");
}

/* derive a human source label from an explicit field or the URL host */
function inferSource(e) {
  if (e.source) return e.source;
  const u = e.url || "";
  const m = u.match(/^https?:\/\/([^/]+)/i);
  if (!m) return "";
  const host = m[1].replace(/^www\./, "");
  const MAP = {
    "arxiv.org": "arXiv", "huggingface.co": "Hugging Face", "github.com": "GitHub",
    "docs.vllm.ai": "vLLM docs", "seed.bytedance.com": "ByteDance Seed",
    "trendforce.com": "TrendForce", "tomshardware.com": "Tom's Hardware",
    "techpowerup.com": "TechPowerUp", "llm-stats.com": "llm-stats",
  };
  return MAP[host] || host;
}
function provenanceLine(e, key) {
  const src = inferSource(e);
  if (!src) return "";
  const verified = e.verified || storeUpdated[key] || "";
  return `<div class="provenance" title="数据来源与最近校验日期">source: ${escapeHtml(src)}${verified ? ` · verified ${escapeHtml(verified)}` : ""}</div>`;
}

function cardHTML(e, key) {
  const meta = [e.org, e.year].filter(Boolean).join(" · ");
  const tags = (e.tags || []).map((t) => `<span class="t">${escapeHtml(t)}</span>`).join("");
  const slim = JSON.stringify({ title: e.title, org: e.org, year: e.year, category: e.category, innovation: e.innovation, summary: e.summary, url: e.url, tags: e.tags, analysis: e.analysis });
  const analyzed = !!e.analysis || (window.hasSavedAnalysis && window.hasSavedAnalysis(e));
  return `<article class="card">
    <div class="card-top">${e.category ? `<span class="cat">${escapeHtml(e.category)}</span>` : "<span></span>"}<span class="card-badges">${trackBadge(e.track)}${confBadge(e)}${ascendBadge(e)}</span></div>
    <h3>${hl(e.title || "Untitled")}</h3>
    ${meta ? `<div class="meta">${escapeHtml(meta)}</div>` : ""}
    ${provenanceLine(e, key)}
    ${e.innovation ? `<div class="innov">▸ ${hl(e.innovation)}</div>` : ""}
    ${e.summary ? `<p class="summary">${hl(e.summary)}</p>` : ""}
    <div class="tags">${tags}</div>
    <div class="card-actions">
      ${e.url ? `<a class="link" href="${escapeAttr(e.url)}" target="_blank" rel="noopener">Open source ↗</a>` : "<span></span>"}
      <span class="card-act-right">
        ${analyzed ? `<span class="saved-badge" title="Analysis available">★ analyzed</span>` : ""}
        <button class="analyze-btn" data-entry="${escapeAttr(slim)}">${analyzed ? "✨ Analysis" : "✨ Analyze"}</button>
      </span>
    </div>
  </article>`;
}

// highlight the active search term inside escaped text
function hl(s) {
  const safe = escapeHtml(s);
  if (!searchTerm) return safe;
  try {
    const re = new RegExp("(" + searchTerm.replace(/[.*+?^${}()|[\]\\]/g, "\\$&") + ")", "ig");
    return safe.replace(re, "<mark>$1</mark>");
  } catch (_) { return safe; }
}

/* source-confidence badge: confirmed / secondary / self-reported */
function confBadge(e) {
  if (!e.confidence) return "";
  const map = {
    confirmed: ["确证", "conf"], secondary: ["二手", "sec"], "self-reported": ["自报", "self"],
    "确证": ["确证", "conf"], "二手": ["二手", "sec"], "自报": ["自报", "self"],
  };
  const m = map[e.confidence];
  if (!m) return "";
  return `<span class="conf ${m[1]}" title="信源可信度: ${escapeAttr(e.confidence)}">${m[0]}</span>`;
}

/* Ascend-readiness badge (✅ ready / ⚠️ partial / ❌ none) */
function ascendBadge(e) {
  if (!e.ascend) return "";
  const map = {
    ready: { icon: "✅", label: "Ascend-ready" },
    partial: { icon: "⚠️", label: "Ascend partial" },
    none: { icon: "❌", label: "Not on Ascend" },
  };
  const m = map[e.ascend];
  if (!m) return "";
  const note = escapeAttr(e.ascendNote || m.label);
  return `<span class="ascend-badge ${escapeAttr(e.ascend)}" title="${note}">${m.icon} ${escapeHtml(m.label)}</span>`;
}

function renderIdeas() {
  const container = document.querySelector('[data-cards="ideas"]');
  if (!container) return;
  const items = (store.ideas || []).filter((e) => matchesFilter("ideas", e));
  if (items.length === 0) {
    container.innerHTML = `<div class="empty">No ideas match.</div>`;
    return;
  }
  container.innerHTML = items.map(ideaHTML).join("");
}

function ideaHTML(e) {
  const bar = (cls, v) => `<div class="rating">${cls[0].toUpperCase() + cls.slice(1)}: ${v}/5
    <div class="bar ${cls}"><span style="width:${(v / 5) * 100}%"></span></div></div>`;
  const steps = (e.steps || []).map((s) => `<li>${escapeHtml(s)}</li>`).join("");
  return `<article class="idea">
    <h3>${escapeHtml(e.title)}</h3>
    <p class="pitch">${escapeHtml(e.pitch || "")}</p>
    <div class="ratings">
      ${bar("impact", e.impact ?? 0)}
      ${bar("difficulty", e.difficulty ?? 0)}
      ${bar("novelty", e.novelty ?? 0)}
    </div>
    ${e.why ? `<div class="why"><strong>Why now:</strong> ${escapeHtml(e.why)}</div>` : ""}
    ${e.minimumExperiment ? `<div class="why"><strong>Minimum experiment:</strong> ${escapeHtml(e.minimumExperiment)}</div>` : ""}
    ${e.successMetric ? `<div class="why"><strong>Success metric:</strong> ${escapeHtml(e.successMetric)}</div>` : ""}
    ${steps ? `<div class="why"><strong>First steps:</strong><ul>${steps}</ul></div>` : ""}
    ${(e.tags || []).length ? `<div class="tags">${(e.tags || []).map((t) => `<span class="t">${escapeHtml(t)}</span>`).join("")}</div>` : ""}
  </article>`;
}

/* ---------- tabs / search ---------- */
function activateTab(name, push) {
  const tab = document.querySelector(`.tab[data-tab="${name}"]`);
  const panel = document.getElementById(name);
  if (!tab || !panel) return;
  document.querySelectorAll(".tab").forEach((t) => {
    const on = t === tab;
    t.classList.toggle("active", on);
    t.setAttribute("aria-selected", on ? "true" : "false");
    t.tabIndex = on ? 0 : -1;
    if (on) { try { t.scrollIntoView({ inline: "nearest", block: "nearest" }); } catch (e) {} }
  });
  document.querySelectorAll(".panel").forEach((p) => {
    const on = p === panel;
    p.classList.toggle("active", on);
    if (on) p.removeAttribute("hidden"); else p.setAttribute("hidden", "");
  });
  if (push && location.hash.slice(1) !== name) history.replaceState(null, "", "#" + name);
}

function wireTabs() {
  const tabs = [...document.querySelectorAll(".tab")];
  tabs.forEach((tab) => {
    tab.addEventListener("click", () => activateTab(tab.dataset.tab, true));
  });
  // standard tablist keyboard interaction: ←/→/Home/End move focus + activate
  const tablist = document.getElementById("tabs");
  if (tablist) {
    tablist.addEventListener("keydown", (e) => {
      if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(e.key)) return;
      const cur = tabs.indexOf(document.activeElement);
      if (cur < 0) return;
      e.preventDefault();
      let next = cur;
      if (e.key === "ArrowRight") next = (cur + 1) % tabs.length;
      else if (e.key === "ArrowLeft") next = (cur - 1 + tabs.length) % tabs.length;
      else if (e.key === "Home") next = 0;
      else if (e.key === "End") next = tabs.length - 1;
      tabs[next].focus();
      activateTab(tabs[next].dataset.tab, true);
    });
  }
  // deep-link: open the tab named in the URL hash, and react to back/forward
  const fromHash = () => { const h = location.hash.slice(1).split("/")[0]; if (h && document.getElementById(h)) activateTab(h, false); };
  window.addEventListener("hashchange", fromHash);
  fromHash();
  // number keys 1..9 jump to tabs (when not typing in a field)
  document.addEventListener("keydown", (e) => {
    if (!/^[1-9]$/.test(e.key)) return;
    const el = document.activeElement;
    if (el && /INPUT|TEXTAREA|SELECT/.test(el.tagName)) return;
    if (tabs[+e.key - 1]) tabs[+e.key - 1].click();
  });
}

/* mobile: fade hints on the .tabs strip when it can scroll left/right (CSS draws the fades) */
function wireTabsScrollHint() {
  const tabs = document.querySelector(".tabs");
  if (!tabs) return;
  const upd = () => {
    const canL = tabs.scrollLeft > 4;
    const canR = tabs.scrollLeft + tabs.clientWidth < tabs.scrollWidth - 4;
    tabs.classList.toggle("fade-l", canL);
    tabs.classList.toggle("fade-r", canR);
  };
  let ticking = false;
  tabs.addEventListener("scroll", () => {
    if (ticking) return;
    ticking = true;
    requestAnimationFrame(() => { ticking = false; upd(); });
  }, { passive: true });
  window.addEventListener("resize", upd);
  upd();
  // bring the active tab into view on load (only when the strip actually overflows)
  if (tabs.scrollWidth > tabs.clientWidth) {
    const active = tabs.querySelector(".tab.active");
    if (active) { try { active.scrollIntoView({ inline: "center", block: "nearest" }); } catch (e) {} }
  }
}

function updateTabCounts() {
  ["rl", "ascend", "modeling", "agentic", "ideas", "live"].forEach((key) => {
    const tab = document.querySelector(`.tab[data-tab="${key === "live" ? "live" : key}"]`);
    if (!tab) return;
    let sup = tab.querySelector(".tab-count");
    if (!searchTerm) { if (sup) sup.remove(); return; }
    const n = (store[key] || []).filter((e) => matchesFilter(key, e)).length;
    if (!sup) { sup = document.createElement("sup"); sup.className = "tab-count"; tab.appendChild(sup); }
    sup.textContent = n;
    sup.classList.toggle("zero", n === 0);
  });
  // blog tab badge — counts matching Dispatch posts
  const blogTab = document.querySelector('.tab[data-tab="blog"]');
  if (blogTab) {
    let sup = blogTab.querySelector(".tab-count");
    if (!searchTerm) { if (sup) sup.remove(); return; }
    const n = window.__blogMatches
      ? (window.__blogPosts || []).filter(window.__blogMatches).length
      : (window.__blogPosts || []).filter((p) =>
          (p.title + " " + (p.subtitle || "") + " " + (p.tags || []).join(" ")).toLowerCase().includes(searchTerm)).length;
    if (!sup) { sup = document.createElement("sup"); sup.className = "tab-count"; blogTab.appendChild(sup); }
    sup.textContent = n;
    sup.classList.toggle("zero", n === 0);
  }
}

function wireSearch() {
  const box = document.getElementById("search");
  box.addEventListener("input", () => {
    searchTerm = box.value.trim().toLowerCase();
    ["rl", "ascend", "modeling", "agentic", "ideas", "live"].forEach(renderPanel);
    window.__refreshBlogIndex && window.__refreshBlogIndex();
    updateTabCounts();
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "/" && document.activeElement !== box) { e.preventDefault(); box.focus(); }
    if (e.key === "Escape" && document.activeElement === box && box.value) {
      box.value = ""; searchTerm = "";
      ["rl", "ascend", "modeling", "agentic", "ideas", "live"].forEach(renderPanel);
      window.__refreshBlogIndex && window.__refreshBlogIndex();
      updateTabCounts();
    }
  });
}

/* ---------- live papers (best-effort client-side) ---------- */
function wireLive() {
  const btn = document.getElementById("live-refresh");
  const reloadBtn = document.getElementById("live-reload");
  const status = document.getElementById("live-status");
  if (store.live && store.live.length) {
    status.textContent = `Showing ${store.live.length} entries from data/feed.json`;
  } else {
    status.textContent = "No cached feed. Run scripts/fetch_arxiv.py or the GitHub Action.";
  }
  if (reloadBtn) {
    reloadBtn.addEventListener("click", async () => {
      status.textContent = "Reloading data/feed.json…";
      const payload = await loadJSON(DATA_SOURCES.live + "?t=" + Date.now());
      const items = payload && (payload.items || (Array.isArray(payload) ? payload : null));
      if (items) {
        store.live = items;
        renderPanel("live");
        buildFilterbars();
        const upd = payload.updated ? ` · updated ${String(payload.updated).slice(0, 10)}` : "";
        status.textContent = `Loaded ${items.length} entries${upd}.`;
      } else {
        status.textContent = "Couldn't load data/feed.json.";
      }
    });
  }
  btn.addEventListener("click", async () => {
    status.textContent = "Trying arXiv (may be blocked by CORS in-browser)…";
    const fresh = await tryArxivLive();
    if (fresh && fresh.length) {
      store.live = fresh;
      renderPanel("live");
      buildFilterbars();
      status.textContent = `Fetched ${fresh.length} live entries.`;
    } else {
      status.textContent = "Live in-browser fetch blocked (CORS). Use scripts/fetch_arxiv.py to refresh data/feed.json instead.";
    }
  });
}

async function tryArxivLive() {
  // arXiv API has no CORS headers; this will usually fail in-browser — that's expected.
  const q = encodeURIComponent('(abs:"reinforcement learning" AND abs:LLM) OR abs:"Ascend NPU"');
  const url = `https://export.arxiv.org/api/query?search_query=${q}&sortBy=submittedDate&sortOrder=descending&max_results=20`;
  try {
    const res = await fetch(url);
    const xml = await res.text();
    const doc = new DOMParser().parseFromString(xml, "text/xml");
    return [...doc.querySelectorAll("entry")].map((en) => ({
      title: text(en, "title"),
      org: (en.querySelector("author name") || {}).textContent || "",
      year: (text(en, "published") || "").slice(0, 10),
      summary: (text(en, "summary") || "").trim().slice(0, 280) + "…",
      url: (en.querySelector("id") || {}).textContent || "",
      category: "arxiv-live",
      tags: ["live"],
    }));
  } catch (e) {
    return null;
  }
}
const text = (el, sel) => { const n = el.querySelector(sel); return n ? n.textContent : ""; };

/* ---------- 2026 model comparison (Overview) ---------- */
async function wireCompare() {
  const wrap = document.getElementById("compare-wrap");
  if (!wrap) return;
  const d = await loadJSON("data/compare.json");
  if (!d) return;

  const table = (t, opts = {}) => {
    const head = `<tr>${t.columns.map((c, i) => `<th${i === 0 && opts.firstCol ? ' class="rowhead"' : ""}>${escapeHtml(c)}</th>`).join("")}</tr>`;
    const body = t.rows.map((r) => `<tr>${r.map((cell, i) => {
      const tag = i === 0 ? "th" : "td";
      const cls = i === 0 ? ' class="rowhead"' : (opts.badge && i > 0 && /^(ready|partial|none)$/.test(cell) ? ` class="cmp-${cell}"` : "");
      const txt = (opts.badge && /^(ready|partial|none)$/.test(cell))
        ? ({ ready: "✅ ready", partial: "⚠️ partial", none: "❌ none" }[cell]) : cell;
      return `<${tag}${cls}>${escapeHtml(txt)}</${tag}>`;
    }).join("")}</tr>`).join("");
    return `<div class="cmp-scroll"><table class="cmp">${head}${body}</table></div>`;
  };

  const takeaways = (d.takeaways || []).map((t) => `<li>${escapeHtml(t)}</li>`).join("");
  const M = d.metrics;
  const sel = M ? `<select id="cmp-metric" class="ghost-btn">${M.list.map((m) =>
    `<option value="${escapeAttr(m.id)}"${m.id === M.default ? " selected" : ""}>${escapeHtml(m.label)}</option>`).join("")}</select>` : "";
  wrap.innerHTML = `
    <div class="cmp-head">
      <h3>2026 frontier models · quick compare</h3>
      <a class="link" href="${escapeAttr(d.doc || "#")}" target="_blank" rel="noopener">Full deep-dive ↗</a>
    </div>
    ${table(d.models, { badge: true })}
    ${M ? `<div class="cmp-metricbar"><h4 class="cmp-sub" style="margin:0">Benchmarks</h4>${sel}</div>
    <div class="swe-legend"><span><i class="sw open"></i>open-weight</span><span><i class="sw closed"></i>closed (ref)</span></div>
    <div id="metric-chart"></div>` : ""}
    <h4 class="cmp-sub">Sparse-attention designs (the 2026 dividing line)</h4>
    ${table(d.attention)}
    ${takeaways ? `<div class="cmp-take"><strong>For RL-on-NPU:</strong><ul>${takeaways}</ul></div>` : ""}
    <div class="cmp-note">${escapeHtml(d.note || "")} <span class="dim">* provisional</span></div>`;

  if (M) {
    const byId = Object.fromEntries(M.list.map((m) => [m.id, m]));
    const draw = (id) => { document.getElementById("metric-chart").innerHTML = metricBars(byId[id]); };
    document.getElementById("cmp-metric").addEventListener("change", (e) => draw(e.target.value));
    draw(M.default);
  }
}

function fmtScore(m, v) {
  if (m.unit === "tok") return v >= 1e6 ? (v / 1e6) + "M" : (v / 1e3) + "K";
  if (m.unit === "$") return "$" + v;
  if (m.unit === "%") return v + "%";
  return "" + v;
}

function metricBars(m) {
  if (!m) return "";
  const items = m.items.slice().sort((a, b) => m.lowerBetter ? a.score - b.score : b.score - a.score);
  const max = m.unit === "%" ? Math.max(100, ...items.map((i) => i.score)) : Math.max(...items.map((i) => i.score));
  const fillClass = m.lowerBetter ? "price" : (m.neutral ? "neutral" : null);
  const rows = items.map((i) => {
    const w = max ? (i.score / max) * 100 : 0;
    const cls = fillClass || (i.open ? "open" : "closed");
    return `<div class="bar-row">
      <span class="bar-label">${escapeHtml(i.model)}</span>
      <span class="bar-track"><span class="bar-fill ${cls}" style="width:${w.toFixed(1)}%"></span></span>
      <span class="bar-val">${escapeHtml(fmtScore(m, i.score))}</span>
    </div>`;
  }).join("");
  const tag = m.lowerBetter ? " · lower is better" : (m.neutral ? "" : " · higher is better");
  return `<div class="swe-bars">${rows}</div>
    <div class="cmp-note">${escapeHtml(m.label + " (" + m.unit + ")" + tag)} — ${escapeHtml(m.note || "")}</div>`;
}

/* ---------- training curves (SVG, no deps) ---------- */
let curvesData = null;
const METRIC_LABELS = {
  reward_mean: "Reward (mean)", kl: "KL", entropy: "Entropy",
  response_length: "Response length",
};
const DEVICE_COLORS = { gpu: "#5ad1a0", npu: "#f0a85a" };
const FALLBACK_COLORS = ["#6aa6ff", "#c98bff", "#ff8a8a", "#e2c84b"];
const metricLabel = (m) => METRIC_LABELS[m] || m;
const curveColor = (device, i) => DEVICE_COLORS[device] || FALLBACK_COLORS[i % FALLBACK_COLORS.length];

async function wireCurves() {
  const status = document.getElementById("curve-status");
  const sel = document.getElementById("curve-metric");
  if (!sel) return;
  const payload = await loadJSON("data/curves.json");
  if (!payload || !payload.experiments || !payload.experiments.length) {
    if (status) status.textContent = "No data/curves.json yet — run tools/logs_to_dashboard.py --synthetic.";
    return;
  }
  curvesData = payload;
  renderCurveMeta(payload);
  renderPublishedCurves(payload);
  const metrics = [];
  payload.experiments.forEach((e) =>
    Object.keys(e.metrics || {}).forEach((m) => { if (!metrics.includes(m)) metrics.push(m); }));
  sel.innerHTML = metrics.map((m) => `<option value="${escapeAttr(m)}">${escapeHtml(metricLabel(m))}</option>`).join("");
  sel.addEventListener("change", () => renderCurve(sel.value));
  if (status) {
    const upd = payload.updated ? ` · updated ${String(payload.updated).slice(0, 10)}` : "";
    const published = (payload.publishedEvidence || []).length;
    const real = published ? ` · ${published} published real run(s)` : "";
    status.textContent = `${payload.experiments.length} interactive run(s)${real}${upd}`;
  }
  if (metrics.length) renderCurve(metrics[0]);
}

/* Experiment metadata panel — makes provenance of the curves explicit */
function renderCurveMeta(payload) {
  const el = document.getElementById("curve-meta");
  if (!el) return;
  const exps = payload.experiments || [];
  const isSynthetic = payload.synthetic || exps.some((e) => e.meta && e.meta.synthetic);
  const COLS = [
    ["model", "Model"], ["dataset", "Dataset"], ["hardware", "Hardware"],
    ["framework", "Framework"], ["precision", "Precision"], ["seed", "Seed"],
  ];
  const banner = isSynthetic
    ? `<div class="curve-synth">⚠ Synthetic demo data — illustrative shapes, not measured results. Replace via <code>logs_to_dashboard.py --log train.log</code>.</div>`
    : `<div class="curve-synth real">✓ Parsed from real training logs.</div>`;
  const head = `<tr><th>Run</th>${COLS.map(([, l]) => `<th>${escapeHtml(l)}</th>`).join("")}</tr>`;
  const rows = exps.map((e) => {
    const m = e.meta || {};
    const cells = COLS.map(([k]) => `<td>${m[k] !== undefined && m[k] !== "" ? escapeHtml(String(m[k])) : "—"}</td>`).join("");
    return `<tr><td class="rowhead">${escapeHtml(e.name)} · ${escapeHtml(e.device || "")}</td>${cells}</tr>`;
  }).join("");
  el.innerHTML = `${banner}<div class="cmp-scroll"><table class="cmp">${head}${rows}</table></div>`;
}

function renderPublishedCurves(payload) {
  const el = document.getElementById("curve-evidence");
  if (!el) return;
  const runs = payload.publishedEvidence || [];
  if (!runs.length) {
    el.innerHTML = "";
    return;
  }
  const runHtml = runs.map((run) => {
    const charts = (run.charts || []).map((chart) => `<figure>
      <a href="${escapeAttr(chart.originalUrl || chart.asset)}" target="_blank" rel="noopener">
        <img src="${escapeAttr(chart.asset)}" alt="${escapeAttr(chart.metric)} from ${escapeAttr(run.hardware)}" loading="lazy">
      </a>
      <figcaption>${escapeHtml(chart.metric)}</figcaption>
    </figure>`).join("");
    const facts = [
      ["Model", run.model], ["Algorithm", run.algorithm], ["Workload", run.dataset],
      ["Hardware", run.hardware], ["Stack", run.stack], ["Reported", run.reported],
    ].map(([label, value]) => `<div><dt>${escapeHtml(label)}</dt><dd>${escapeHtml(value || "-")}</dd></div>`).join("");
    return `<article class="curve-evidence-run">
      <div class="curve-evidence-title">
        <div><span class="evidence-badge">REAL 910B TELEMETRY</span><h4>${escapeHtml(run.name)}</h4></div>
        <a href="${escapeAttr(run.sourceUrl)}" target="_blank" rel="noopener">Open primary source &#8599;</a>
      </div>
      <dl class="curve-evidence-facts">${facts}</dl>
      <div class="curve-evidence-grid">${charts}</div>
      <p class="curve-evidence-caveat"><strong>Scope:</strong> ${escapeHtml(run.caveat || "")}</p>
      <p class="provenance">${escapeHtml(run.sourceType || "published run")} · verified ${escapeHtml(run.verified || "-")} · point-level samples ${run.rawSamples ? "available" : "not published"}</p>
    </article>`;
  }).join("");
  el.innerHTML = `<div class="curve-evidence-head"><h3>Published 910B evidence</h3><span>Source-preserving snapshots; not digitized into synthetic points.</span></div>${runHtml}`;
}

function renderCurve(metric) {
  const chart = document.getElementById("curve-chart");
  const legend = document.getElementById("curve-legend");
  if (!chart || !curvesData) return;
  const series = [];
  curvesData.experiments.forEach((e, i) => {
    const pts = (e.metrics || {})[metric];
    if (pts && pts.length) series.push({ label: `${e.name} · ${e.device}`, color: curveColor(e.device, i), points: pts });
  });
  if (!series.length) {
    chart.innerHTML = `<div class="empty">No data for ${escapeHtml(metric)}.</div>`;
    legend.innerHTML = "";
    return;
  }
  chart.innerHTML = svgLineChart(series, metricLabel(metric));
  legend.innerHTML = series.map((s) =>
    `<span class="lg"><span class="sw" style="background:${s.color}"></span>${escapeHtml(s.label)}</span>`).join("");
}

function svgLineChart(series, title) {
  const W = 860, H = 340, padL = 52, padR = 16, padT = 18, padB = 36;
  const xs = [], ys = [];
  series.forEach((s) => s.points.forEach((p) => { xs.push(p[0]); ys.push(p[1]); }));
  let xmin = Math.min(...xs), xmax = Math.max(...xs);
  let ymin = Math.min(...ys), ymax = Math.max(...ys);
  if (xmin === xmax) xmax = xmin + 1;
  if (ymin === ymax) { ymin -= 1; ymax += 1; }
  const pad = (ymax - ymin) * 0.08; ymin -= pad; ymax += pad;
  const X = (x) => padL + ((x - xmin) / (xmax - xmin)) * (W - padL - padR);
  const Y = (y) => H - padB - ((y - ymin) / (ymax - ymin)) * (H - padT - padB);
  const fmt = (v) => (Math.abs(v) >= 1000 || (v !== 0 && Math.abs(v) < 0.01)) ? v.toPrecision(3) : (Math.round(v * 1000) / 1000);

  let g = "";
  for (let i = 0; i <= 4; i++) {
    const yv = ymin + (i / 4) * (ymax - ymin), y = Y(yv);
    g += `<line class="grid" x1="${padL}" y1="${y}" x2="${W - padR}" y2="${y}"/>`;
    g += `<text class="tick" x="${padL - 6}" y="${y + 3}" text-anchor="end">${fmt(yv)}</text>`;
  }
  [xmin, (xmin + xmax) / 2, xmax].forEach((xv) => {
    g += `<text class="tick" x="${X(xv)}" y="${H - padB + 16}" text-anchor="middle">${Math.round(xv)}</text>`;
  });
  const axes = `<line class="axis" x1="${padL}" y1="${padT}" x2="${padL}" y2="${H - padB}"/>` +
    `<line class="axis" x1="${padL}" y1="${H - padB}" x2="${W - padR}" y2="${H - padB}"/>`;
  const lines = series.map((s) => {
    const d = s.points.map((p) => `${X(p[0]).toFixed(1)},${Y(p[1]).toFixed(1)}`).join(" ");
    return `<polyline class="ser" points="${d}" stroke="${s.color}"/>`;
  }).join("");
  const labels = `<text class="tick" x="${padL}" y="${padT - 4}">${escapeHtml(title)}</text>` +
    `<text class="tick" x="${W - padR}" y="${H - 4}" text-anchor="end">step →</text>`;
  return `<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="xMidYMid meet" role="img" aria-label="${escapeAttr(title)} curve">${g}${axes}${lines}${labels}</svg>`;
}

/* ---------- utils ---------- */
function setLastUpdated() {
  let stamp = null;
  for (const k of ["rl", "ascend", "modeling"]) {
    const p = store[k + "_meta"];
    if (p && p.updated) stamp = p.updated;
  }
  document.getElementById("last-updated").textContent =
    "Curated content snapshot · refresh Live Papers for the newest arXiv entries.";
}
function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
function escapeAttr(s) { return escapeHtml(s); }

// re-render all card panels (used by analyst.js after saving an analysis)
window.refreshDashboard = () => ["rl", "ascend", "modeling", "live"].forEach(renderPanel);

init();
