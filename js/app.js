/* RL-on-NPU Research Dashboard — vanilla JS, no build step. */

const DATA_SOURCES = {
  rl: "data/rl.json",
  ascend: "data/ascend.json",
  modeling: "data/modeling.json",
  ideas: "data/ideas.json",
  live: "data/feed.json",
};

const store = {}; // key -> array of entries
const activeFilters = {}; // key -> Set of active tags
let searchTerm = "";

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
    store[k + "_meta"] = payload && !Array.isArray(payload) ? payload : {};
    activeFilters[k] = new Set();
  });

  wireTheme();
  buildStats();
  ["rl", "ascend", "modeling", "ideas", "live"].forEach(renderPanel);
  buildFilterbars();
  wireTabs();
  wireSearch();
  wireLive();
  wireCurves();
  wireCompare();
  wireTimeline();
  wireBlog();
  if (window.wireArch) wireArch();
  setLastUpdated();
}

/* ---------- light/dark theme toggle (default: light) ---------- */
function wireTheme() {
  const btn = document.getElementById("theme-toggle");
  const root = document.documentElement;
  const apply = (t) => {
    if (t === "dark") root.setAttribute("data-theme", "dark");
    else root.removeAttribute("data-theme");
    if (btn) { btn.textContent = t === "dark" ? "☀️" : "🌙"; btn.title = t === "dark" ? "切换到浅色" : "切换到深色"; }
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

/* ---------- blog / dispatch (markdown posts, unipat-style index) ---------- */
async function wireBlog() {
  const idx = document.getElementById("blog-index");
  const post = document.getElementById("blog-post");
  if (!idx || !post) return;
  let posts = [];
  try {
    const res = await fetch("data/blog.json", { cache: "no-cache" });
    if (!res.ok) throw new Error(res.status);
    posts = (await res.json()).posts || [];
  } catch (e) {
    idx.innerHTML = `<div class="empty">Couldn't load blog index (${escapeHtml(String(e.message || e))}).</div>`;
    return;
  }

  function showIndex() {
    post.hidden = true; idx.hidden = false;
    if (!posts.length) { idx.innerHTML = `<div class="empty">No posts yet.</div>`; return; }
    idx.innerHTML = posts.map((p) => `<a class="blog-card" href="#blog/${escapeAttr(p.id)}">
      <div class="blog-card-date">${escapeHtml(p.date || "")}</div>
      <h3>${escapeHtml(p.title)}</h3>
      <p>${escapeHtml(p.subtitle || "")}</p>
      <div class="blog-card-tags">${(p.tags || []).map((t) => `<span class="tag">${escapeHtml(t)}</span>`).join("")}</div>
    </a>`).join("");
  }

  async function openPost(id) {
    const p = posts.find((x) => x.id === id);
    if (!p) { showIndex(); return; }
    idx.hidden = true; post.hidden = false;
    post.innerHTML = `<div class="empty">Loading…</div>`;
    try {
      const res = await fetch(p.file, { cache: "no-cache" });
      if (!res.ok) throw new Error(res.status);
      post.innerHTML = `<a class="blog-back" href="#blog">← 所有 Dispatch</a>`
        + `<div class="prose blog-prose">${renderMarkdown(await res.text())}</div>`
        + `<a class="blog-back foot" href="#blog">← 所有 Dispatch</a>`;
    } catch (e) {
      post.innerHTML = `<a class="blog-back" href="#blog">← 返回</a><div class="empty">Couldn't load post (${escapeHtml(String(e.message || e))}).</div>`;
    }
    window.scrollTo(0, 0);
  }

  function route() {
    const parts = location.hash.slice(1).split("/");
    if (parts[0] !== "blog") return;       // only react to our own tab
    if (parts[1]) openPost(parts[1]); else showIndex();
  }
  window.addEventListener("hashchange", route);
  showIndex();
  route();
}

/* ---------- timeline (aggregated from curated data) ---------- */
function wireTimeline() {
  const el = document.getElementById("timeline-body");
  if (!el) return;
  const DOMAINS = { rl: "RL", ascend: "Ascend", modeling: "Modeling" };
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
    { num: store.ideas.length, lbl: "Project ideas" },
  ];
  grid.innerHTML = stats
    .map((s) => `<div class="stat"><div class="num">${s.num}</div><div class="lbl">${s.lbl}</div></div>`)
    .join("");
  buildTrustPanel();
}

function buildTrustPanel() {
  const el = document.getElementById("trust-panel");
  if (!el) return;
  const keys = ["rl", "ascend", "modeling", "live"];
  const rows = keys.flatMap((key) => (store[key] || []).map((entry) => ({ key, entry })));
  const total = rows.length || 1;
  const sourced = rows.filter(({ entry }) => entry.url).length;
  const confidence = rows.filter(({ entry }) => entry.confidence).length;
  const ascend = rows.filter(({ entry }) => entry.ascend).length;
  const analyzed = rows.filter(({ entry }) => entry.analysis || (window.hasSavedAnalysis && window.hasSavedAnalysis(entry))).length;
  const updated = keys
    .map((key) => store[key + "_meta"] && store[key + "_meta"].updated)
    .filter(Boolean)
    .sort()
    .reverse()[0];
  const pct = (n) => Math.round((n / total) * 100);
  const items = [
    ["Source links", `${sourced}/${rows.length}`, `${pct(sourced)}% cards point to a primary or tracking URL.`],
    ["Confidence labels", `${confidence}/${rows.length}`, "Use confirmed / secondary / self-reported before citing."],
    ["Ascend status", `${ascend}/${rows.length}`, "Cards with ready / partial / none portability notes."],
    ["Analyst notes", `${analyzed}/${rows.length}`, "Hand-written or saved analysis attached to cards."],
  ];
  el.innerHTML = `<div class="trust-head">
      <div>
        <h3>Data Quality Ledger</h3>
        <p>Every card now exposes its source type and snapshot date. Treat live model metrics as provisional unless the badge says confirmed.</p>
      </div>
      <span class="trust-date">Latest snapshot ${escapeHtml(updated || "unknown")}</span>
    </div>
    <div class="trust-grid">${items.map(([label, value, note]) => `<div class="trust-item">
      <div class="trust-val">${escapeHtml(value)}</div>
      <div class="trust-label">${escapeHtml(label)}</div>
      <p>${escapeHtml(note)}</p>
    </div>`).join("")}</div>
    <div class="trust-legend">
      <span class="conf conf">确证</span><span>primary/official or directly checked</span>
      <span class="conf sec">二手</span><span>media/community sourced</span>
      <span class="conf self">自报</span><span>vendor or model-card claim</span>
    </div>`;
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

function cardHTML(e, key) {
  const meta = [e.org, e.year].filter(Boolean).join(" · ");
  const tags = (e.tags || []).map((t) => `<span class="t">${escapeHtml(t)}</span>`).join("");
  const slim = JSON.stringify({ title: e.title, org: e.org, year: e.year, category: e.category, innovation: e.innovation, summary: e.summary, url: e.url, tags: e.tags, analysis: e.analysis, confidence: e.confidence, ascend: e.ascend });
  const analyzed = !!e.analysis || (window.hasSavedAnalysis && window.hasSavedAnalysis(e));
  return `<article class="card">
    <div class="card-top">${e.category ? `<span class="cat">${escapeHtml(e.category)}</span>` : "<span></span>"}<span class="card-badges">${confBadge(e)}${ascendBadge(e)}</span></div>
    <h3>${hl(e.title || "Untitled")}</h3>
    ${meta ? `<div class="meta">${escapeHtml(meta)}</div>` : ""}
    ${sourceMetaHTML(e, key)}
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

function sourceMetaHTML(e, key) {
  const meta = store[key + "_meta"] || {};
  const verified = e.verified || e.lastVerified || e.checked || meta.updated;
  const pieces = [];
  if (e.url) pieces.push(sourceKind(e.url));
  if (verified) pieces.push(`verified ${String(verified).slice(0, 10)}`);
  if (e.sourceType) pieces.push(e.sourceType);
  if (!pieces.length) return "";
  return `<div class="source-line" title="Source provenance for this snapshot">${pieces.map((p) => `<span>${escapeHtml(p)}</span>`).join("")}</div>`;
}

function sourceKind(url) {
  try {
    const u = new URL(url, location.href);
    const host = u.hostname.replace(/^www\./, "");
    if (host === "arxiv.org") return "source: arXiv";
    if (host === "github.com" || host === "gitee.com" || host === "gitcode.com") return `source: ${host.split(".")[0]}`;
    if (host.includes("huggingface.co")) return "source: Hugging Face";
    if (host.includes("hiascend.com")) return "source: Ascend docs";
    if (host.includes("docs.vllm.ai")) return "source: docs";
    if (u.pathname.startsWith("/mission44/docs/") || u.pathname.startsWith("/docs/")) return "source: local note";
    return `source: ${host}`;
  } catch (_) {
    return String(url).startsWith("docs/") ? "source: local note" : "source: linked";
  }
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
  const mvp = e.minimumExperiment || e.mvp;
  const success = e.successMetric || e.acceptanceCriteria;
  return `<article class="idea">
    <h3>${escapeHtml(e.title)}</h3>
    <p class="pitch">${escapeHtml(e.pitch || "")}</p>
    <div class="ratings">
      ${bar("impact", e.impact ?? 0)}
      ${bar("difficulty", e.difficulty ?? 0)}
      ${bar("novelty", e.novelty ?? 0)}
    </div>
    ${e.why ? `<div class="why"><strong>Why now:</strong> ${escapeHtml(e.why)}</div>` : ""}
    ${mvp || success ? `<div class="idea-proof">
      ${mvp ? `<div><strong>Minimum experiment</strong><span>${escapeHtml(mvp)}</span></div>` : ""}
      ${success ? `<div><strong>Success signal</strong><span>${escapeHtml(success)}</span></div>` : ""}
    </div>` : ""}
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
    const isActive = t === tab;
    t.classList.toggle("active", isActive);
    t.setAttribute("aria-selected", isActive ? "true" : "false");
    t.tabIndex = isActive ? 0 : -1;
  });
  document.querySelectorAll(".panel").forEach((p) => {
    const isActive = p === panel;
    p.classList.toggle("active", isActive);
    p.hidden = !isActive;
  });
  tab.classList.add("active");
  panel.classList.add("active");
  panel.hidden = false;
  if (push && location.hash.slice(1) !== name) history.replaceState(null, "", "#" + name);
}

function wireTabs() {
  document.querySelectorAll(".tab").forEach((tab) => {
    tab.addEventListener("click", () => activateTab(tab.dataset.tab, true));
    tab.addEventListener("keydown", (e) => {
      if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(e.key)) return;
      e.preventDefault();
      const tabs = [...document.querySelectorAll(".tab")];
      const i = tabs.indexOf(tab);
      const next = e.key === "Home" ? tabs[0]
        : e.key === "End" ? tabs[tabs.length - 1]
        : tabs[(i + (e.key === "ArrowRight" ? 1 : -1) + tabs.length) % tabs.length];
      next.focus();
      activateTab(next.dataset.tab, true);
    });
  });
  // deep-link: open the tab named in the URL hash, and react to back/forward
  const fromHash = () => { const h = location.hash.slice(1).split("/")[0]; if (h && document.getElementById(h)) activateTab(h, false); };
  window.addEventListener("hashchange", fromHash);
  fromHash();
  // number keys 1..9 jump to tabs (when not typing in a field)
  document.addEventListener("keydown", (e) => {
    if (!/^[1-9]$/.test(e.key)) return;
    const el = document.activeElement;
    if (el && /INPUT|TEXTAREA|SELECT/.test(el.tagName)) return;
    const tabs = [...document.querySelectorAll(".tab")];
    if (tabs[+e.key - 1]) tabs[+e.key - 1].click();
  });
}

function updateTabCounts() {
  ["rl", "ascend", "modeling", "ideas", "live"].forEach((key) => {
    const tab = document.querySelector(`.tab[data-tab="${key === "live" ? "live" : key}"]`);
    if (!tab) return;
    let sup = tab.querySelector(".tab-count");
    if (!searchTerm) { if (sup) sup.remove(); return; }
    const n = (store[key] || []).filter((e) => matchesFilter(key, e)).length;
    if (!sup) { sup = document.createElement("sup"); sup.className = "tab-count"; tab.appendChild(sup); }
    sup.textContent = n;
    sup.classList.toggle("zero", n === 0);
  });
}

function wireSearch() {
  const box = document.getElementById("search");
  box.addEventListener("input", () => {
    searchTerm = box.value.trim().toLowerCase();
    ["rl", "ascend", "modeling", "ideas", "live"].forEach(renderPanel);
    updateTabCounts();
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "/" && document.activeElement !== box) { e.preventDefault(); box.focus(); }
    if (e.key === "Escape" && document.activeElement === box && box.value) {
      box.value = ""; searchTerm = ""; ["rl", "ascend", "modeling", "ideas", "live"].forEach(renderPanel); updateTabCounts();
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
        store.live_meta = payload && !Array.isArray(payload) ? payload : {};
        renderPanel("live");
        buildFilterbars();
        buildTrustPanel();
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
      store.live_meta = { updated: new Date().toISOString(), source: "browser-arxiv-attempt" };
      renderPanel("live");
      buildFilterbars();
      buildTrustPanel();
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
  const metrics = [];
  payload.experiments.forEach((e) =>
    Object.keys(e.metrics || {}).forEach((m) => { if (!metrics.includes(m)) metrics.push(m); }));
  sel.innerHTML = metrics.map((m) => `<option value="${escapeAttr(m)}">${escapeHtml(metricLabel(m))}</option>`).join("");
  sel.addEventListener("change", () => renderCurve(sel.value));
  if (status) {
    const upd = payload.updated ? ` · updated ${String(payload.updated).slice(0, 10)}` : "";
    const kind = payload.provenance && payload.provenance.kind ? ` · ${payload.provenance.kind}` : "";
    status.textContent = `${payload.experiments.length} run(s)${upd}${kind}`;
  }
  if (metrics.length) renderCurve(metrics[0]);
}

function renderCurve(metric) {
  const chart = document.getElementById("curve-chart");
  const legend = document.getElementById("curve-legend");
  if (!chart || !curvesData) return;
  const series = [];
  curvesData.experiments.forEach((e, i) => {
    const pts = (e.metrics || {})[metric];
    if (pts && pts.length) series.push({ label: `${e.name} · ${e.device}`, color: curveColor(e.device, i), points: pts, meta: e });
  });
  if (!series.length) {
    chart.innerHTML = `<div class="empty">No data for ${escapeHtml(metric)}.</div>`;
    legend.innerHTML = "";
    renderCurveMeta(metric, []);
    return;
  }
  chart.innerHTML = svgLineChart(series, metricLabel(metric));
  legend.innerHTML = series.map((s) =>
    `<span class="lg"><span class="sw" style="background:${s.color}"></span>${escapeHtml(s.label)}</span>`).join("");
  renderCurveMeta(metric, series);
}

function renderCurveMeta(metric, series) {
  const el = document.getElementById("curve-meta");
  if (!el) return;
  if (!series.length) { el.innerHTML = ""; return; }
  const prov = curvesData.provenance || {};
  const provenance = [
    prov.kind,
    prov.generator,
    curvesData.updated ? `updated ${String(curvesData.updated).slice(0, 10)}` : "",
  ].filter(Boolean).join(" · ");
  const rows = series.map((s) => {
    const e = s.meta || {};
    const fields = [
      ["hardware", e.hardware],
      ["framework", e.framework],
      ["model", e.model],
      ["dataset", e.dataset],
      ["precision", e.precision],
      ["seed", e.seed],
      ["commit", e.commit],
    ].filter(([, v]) => v !== undefined && v !== null && String(v).trim() !== "");
    return `<div class="curve-run">
      <div class="curve-run-title"><span class="sw" style="background:${s.color}"></span>${escapeHtml(s.label)}</div>
      <div class="curve-run-fields">${fields.map(([k, v]) => `<span><strong>${escapeHtml(k)}</strong> ${escapeHtml(v)}</span>`).join("") || `<span class="dim">No run metadata yet</span>`}</div>
      ${e.notes ? `<p>${escapeHtml(e.notes)}</p>` : ""}
    </div>`;
  }).join("");
  el.innerHTML = `<div class="curve-meta-head">
      <strong>${escapeHtml(metricLabel(metric))} provenance</strong>
      ${provenance ? `<span>${escapeHtml(provenance)}</span>` : ""}
    </div>${rows}`;
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
  const stamp = ["rl", "ascend", "modeling", "ideas", "live"]
    .map((k) => store[k + "_meta"] && store[k + "_meta"].updated)
    .filter(Boolean)
    .sort()
    .reverse()[0];
  document.getElementById("last-updated").textContent =
    `Curated content snapshot${stamp ? ` · latest ${String(stamp).slice(0, 10)}` : ""} · refresh Live Papers for the newest arXiv entries.`;
}
function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
function escapeAttr(s) { return escapeHtml(s); }

// re-render all card panels (used by analyst.js after saving an analysis)
window.refreshDashboard = () => ["rl", "ascend", "modeling", "ideas", "live"].forEach(renderPanel);

init();
