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
    activeFilters[k] = new Set();
  });

  buildStats();
  ["rl", "ascend", "modeling", "ideas", "live"].forEach(renderPanel);
  buildFilterbars();
  wireTabs();
  wireSearch();
  wireLive();
  wireCurves();
  wireCompare();
  wireSurvey();
  if (window.wireArch) wireArch();
  setLastUpdated();
}

/* ---------- survey (tiny markdown renderer, no deps) ---------- */
async function wireSurvey() {
  const el = document.getElementById("survey-body");
  if (!el) return;
  try {
    const res = await fetch("docs/2026-h1-architecture-survey.md", { cache: "no-cache" });
    if (!res.ok) throw new Error(res.status);
    el.innerHTML = renderMarkdown(await res.text());
  } catch (e) {
    el.innerHTML = `<div class="empty">Couldn't load the survey markdown (${escapeHtml(String(e.message || e))}).</div>`;
  }
}

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
  container.innerHTML = count + items.map(cardHTML).join("");
}

function cardHTML(e) {
  const meta = [e.org, e.year].filter(Boolean).join(" · ");
  const tags = (e.tags || []).map((t) => `<span class="t">${escapeHtml(t)}</span>`).join("");
  const slim = JSON.stringify({ title: e.title, org: e.org, year: e.year, category: e.category, innovation: e.innovation, summary: e.summary, url: e.url, tags: e.tags, analysis: e.analysis });
  const analyzed = !!e.analysis || (window.hasSavedAnalysis && window.hasSavedAnalysis(e));
  return `<article class="card">
    <div class="card-top">${e.category ? `<span class="cat">${escapeHtml(e.category)}</span>` : "<span></span>"}<span class="card-badges">${confBadge(e)}${ascendBadge(e)}</span></div>
    <h3>${hl(e.title || "Untitled")}</h3>
    ${meta ? `<div class="meta">${escapeHtml(meta)}</div>` : ""}
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
    ${steps ? `<div class="why"><strong>First steps:</strong><ul>${steps}</ul></div>` : ""}
    ${(e.tags || []).length ? `<div class="tags">${(e.tags || []).map((t) => `<span class="t">${escapeHtml(t)}</span>`).join("")}</div>` : ""}
  </article>`;
}

/* ---------- tabs / search ---------- */
function wireTabs() {
  document.querySelectorAll(".tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      document.querySelectorAll(".tab").forEach((t) => t.classList.remove("active"));
      document.querySelectorAll(".panel").forEach((p) => p.classList.remove("active"));
      tab.classList.add("active");
      document.getElementById(tab.dataset.tab).classList.add("active");
    });
  });
  // number keys 1..9 jump to tabs (when not typing in a field)
  document.addEventListener("keydown", (e) => {
    if (!/^[1-9]$/.test(e.key)) return;
    const el = document.activeElement;
    if (el && /INPUT|TEXTAREA|SELECT/.test(el.tagName)) return;
    const tabs = [...document.querySelectorAll(".tab")];
    if (tabs[+e.key - 1]) tabs[+e.key - 1].click();
  });
}

function wireSearch() {
  const box = document.getElementById("search");
  box.addEventListener("input", () => {
    searchTerm = box.value.trim().toLowerCase();
    ["rl", "ascend", "modeling", "ideas", "live"].forEach(renderPanel);
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "/" && document.activeElement !== box) { e.preventDefault(); box.focus(); }
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
  const metrics = [];
  payload.experiments.forEach((e) =>
    Object.keys(e.metrics || {}).forEach((m) => { if (!metrics.includes(m)) metrics.push(m); }));
  sel.innerHTML = metrics.map((m) => `<option value="${escapeAttr(m)}">${escapeHtml(metricLabel(m))}</option>`).join("");
  sel.addEventListener("change", () => renderCurve(sel.value));
  if (status) {
    const upd = payload.updated ? ` · updated ${String(payload.updated).slice(0, 10)}` : "";
    status.textContent = `${payload.experiments.length} run(s)${upd}`;
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
