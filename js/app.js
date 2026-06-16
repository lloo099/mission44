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
  setLastUpdated();
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
  container.innerHTML = items.map(cardHTML).join("");
}

function cardHTML(e) {
  const meta = [e.org, e.year].filter(Boolean).join(" · ");
  const tags = (e.tags || []).map((t) => `<span class="t">${escapeHtml(t)}</span>`).join("");
  return `<article class="card">
    <div class="card-top">${e.category ? `<span class="cat">${escapeHtml(e.category)}</span>` : "<span></span>"}${ascendBadge(e)}</div>
    <h3>${escapeHtml(e.title || "Untitled")}</h3>
    ${meta ? `<div class="meta">${escapeHtml(meta)}</div>` : ""}
    ${e.innovation ? `<div class="innov">▸ ${escapeHtml(e.innovation)}</div>` : ""}
    ${e.summary ? `<p class="summary">${escapeHtml(e.summary)}</p>` : ""}
    <div class="tags">${tags}</div>
    ${e.url ? `<a class="link" href="${escapeAttr(e.url)}" target="_blank" rel="noopener">Open source ↗</a>` : ""}
  </article>`;
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
  const status = document.getElementById("live-status");
  if (store.live && store.live.length) {
    status.textContent = `Showing ${store.live.length} cached entries from data/feed.json`;
  } else {
    status.textContent = "No cached feed. Run scripts/fetch_arxiv.py to populate data/feed.json.";
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

init();
