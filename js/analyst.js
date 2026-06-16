/* Per-card "Analyze with Claude" + follow-up chat. BYO key (localStorage only).
   Static-site safe: calls the Anthropic API directly from the browser with the
   official dangerous-direct-browser-access header. The key never leaves the
   browser and is never committed.

   Extras: copy / export the analysis, save it back onto the card (localStorage),
   and optionally fetch the paper's text through a user-supplied proxy. */
(function () {
  const KEY_LS = "anthropic_api_key";
  const PROXY_LS = "analyst_proxy_url";
  const SAVED_LS = "analyst_saved";
  const MODEL = "claude-opus-4-8";
  const API = "https://api.anthropic.com/v1/messages";

  const getKey = () => localStorage.getItem(KEY_LS) || "";
  const setKey = (k) => localStorage.setItem(KEY_LS, k.trim());
  const clearKey = () => localStorage.removeItem(KEY_LS);
  const getProxy = () => localStorage.getItem(PROXY_LS) || "";
  const esc = (s) => String(s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

  const entryKey = (e) => e.url || e.title || "";
  function savedMap() { try { return JSON.parse(localStorage.getItem(SAVED_LS) || "{}"); } catch (_) { return {}; } }
  function getSaved(e) { return savedMap()[entryKey(e)] || null; }
  function putSaved(e, messages) {
    const m = savedMap();
    m[entryKey(e)] = { title: e.title, messages, ts: new Date().toISOString() };
    localStorage.setItem(SAVED_LS, JSON.stringify(m));
  }
  function delSaved(e) { const m = savedMap(); delete m[entryKey(e)]; localStorage.setItem(SAVED_LS, JSON.stringify(m)); }
  window.hasSavedAnalysis = (e) => !!getSaved(e);

  let state = null; // { entry, messages: [{role, content}] }

  /* ---------- modal shell ---------- */
  function ensureModal() {
    let m = document.getElementById("analyst-modal");
    if (m) return m;
    m = document.createElement("div");
    m.id = "analyst-modal";
    m.className = "analyst-modal hidden";
    m.innerHTML = `
      <div class="analyst-backdrop" data-close="1"></div>
      <div class="analyst-panel" role="dialog" aria-modal="true">
        <div class="analyst-head">
          <div>
            <h3 id="analyst-title">Analyze</h3>
            <div class="analyst-meta" id="analyst-meta"></div>
          </div>
          <div class="analyst-head-actions">
            <button class="ghost-btn" id="analyst-copy" title="Copy transcript">Copy</button>
            <button class="ghost-btn" id="analyst-export" title="Download as Markdown">Export</button>
            <button class="ghost-btn" id="analyst-save" title="Save this analysis onto the card">Save ★</button>
            <button class="ghost-btn" id="analyst-cfg" title="API key & paper-text proxy">⚙</button>
            <button class="ghost-btn" id="analyst-close" data-close="1">✕</button>
          </div>
        </div>
        <div class="analyst-body" id="analyst-body"></div>
        <form class="analyst-input" id="analyst-form">
          <textarea id="analyst-q" rows="1" placeholder="Ask a follow-up… (Enter to send, Shift+Enter for newline)"></textarea>
          <button class="ghost-btn" id="analyst-send" type="submit">Send</button>
        </form>
        <div class="analyst-note">Powered by Claude (${MODEL}) · key & saved analyses live only in this browser.</div>
      </div>`;
    document.body.appendChild(m);

    m.addEventListener("click", (e) => { if (e.target.dataset.close) close(); });
    document.addEventListener("keydown", (e) => { if (e.key === "Escape") close(); });
    m.querySelector("#analyst-cfg").addEventListener("click", configPrompt);
    m.querySelector("#analyst-copy").addEventListener("click", copyTranscript);
    m.querySelector("#analyst-export").addEventListener("click", exportTranscript);
    m.querySelector("#analyst-save").addEventListener("click", saveToCard);
    m.querySelector("#analyst-form").addEventListener("submit", onSubmit);
    const ta = m.querySelector("#analyst-q");
    ta.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); onSubmit(e); }
    });
    return m;
  }

  function close() {
    const m = document.getElementById("analyst-modal");
    if (m) m.classList.add("hidden");
    state = null;
  }

  function configPrompt() {
    const v = window.prompt(
      "Anthropic API key (stored only in this browser; blank = clear):", getKey());
    if (v !== null) { if (v.trim() === "") clearKey(); else setKey(v); }
    const p = window.prompt(
      "Optional: paper-text proxy URL (lets the model read the linked paper).\n" +
      "Leave blank to disable. See proxy/README.md to deploy one.", getProxy());
    if (p !== null) { if (p.trim() === "") localStorage.removeItem(PROXY_LS); else localStorage.setItem(PROXY_LS, p.trim()); }
  }

  /* ---------- transcript rendering ---------- */
  function bodyEl() { return document.getElementById("analyst-body"); }

  function addBubble(role, label) {
    const b = document.createElement("div");
    b.className = `bubble ${role}`;
    b.innerHTML = `<div class="who">${esc(label || (role === "user" ? "You" : "Claude"))}</div><div class="txt"></div>`;
    bodyEl().appendChild(b);
    bodyEl().scrollTop = bodyEl().scrollHeight;
    return b.querySelector(".txt");
  }

  /* ---------- API call (streaming SSE) ---------- */
  async function streamClaude(messages, system, onDelta) {
    const resp = await fetch(API, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        "x-api-key": getKey(),
        "anthropic-version": "2023-06-01",
        "anthropic-dangerous-direct-browser-access": "true",
      },
      body: JSON.stringify({ model: MODEL, max_tokens: 2048, stream: true, system, messages }),
    });
    if (!resp.ok) {
      let msg = `HTTP ${resp.status}`;
      try { const j = await resp.json(); msg = j.error?.message || msg; } catch (_) {}
      throw new Error(msg);
    }
    const reader = resp.body.getReader();
    const dec = new TextDecoder();
    let buf = "";
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      const lines = buf.split("\n");
      buf = lines.pop();
      for (const line of lines) {
        const s = line.trim();
        if (!s.startsWith("data:")) continue;
        const payload = s.slice(5).trim();
        if (!payload || payload === "[DONE]") continue;
        try {
          const ev = JSON.parse(payload);
          if (ev.type === "content_block_delta" && ev.delta?.type === "text_delta") onDelta(ev.delta.text);
          else if (ev.type === "error") throw new Error(ev.error?.message || "stream error");
        } catch (_) { /* keep-alive / partial */ }
      }
    }
  }

  function systemPrompt(e, paperText) {
    return [
      "You are a concise research analyst helping a user understand a paper/framework/model in a dashboard about RL for LLMs on Ascend/NPU hardware.",
      "Be specific and technical but readable. Use short paragraphs and bullet points. Do not invent results that aren't plausible; flag uncertainty.",
      "",
      "The item under discussion:",
      `- Title: ${e.title || "?"}`,
      e.org ? `- Org: ${e.org}` : "",
      e.year ? `- Year: ${e.year}` : "",
      e.category ? `- Category: ${e.category}` : "",
      e.innovation ? `- Stated innovation: ${e.innovation}` : "",
      e.summary ? `- Summary on the dashboard: ${e.summary}` : "",
      e.url ? `- Link: ${e.url}` : "",
      (e.tags && e.tags.length) ? `- Tags: ${e.tags.join(", ")}` : "",
      paperText ? "\nFetched paper text (may be truncated):\n" + paperText.slice(0, 12000) : "",
    ].filter(Boolean).join("\n");
  }

  let paperCache = "";

  async function runTurn() {
    if (!getKey()) { needKey(); return; }
    const txt = addBubble("assistant");
    let acc = "";
    const send = document.getElementById("analyst-send");
    send.disabled = true;
    try {
      await streamClaude(state.messages, systemPrompt(state.entry, paperCache), (d) => {
        acc += d; txt.textContent = acc; bodyEl().scrollTop = bodyEl().scrollHeight;
      });
      state.messages.push({ role: "assistant", content: acc });
    } catch (err) {
      txt.innerHTML = `<span class="analyst-err">⚠ ${esc(err.message)}</span>`;
      if (/401|api[_ ]?key|authentication/i.test(err.message)) txt.innerHTML += `<br>Check your API key (⚙).`;
    } finally {
      send.disabled = false;
    }
  }

  function needKey() {
    bodyEl().innerHTML = `
      <div class="analyst-status">
        To analyze with Claude, set your Anthropic API key. It is stored only in
        this browser (localStorage) and sent directly to Anthropic — never to this site.
        <div style="margin-top:10px"><button class="ghost-btn" id="analyst-setkey">Enter API key</button></div>
        <div class="analyst-note" style="margin-top:8px">Get a key at console.anthropic.com. Calls are billed to your account.</div>
      </div>`;
    document.getElementById("analyst-setkey").addEventListener("click", () => {
      configPrompt();
      if (getKey()) startAnalysis();
    });
  }

  function onSubmit(e) {
    e.preventDefault();
    const ta = document.getElementById("analyst-q");
    const q = ta.value.trim();
    if (!q || !state) return;
    if (!getKey()) { needKey(); return; }
    ta.value = "";
    addBubble("user").textContent = q;
    state.messages.push({ role: "user", content: q });
    runTurn();
  }

  async function startAnalysis() {
    bodyEl().innerHTML = "";
    paperCache = "";
    const e = state.entry;
    const proxy = getProxy();
    if (proxy && e.url) {
      const note = addBubble("assistant", "system");
      note.innerHTML = `<span class="analyst-status">Fetching paper text via proxy…</span>`;
      try {
        const u = proxy + (proxy.includes("?") ? "&" : "?") + "url=" + encodeURIComponent(e.url);
        const r = await fetch(u);
        const j = await r.json();
        paperCache = (j && j.text) ? j.text : "";
        note.innerHTML = paperCache
          ? `<span class="analyst-status">Loaded ~${paperCache.length} chars of paper text.</span>`
          : `<span class="analyst-status">Proxy returned no text; analyzing from dashboard summary.</span>`;
      } catch (_) {
        note.innerHTML = `<span class="analyst-status">Proxy fetch failed; analyzing from dashboard summary.</span>`;
      }
    }
    const first = "Summarize this work's key innovation(s) and what makes it novel, then its reported performance / results and any caveats. End with where it fits for RL-on-NPU.";
    addBubble("user").textContent = "Analyze: innovation & performance";
    state.messages = [{ role: "user", content: first }];
    runTurn();
  }

  function replaySaved(saved) {
    bodyEl().innerHTML = "";
    const banner = addBubble("assistant", "system");
    banner.innerHTML = `<span class="analyst-status">Saved analysis from ${esc((saved.ts || "").slice(0, 10))}. ` +
      `<button class="ghost-btn" id="analyst-rerun">Re-analyze</button> ` +
      `<button class="ghost-btn" id="analyst-unsave">Remove</button></span>`;
    state.messages = (saved.messages || []).slice();
    state.messages.forEach((msg) => {
      const label = msg.role === "user" ? "You" : "Claude";
      const disp = (msg.role === "user" && msg.content.length > 200) ? "Analyze: innovation & performance" : msg.content;
      addBubble(msg.role, label).textContent = disp;
    });
    document.getElementById("analyst-rerun").addEventListener("click", startAnalysis);
    document.getElementById("analyst-unsave").addEventListener("click", () => {
      delSaved(state.entry);
      if (window.refreshDashboard) window.refreshDashboard();
      startAnalysis();
    });
  }

  /* ---------- copy / export / save ---------- */
  function transcriptMarkdown() {
    if (!state) return "";
    const e = state.entry;
    const head = `# Analysis: ${e.title || ""}\n${[e.org, e.year, e.category].filter(Boolean).join(" · ")}\n${e.url || ""}\n`;
    const body = state.messages.map((m) => {
      const who = m.role === "user" ? "## You" : "## Claude";
      const txt = (m.role === "user" && m.content.length > 200) ? "Analyze: innovation & performance" : m.content;
      return `${who}\n\n${txt}`;
    }).join("\n\n");
    return head + "\n" + body + "\n";
  }

  function flash(id, label) {
    const b = document.getElementById(id); if (!b) return;
    const old = b.textContent; b.textContent = label;
    setTimeout(() => { b.textContent = old; }, 1200);
  }

  async function copyTranscript() {
    const md = transcriptMarkdown();
    if (!md) return;
    try { await navigator.clipboard.writeText(md); flash("analyst-copy", "Copied ✓"); }
    catch (_) { flash("analyst-copy", "Copy failed"); }
  }

  function exportTranscript() {
    const md = transcriptMarkdown();
    if (!md || !state) return;
    const blob = new Blob([md], { type: "text/markdown" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = (state.entry.title || "analysis").replace(/[^\w.-]+/g, "_").slice(0, 60) + ".md";
    a.click();
    URL.revokeObjectURL(a.href);
  }

  function saveToCard() {
    if (!state || !state.messages.some((m) => m.role === "assistant")) { flash("analyst-save", "Nothing yet"); return; }
    putSaved(state.entry, state.messages);
    if (window.refreshDashboard) window.refreshDashboard();
    flash("analyst-save", "Saved ★");
  }

  /* ---------- entry point ---------- */
  function open(entry) {
    const m = ensureModal();
    state = { entry, messages: [] };
    paperCache = "";
    m.querySelector("#analyst-title").textContent = entry.title || "Analyze";
    m.querySelector("#analyst-meta").textContent = [entry.org, entry.year, entry.category].filter(Boolean).join(" · ");
    m.classList.remove("hidden");
    const saved = getSaved(entry);
    if (saved) replaySaved(saved);
    else if (getKey()) startAnalysis();
    else needKey();
  }

  // delegated click on any .analyze-btn (cards re-render, so bind on document)
  document.addEventListener("click", (e) => {
    const btn = e.target.closest && e.target.closest(".analyze-btn");
    if (!btn) return;
    try { open(JSON.parse(btn.dataset.entry)); } catch (_) {}
  });

  window.openAnalyst = open;
})();
