/* Per-card "Analyze with Claude" + follow-up chat. BYO key (localStorage only).
   Static-site safe: calls the Anthropic API directly from the browser with the
   official dangerous-direct-browser-access header. The key never leaves the
   browser and is never committed. */
(function () {
  const KEY_LS = "anthropic_api_key";
  const MODEL = "claude-opus-4-8";
  const API = "https://api.anthropic.com/v1/messages";

  const getKey = () => localStorage.getItem(KEY_LS) || "";
  const setKey = (k) => localStorage.setItem(KEY_LS, k.trim());
  const clearKey = () => localStorage.removeItem(KEY_LS);
  const esc = (s) => String(s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

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
            <button class="ghost-btn" id="analyst-key">API key</button>
            <button class="ghost-btn" id="analyst-close" data-close="1">✕</button>
          </div>
        </div>
        <div class="analyst-body" id="analyst-body"></div>
        <form class="analyst-input" id="analyst-form">
          <textarea id="analyst-q" rows="1" placeholder="Ask a follow-up… (Enter to send, Shift+Enter for newline)"></textarea>
          <button class="ghost-btn" id="analyst-send" type="submit">Send</button>
        </form>
        <div class="analyst-note">Powered by Claude (${MODEL}) · your API key is stored only in this browser.</div>
      </div>`;
    document.body.appendChild(m);

    m.addEventListener("click", (e) => { if (e.target.dataset.close) close(); });
    document.addEventListener("keydown", (e) => { if (e.key === "Escape") close(); });
    m.querySelector("#analyst-key").addEventListener("click", keyPrompt);
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

  function keyPrompt() {
    const cur = getKey();
    const v = window.prompt(
      "Paste your Anthropic API key (stored only in this browser's localStorage).\nLeave blank and press OK to clear it.",
      cur);
    if (v === null) return;
    if (v.trim() === "") { clearKey(); return; }
    setKey(v);
  }

  /* ---------- transcript rendering ---------- */
  function bodyEl() { return document.getElementById("analyst-body"); }

  function addBubble(role) {
    const b = document.createElement("div");
    b.className = `bubble ${role}`;
    b.innerHTML = `<div class="who">${role === "user" ? "You" : "Claude"}</div><div class="txt"></div>`;
    bodyEl().appendChild(b);
    bodyEl().scrollTop = bodyEl().scrollHeight;
    return b.querySelector(".txt");
  }

  function setStatus(txt) {
    const el = bodyEl();
    el.innerHTML = `<div class="analyst-status">${esc(txt)}</div>`;
  }

  /* ---------- API call (streaming SSE) ---------- */
  async function streamClaude(messages, system, onDelta) {
    const key = getKey();
    const resp = await fetch(API, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        "x-api-key": key,
        "anthropic-version": "2023-06-01",
        "anthropic-dangerous-direct-browser-access": "true",
      },
      body: JSON.stringify({
        model: MODEL,
        max_tokens: 2048,
        stream: true,
        system,
        messages,
      }),
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
          if (ev.type === "content_block_delta" && ev.delta?.type === "text_delta") {
            onDelta(ev.delta.text);
          } else if (ev.type === "error") {
            throw new Error(ev.error?.message || "stream error");
          }
        } catch (_) { /* ignore keep-alive / partial */ }
      }
    }
  }

  function systemPrompt(e) {
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
    ].filter(Boolean).join("\n");
  }

  async function runTurn() {
    if (!getKey()) { needKey(); return; }
    const txt = addBubble("assistant");
    let acc = "";
    const send = document.getElementById("analyst-send");
    send.disabled = true;
    try {
      await streamClaude(state.messages, systemPrompt(state.entry), (d) => {
        acc += d; txt.textContent = acc; bodyEl().scrollTop = bodyEl().scrollHeight;
      });
      state.messages.push({ role: "assistant", content: acc });
    } catch (err) {
      txt.innerHTML = `<span class="analyst-err">⚠ ${esc(err.message)}</span>`;
      if (/401|api[_ ]?key|authentication/i.test(err.message)) txt.innerHTML += `<br>Check your API key (top-right “API key”).`;
    } finally {
      send.disabled = false;
    }
  }

  function needKey() {
    bodyEl().innerHTML = `
      <div class="analyst-status">
        To analyze with Claude, paste your Anthropic API key. It is stored only in
        this browser (localStorage) and sent directly to Anthropic — never to this site.
        <div style="margin-top:10px"><button class="ghost-btn" id="analyst-setkey">Enter API key</button></div>
        <div class="analyst-note" style="margin-top:8px">Get a key at console.anthropic.com. Calls are billed to your account.</div>
      </div>`;
    document.getElementById("analyst-setkey").addEventListener("click", () => {
      keyPrompt();
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

  function startAnalysis() {
    bodyEl().innerHTML = "";
    const first = "Summarize this work's key innovation(s) and what makes it novel, then its reported performance / results and any caveats. End with where it fits for RL-on-NPU.";
    addBubble("user").textContent = "Analyze: innovation & performance";
    state.messages = [{ role: "user", content: first }];
    runTurn();
  }

  /* ---------- entry point ---------- */
  function open(entry) {
    const m = ensureModal();
    state = { entry, messages: [] };
    m.querySelector("#analyst-title").textContent = entry.title || "Analyze";
    m.querySelector("#analyst-meta").textContent =
      [entry.org, entry.year, entry.category].filter(Boolean).join(" · ");
    m.classList.remove("hidden");
    if (getKey()) startAnalysis(); else needKey();
  }

  // delegated click on any .analyze-btn (cards re-render, so bind on document)
  document.addEventListener("click", (e) => {
    const btn = e.target.closest && e.target.closest(".analyze-btn");
    if (!btn) return;
    try { open(JSON.parse(btn.dataset.entry)); } catch (_) {}
  });

  window.openAnalyst = open;
})();
