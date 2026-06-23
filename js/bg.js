/* Sci-fi animated background: floating "token stream".
   RL / NPU / LLM terms drift slowly upward with a gentle sway and fade in/out.
   Light-gray via --rain (theme-aware), pauses for reduced-motion / hidden tabs,
   and sits behind all content (#bg-canvas). clearRect each frame → never builds
   up over text. */
(function () {
  const canvas = document.getElementById("bg-canvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const reduce = !!(window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches);

  const WORDS = [
    "rollout", "GRPO", "RLVR", "FP8", "MXFP4", "KV-cache", "sleep-mode", "async-RL",
    "Ascend", "910B", "950DT", "vLLM-Ascend", "MindSpeed", "CANN", "Da Vinci", "Cube",
    "HBM", "HCCS", "CloudMatrix", "MoE", "DSA", "MSA", "sparse-attn", "1M ctx",
    "policy", "reward", "advantage", "log-prob", "KL", "logits", "PPO", "veRL",
    "token", "NPU", "FP4", "Jet-RL", "Seed 2.1", "rollout-bubble", "staleness",
  ];

  let w = 0, h = 0, dpr = 1, tokens = [], raf = null, glyph = "139,147,163";

  function hexToRgb(hex) {
    const m = hex.trim().replace("#", "");
    if (m.length === 6) { const n = parseInt(m, 16); return `${(n >> 16) & 255},${(n >> 8) & 255},${n & 255}`; }
    if (m.length === 3) return `${parseInt(m[0] + m[0], 16)},${parseInt(m[1] + m[1], 16)},${parseInt(m[2] + m[2], 16)}`;
    return null;
  }
  function readColor() {
    glyph = hexToRgb(getComputedStyle(document.documentElement).getPropertyValue("--rain")) || glyph;
  }

  function mk(spread) {
    return {
      txt: WORDS[(Math.random() * WORDS.length) | 0],
      x: Math.random() * w,
      y: spread ? Math.random() * h : h + 16 + Math.random() * 80,
      vy: 0.14 + Math.random() * 0.42,
      sway: Math.random() * Math.PI * 2,
      swaySpd: 0.004 + Math.random() * 0.009,
      amp: 5 + Math.random() * 14,
      size: 11 + Math.random() * 9,
      max: 0.16 + Math.random() * 0.30,
    };
  }
  function resize() {
    dpr = Math.min(window.devicePixelRatio || 1, 2);
    w = window.innerWidth; h = window.innerHeight;
    canvas.width = Math.floor(w * dpr); canvas.height = Math.floor(h * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.textBaseline = "middle";
    const target = Math.round(Math.min(48, Math.max(12, (w * h) / 45000)));
    if (tokens.length < target) { for (let i = tokens.length; i < target; i++) tokens.push(mk(true)); }
    else tokens.length = target;
  }

  function draw(animate) {
    ctx.clearRect(0, 0, w, h);
    for (const t of tokens) {
      if (animate) { t.y -= t.vy; t.sway += t.swaySpd; }
      const x = t.x + Math.sin(t.sway) * t.amp;
      // fade in from the bottom, fade out toward the top
      const fadeBot = Math.min(1, (h - t.y) / (h * 0.12));
      const fadeTop = Math.min(1, t.y / (h * 0.20));
      const a = t.max * Math.max(0, Math.min(fadeBot, fadeTop, 1));
      if (a > 0.002) {
        ctx.font = `${t.size.toFixed(1)}px ui-monospace, "SF Mono", Menlo, Consolas, monospace`;
        ctx.fillStyle = `rgba(${glyph},${a.toFixed(3)})`;
        ctx.fillText(t.txt, x, t.y);
      }
      if (animate && t.y < -20) Object.assign(t, mk(false));
    }
  }

  function frame() { draw(true); raf = requestAnimationFrame(frame); }
  function start() { if (!raf && !reduce) raf = requestAnimationFrame(frame); }
  function stop() { if (raf) { cancelAnimationFrame(raf); raf = null; } }

  readColor();
  resize();
  if (reduce) draw(false); else start();

  window.addEventListener("resize", () => { resize(); if (reduce) draw(false); }, { passive: true });
  document.addEventListener("visibilitychange", () => { if (document.hidden) stop(); else start(); });
  new MutationObserver(() => { readColor(); if (reduce) draw(false); })
    .observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
})();
