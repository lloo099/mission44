/* Sci-fi animated background: Matrix-style digital rain.
   Theme-aware (glyphs use --accent; trail fades to the page bg), and it
   pauses for reduced-motion / hidden tabs. Sits behind all content (#bg-canvas). */
(function () {
  const canvas = document.getElementById("bg-canvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const reduce = !!(window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches);
  const GLYPHS = "アイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロ0123456789=+*<>/\\{}[]ZX01";

  let w = 0, h = 0, dpr = 1, cols = 0, font = 16, drops = [], speeds = [], raf = null;
  let glyph = "139,147,163", fade = "255,255,255";

  function hexToRgb(hex) {
    const m = hex.trim().replace("#", "");
    if (m.length === 6) { const n = parseInt(m, 16); return `${(n >> 16) & 255},${(n >> 8) & 255},${n & 255}`; }
    if (m.length === 3) return `${parseInt(m[0] + m[0], 16)},${parseInt(m[1] + m[1], 16)},${parseInt(m[2] + m[2], 16)}`;
    return null;
  }
  function readTheme() {
    const cs = getComputedStyle(document.documentElement);
    glyph = hexToRgb(cs.getPropertyValue("--rain")) || glyph;
    const dark = document.documentElement.getAttribute("data-theme") === "dark";
    fade = dark ? "11,14,20" : "255,255,255";
    if (w) { ctx.fillStyle = `rgb(${fade})`; ctx.fillRect(0, 0, w, h); } // reset backdrop on theme switch
  }

  function resize() {
    dpr = Math.min(window.devicePixelRatio || 1, 2);
    w = window.innerWidth; h = window.innerHeight;
    canvas.width = Math.floor(w * dpr); canvas.height = Math.floor(h * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    font = Math.max(13, Math.round(w / 95));
    cols = Math.ceil(w / font);
    drops = new Array(cols).fill(0).map(() => Math.random() * -60);
    speeds = new Array(cols).fill(0).map(() => 0.5 + Math.random() * 0.9);
    ctx.fillStyle = `rgb(${fade})`; ctx.fillRect(0, 0, w, h);
  }

  function frame() {
    // translucent backdrop → leaves fading trails behind each glyph
    ctx.fillStyle = `rgba(${fade},0.085)`;
    ctx.fillRect(0, 0, w, h);
    ctx.font = `${font}px ui-monospace, "SF Mono", Menlo, Consolas, monospace`;
    ctx.textBaseline = "top";
    for (let i = 0; i < cols; i++) {
      const x = i * font, y = drops[i] * font;
      const ch = GLYPHS[(Math.random() * GLYPHS.length) | 0];
      if (y > 0) {
        // bright leading glyph + accent trail
        ctx.fillStyle = `rgba(${glyph},0.92)`;
        ctx.fillText(ch, x, y);
      }
      drops[i] += speeds[i];
      if (y > h && Math.random() > 0.972) { drops[i] = Math.random() * -20; speeds[i] = 0.5 + Math.random() * 0.9; }
    }
    raf = requestAnimationFrame(frame);
  }

  function drawStatic() {
    // single calm frame for reduced-motion
    ctx.fillStyle = `rgb(${fade})`; ctx.fillRect(0, 0, w, h);
    ctx.font = `${font}px ui-monospace, monospace`; ctx.textBaseline = "top";
    for (let i = 0; i < cols; i++) {
      const reps = 2 + ((Math.random() * 4) | 0);
      for (let k = 0; k < reps; k++) {
        ctx.fillStyle = `rgba(${glyph},${(0.15 + Math.random() * 0.5).toFixed(2)})`;
        ctx.fillText(GLYPHS[(Math.random() * GLYPHS.length) | 0], i * font, Math.random() * h);
      }
    }
  }

  function start() { if (!raf && !reduce) raf = requestAnimationFrame(frame); }
  function stop() { if (raf) { cancelAnimationFrame(raf); raf = null; } }

  readTheme();
  resize();
  if (reduce) { drawStatic(); } else { start(); }

  window.addEventListener("resize", () => { resize(); if (reduce) drawStatic(); }, { passive: true });
  document.addEventListener("visibilitychange", () => { if (document.hidden) stop(); else start(); });
  new MutationObserver(() => { readTheme(); if (reduce) drawStatic(); })
    .observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
})();
