/* Sci-fi animated background: circuit-board traces.
   Right-angle PCB routes on a grid with solder pads, and signal pulses that
   travel along the traces. Light-gray via --rain (theme-aware), pauses for
   reduced-motion / hidden tabs, sits behind all content (#bg-canvas). */
(function () {
  const canvas = document.getElementById("bg-canvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const reduce = !!(window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches);
  const CELL = 46; // matches the page's background grid spacing

  let w = 0, h = 0, dpr = 1, traces = [], pulses = [], raf = null, glyph = "139,147,163";

  function hexToRgb(hex) {
    const m = hex.trim().replace("#", "");
    if (m.length === 6) { const n = parseInt(m, 16); return `${(n >> 16) & 255},${(n >> 8) & 255},${n & 255}`; }
    if (m.length === 3) return `${parseInt(m[0] + m[0], 16)},${parseInt(m[1] + m[1], 16)},${parseInt(m[2] + m[2], 16)}`;
    return null;
  }
  function readColor() { glyph = hexToRgb(getComputedStyle(document.documentElement).getPropertyValue("--rain")) || glyph; }
  const ri = (n) => (Math.random() * n) | 0;
  const clamp = (v, a, b) => (v < a ? a : v > b ? b : v);

  function makeTrace() {
    const cols = Math.max(2, Math.floor(w / CELL)), rows = Math.max(2, Math.floor(h / CELL));
    let cx = ri(cols + 1), cy = ri(rows + 1);
    const pts = [[cx * CELL, cy * CELL]];
    let dir = null;
    const segs = 4 + ri(7);
    for (let i = 0; i < segs; i++) {
      const opts = [[1, 0], [-1, 0], [0, 1], [0, -1]].filter((d) => !dir || !(d[0] === -dir[0] && d[1] === -dir[1]));
      const d = opts[ri(opts.length)];
      const nx = clamp(cx + d[0] * (1 + ri(4)), 0, cols), ny = clamp(cy + d[1] * (1 + ri(4)), 0, rows);
      if (nx === cx && ny === cy) continue;
      cx = nx; cy = ny; dir = d; pts.push([cx * CELL, cy * CELL]);
    }
    const segLen = []; let total = 0;
    for (let i = 1; i < pts.length; i++) { const l = Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]); segLen.push(l); total += l; }
    return { pts, segLen, total };
  }
  function pointAt(tr, d) {
    let i = 0;
    while (i < tr.segLen.length && d > tr.segLen[i]) { d -= tr.segLen[i]; i++; }
    if (i >= tr.segLen.length) return tr.pts[tr.pts.length - 1];
    const a = tr.pts[i], b = tr.pts[i + 1], t = tr.segLen[i] ? d / tr.segLen[i] : 0;
    return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t];
  }
  function mkPulse() {
    const ti = ri(traces.length), tr = traces[ti] || { total: 1 };
    return { ti, d: Math.random() * tr.total, spd: 0.6 + Math.random() * 1.5 };
  }
  function resize() {
    dpr = Math.min(window.devicePixelRatio || 1, 2);
    w = window.innerWidth; h = window.innerHeight;
    canvas.width = Math.floor(w * dpr); canvas.height = Math.floor(h * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const tcount = Math.round(Math.min(72, Math.max(14, (w * h) / 20000)));
    traces = []; for (let i = 0; i < tcount; i++) traces.push(makeTrace());
    pulses = []; for (let i = 0, n = Math.round(tcount * 0.7); i < n; i++) pulses.push(mkPulse());
  }

  function frame(animate) {
    ctx.clearRect(0, 0, w, h);
    // traces
    ctx.lineWidth = 1.2; ctx.lineJoin = "round"; ctx.lineCap = "round";
    ctx.strokeStyle = `rgba(${glyph},0.20)`;
    for (const tr of traces) {
      ctx.beginPath();
      ctx.moveTo(tr.pts[0][0], tr.pts[0][1]);
      for (let i = 1; i < tr.pts.length; i++) ctx.lineTo(tr.pts[i][0], tr.pts[i][1]);
      ctx.stroke();
    }
    // solder pads at vertices
    ctx.fillStyle = `rgba(${glyph},0.30)`;
    for (const tr of traces) for (const p of tr.pts) ctx.fillRect(p[0] - 1.7, p[1] - 1.7, 3.4, 3.4);
    // travelling signal pulses
    ctx.shadowColor = `rgba(${glyph},0.9)`; ctx.shadowBlur = 8;
    ctx.fillStyle = `rgba(${glyph},0.95)`;
    for (const pu of pulses) {
      const tr = traces[pu.ti]; if (!tr) continue;
      const p = pointAt(tr, pu.d);
      ctx.beginPath(); ctx.arc(p[0], p[1], 2.2, 0, 6.2832); ctx.fill();
      if (animate) { pu.d += pu.spd; if (pu.d > tr.total) Object.assign(pu, mkPulse()); }
    }
    ctx.shadowBlur = 0;
  }
  function loop() { frame(true); raf = requestAnimationFrame(loop); }
  function start() { if (!raf && !reduce) raf = requestAnimationFrame(loop); }
  function stop() { if (raf) { cancelAnimationFrame(raf); raf = null; } }

  readColor();
  resize();
  if (reduce) frame(false); else start();

  window.addEventListener("resize", () => { resize(); if (reduce) frame(false); }, { passive: true });
  document.addEventListener("visibilitychange", () => { if (document.hidden) stop(); else start(); });
  new MutationObserver(() => { readColor(); if (reduce) frame(false); })
    .observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
})();
