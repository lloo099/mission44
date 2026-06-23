/* Sci-fi animated background: drifting "neural network" of nodes + links.
   Theme-aware (reads --accent), cursor-reactive, reduced-motion friendly,
   and pauses when the tab is hidden. Sits behind all content (#bg-canvas). */
(function () {
  const canvas = document.getElementById("bg-canvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const reduce = !!(window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches);

  let w = 0, h = 0, dpr = 1, particles = [], raf = null;
  let rgb = "37,99,235";
  const mouse = { x: -9999, y: -9999, on: false };
  const LINK_D2 = 20000; // px^2 link distance threshold

  function hexToRgb(hex) {
    const m = hex.trim().replace("#", "");
    if (m.length === 6) { const n = parseInt(m, 16); return `${(n >> 16) & 255},${(n >> 8) & 255},${n & 255}`; }
    if (m.length === 3) { return `${parseInt(m[0] + m[0], 16)},${parseInt(m[1] + m[1], 16)},${parseInt(m[2] + m[2], 16)}`; }
    return null;
  }
  function readColor() {
    const c = getComputedStyle(document.documentElement).getPropertyValue("--accent");
    rgb = hexToRgb(c) || rgb;
  }

  function mk() {
    return { x: Math.random() * w, y: Math.random() * h, vx: (Math.random() - 0.5) * 0.35, vy: (Math.random() - 0.5) * 0.35 };
  }
  function resize() {
    dpr = Math.min(window.devicePixelRatio || 1, 2);
    w = window.innerWidth; h = window.innerHeight;
    canvas.width = Math.floor(w * dpr); canvas.height = Math.floor(h * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const target = Math.round(Math.min(110, Math.max(28, (w * h) / 15000)));
    if (particles.length < target) { for (let i = particles.length; i < target; i++) particles.push(mk()); }
    else particles.length = target;
  }

  function frame() {
    ctx.clearRect(0, 0, w, h);
    if (!reduce) {
      for (const p of particles) {
        p.x += p.vx; p.y += p.vy;
        if (p.x < 0 || p.x > w) p.vx *= -1;
        if (p.y < 0 || p.y > h) p.vy *= -1;
      }
    }
    // links between nearby nodes
    for (let i = 0; i < particles.length; i++) {
      const a = particles[i];
      for (let j = i + 1; j < particles.length; j++) {
        const b = particles[j], dx = a.x - b.x, dy = a.y - b.y, d2 = dx * dx + dy * dy;
        if (d2 < LINK_D2) {
          const al = (1 - d2 / LINK_D2) * 0.26;
          ctx.strokeStyle = `rgba(${rgb},${al.toFixed(3)})`;
          ctx.lineWidth = 1;
          ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
        }
      }
      // links to cursor (brighter)
      if (mouse.on) {
        const dx = a.x - mouse.x, dy = a.y - mouse.y, d2 = dx * dx + dy * dy, R2 = LINK_D2 * 1.6;
        if (d2 < R2) {
          const al = (1 - d2 / R2) * 0.5;
          ctx.strokeStyle = `rgba(${rgb},${al.toFixed(3)})`;
          ctx.lineWidth = 1.1;
          ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(mouse.x, mouse.y); ctx.stroke();
        }
      }
    }
    // nodes
    ctx.fillStyle = `rgba(${rgb},0.55)`;
    for (const p of particles) { ctx.beginPath(); ctx.arc(p.x, p.y, 1.5, 0, 6.2832); ctx.fill(); }
    if (!reduce) raf = requestAnimationFrame(frame);
  }

  function start() { if (!raf && !reduce) raf = requestAnimationFrame(frame); }
  function stop() { if (raf) { cancelAnimationFrame(raf); raf = null; } }

  readColor();
  resize();
  frame();           // draw at least one frame (also the static frame for reduced-motion)
  if (!reduce) start();

  window.addEventListener("resize", resize, { passive: true });
  window.addEventListener("mousemove", (e) => { mouse.x = e.clientX; mouse.y = e.clientY; mouse.on = true; }, { passive: true });
  window.addEventListener("mouseout", () => { mouse.on = false; });
  document.addEventListener("visibilitychange", () => { if (document.hidden) stop(); else start(); });
  // re-read accent when the theme is toggled
  new MutationObserver(readColor).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
})();
