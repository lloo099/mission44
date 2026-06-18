/* Interactive Ascend Da Vinci AI-Core architecture animation — vanilla JS + SVG.
   Two views: "AI Core" (animated matmul datapath) and "Chip / Supernode".
   Exposes window.wireArch(); called from app.js init(). */
(function () {
  const NS = "http://www.w3.org/2000/svg";

  /* ---------- AI Core diagram (the animated one) ---------- */
  const node = (id, x, y, w, h, lines, cls) =>
    `<g class="arch-node ${cls || ""}" id="n-${id}">
       <rect x="${x}" y="${y}" width="${w}" height="${h}" rx="9"/>
       ${lines.map((t, i) => `<text x="${x + w / 2}" y="${y + h / 2 + (i - (lines.length - 1) / 2) * 15 + 4}" text-anchor="middle">${t}</text>`).join("")}
     </g>`;

  const edge = (id, d, cls) => `<path class="arch-edge ${cls || ""}" id="${id}" d="${d}" marker-end="url(#arch-arrow)"/>`;
  const elabel = (x, y, t) => `<text class="arch-elabel" x="${x}" y="${y}" text-anchor="middle">${t}</text>`;

  const CORE_SVG = `
  <svg id="arch-svg" viewBox="0 0 1000 600" preserveAspectRatio="xMidYMid meet" role="img" aria-label="Ascend Da Vinci AI Core datapath">
    <defs>
      <marker id="arch-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
        <path d="M0,0 L10,5 L0,10 z" fill="currentColor"/>
      </marker>
    </defs>

    <!-- core boundary -->
    <rect class="arch-core" x="150" y="42" width="830" height="528" rx="16"/>
    <text class="arch-core-label" x="165" y="64">AI Core · Da Vinci</text>
    <text class="arch-offchip" x="69" y="185" text-anchor="middle">片外 / off-chip</text>

    <!-- edges (drawn first, under nodes) -->
    ${edge("e-hbm-l1", "M114,300 L188,305")}
    ${edge("e-l1-l0a", "M312,288 L393,236")}
    ${edge("e-l1-l0b", "M312,330 L393,366")}
    ${edge("e-l0a-cube", "M492,236 L558,278")}
    ${edge("e-l0b-cube", "M492,366 L558,344")}
    ${edge("e-cube-l0c", "M712,310 L773,305")}
    ${edge("e-l0c-ub", "M828,362 C 828,420 760,432 718,436")}
    ${edge("e-ub-vec", "M727,476 L773,476")}
    ${edge("e-ub-hbm", "M548,512 C 330,566 175,540 114,402")}
    ${edge("e-ctrl-mte2", "M455,120 C 300,165 180,205 150,292", "arch-ctrl")}
    ${edge("e-ctrl-cube", "M560,120 C 625,165 636,205 636,233", "arch-ctrl")}
    ${edge("e-ctrl-vec", "M600,120 C 825,150 832,300 832,433", "arch-ctrl")}

    <!-- edge labels -->
    ${elabel(150, 285, "MTE2")}
    ${elabel(360, 300, "MTE1")}
    ${elabel(300, 470, "MTE3 写回")}

    <!-- nodes -->
    ${node("hbm", 24, 200, 90, 200, ["HBM", "全局内存"], "mem off")}
    ${node("scalar", 405, 72, 210, 48, ["Scalar 单元", "控制 / 微 CPU"], "ctrl")}
    ${node("l1", 188, 250, 124, 110, ["L1 Buffer", "片上缓存"], "mem")}
    ${node("l0a", 393, 206, 99, 60, ["L0A", "左矩阵"], "mem")}
    ${node("l0b", 393, 336, 99, 60, ["L0B", "右矩阵"], "mem")}
    ${node("cube", 558, 235, 154, 150, ["Cube 矩阵单元", "16×16×16 MAC", "4096 FP16 / 拍"], "compute")}
    ${node("l0c", 773, 250, 112, 110, ["L0C", "累加器"], "mem")}
    ${node("ub", 545, 436, 182, 80, ["Unified Buffer", "(UB)"], "mem")}
    ${node("vector", 773, 436, 112, 80, ["Vector 单元", "逐元素"], "compute")}

    <g id="arch-packets"></g>
  </svg>`;

  /* ---------- Chip / supernode diagram (ambient) ---------- */
  function chipSVG() {
    let cores = "";
    for (let r = 0; r < 4; r++) for (let c = 0; c < 6; c++) {
      const x = 120 + c * 70, y = 110 + r * 70;
      cores += `<rect class="chip-core" x="${x}" y="${y}" width="52" height="52" rx="6" style="animation-delay:${(r * 6 + c) * 0.12}s"/>`;
    }
    return `
    <svg id="arch-svg" viewBox="0 0 1000 560" preserveAspectRatio="xMidYMid meet" role="img" aria-label="Ascend chip and supernode">
      <rect class="arch-core" x="80" y="70" width="470" height="330" rx="16"/>
      <text class="arch-core-label" x="96" y="94">Ascend 910B/C · 多个 Da Vinci AI Core</text>
      ${cores}
      <text class="arch-offchip" x="315" y="430" text-anchor="middle">数十个 AI Core 并行 · 片上网络(NoC)互联</text>

      <rect class="chip-hbm" x="600" y="90" width="120" height="120" rx="10"/>
      <text class="chip-lbl" x="660" y="155" text-anchor="middle">HBM 栈</text>
      <path class="arch-edge active" d="M550,150 L598,150" marker-end="url(#arch-arrow2)"/>
      <text class="arch-elabel" x="575" y="140" text-anchor="middle">HBM I/O</text>

      <rect class="chip-node" x="600" y="260" width="300" height="120" rx="12"/>
      <text class="chip-lbl" x="750" y="300" text-anchor="middle">HCCS / 光互联</text>
      <text class="chip-lbl small" x="750" y="324" text-anchor="middle">芯片↔芯片 集合通信 (≈NCCL)</text>
      <text class="chip-lbl small" x="750" y="348" text-anchor="middle">CloudMatrix 384:384×910C 全互联超节点</text>

      <circle class="hccs-ring" cx="750" cy="320" r="0"/>
      <defs><marker id="arch-arrow2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="currentColor"/></marker></defs>
    </svg>
    <p class="arch-chip-note">单芯片把多个 Da Vinci AI Core 用片上网络连起来共享 HBM;多芯片再用 <strong>HCCS / 光互联</strong> 做集合通信(全归约等)。Huawei 的打法是用 <strong>CloudMatrix 384</strong> 这种全互联超节点,以"规模"补"单芯片效率",对标 NVIDIA NVL72。</p>`;
  }

  /* ---------- stages (AI Core view) ---------- */
  const STAGES = [
    { t: "1 · 调度 (Scalar)", active: ["scalar"], edges: ["e-ctrl-mte2", "e-ctrl-cube", "e-ctrl-vec"],
      zh: "Scalar 单元像一个微型 CPU:解析指令,向 MTE / Cube / Vector 各自的队列派发任务。各单元异步并行执行,靠事件同步——这是 NPU 高吞吐的基础。" },
    { t: "2 · 搬入 (MTE2)", active: ["hbm", "l1"], edges: ["e-hbm-l1"],
      zh: "MTE2 把权重和激活从片外 HBM 搬到片上 L1 Buffer。片上带宽远高于 HBM,先缓存、再复用,是省带宽的关键。910B 单卡 64GB HBM 是 RL 的硬约束。" },
    { t: "3 · 喂入矩阵 (MTE1)", active: ["l1", "l0a", "l0b"], edges: ["e-l1-l0a", "e-l1-l0b"],
      zh: "MTE1 从 L1 把左矩阵装入 L0A、右矩阵装入 L0B(含 Img2Col 等重排),作为 Cube 的两路输入。" },
    { t: "4 · 矩阵乘 (Cube)", active: ["l0a", "l0b", "cube", "l0c"], edges: ["e-l0a-cube", "e-l0b-cube", "e-cube-l0c"],
      zh: "Cube 单元每拍完成一个 16×16×16 的 FP16 矩阵乘累加(4096 MAC/拍;INT8 翻倍到 8192),结果累加进 L0C。LLM 里绝大多数算力都花在这里。" },
    { t: "5 · 转出到 UB", active: ["l0c", "ub"], edges: ["e-l0c-ub"],
      zh: "矩阵结果从累加器 L0C 转入 Unified Buffer(UB),交给向量流水线做后处理。" },
    { t: "6 · 向量后处理 (Vector)", active: ["ub", "vector"], edges: ["e-ub-vec"],
      zh: "Vector 单元在 UB 上做逐元素运算:bias、激活(GELU/SwiGLU)、LayerNorm/RMSNorm、量化等。RL 里那条 logits→log-prob→loss 链也主要落在这。" },
    { t: "7 · 写回 (MTE3)", active: ["ub", "hbm"], edges: ["e-ub-hbm"],
      zh: "MTE3 把 UB 里的最终结果写回 HBM。一个算子完成,流水线继续下一块——成千上万次这样的循环拼成一次前向/反向。" },
  ];
  const STAGE_MS = 2800;

  let svg, view = "core", idx = 0, playing = false, speed = 1;
  let raf = null, lastTs = 0, stageElapsed = 0, packets = [];

  function el(id) { return svg && svg.querySelector(id); }

  function setStage(i) {
    idx = (i + STAGES.length) % STAGES.length;
    stageElapsed = 0;
    const s = STAGES[idx];
    svg.querySelectorAll(".arch-node").forEach((n) => n.classList.remove("active"));
    svg.querySelectorAll(".arch-edge").forEach((e) => e.classList.remove("active"));
    s.active.forEach((id) => { const n = el("#n-" + id); if (n) n.classList.add("active"); });
    // build packets for active edges
    packets = [];
    const pl = el("#arch-packets"); pl.innerHTML = "";
    s.edges.forEach((eid) => {
      const path = el("#" + eid); if (!path) return;
      path.classList.add("active");
      const len = path.getTotalLength();
      const isCtrl = path.classList.contains("arch-ctrl");
      const n = isCtrl ? 2 : 3;
      for (let k = 0; k < n; k++) {
        const c = document.createElementNS(NS, "circle");
        c.setAttribute("r", isCtrl ? 3 : 5);
        c.setAttribute("class", "arch-packet" + (isCtrl ? " ctrl" : ""));
        pl.appendChild(c);
        packets.push({ c, path, len, off: (k / n) * len });
      }
    });
    const nb = document.getElementById("arch-narr");
    if (nb) nb.innerHTML = `<span class="arch-step">${s.t}</span> ${s.zh}`;
    const dots = document.getElementById("arch-dots");
    if (dots) [...dots.children].forEach((d, j) => d.classList.toggle("on", j === idx));
  }

  function frame(ts) {
    if (!lastTs) lastTs = ts;
    const dt = ts - lastTs; lastTs = ts;
    const v = 150 * speed; // px/s
    packets.forEach((p) => {
      if (!p.len) { p.len = p.path.getTotalLength(); } // recompute once visible
      if (!p.len) return;
      p.off = (p.off + (dt / 1000) * v) % p.len;
      const pt = p.path.getPointAtLength(p.off);
      p.c.setAttribute("cx", pt.x); p.c.setAttribute("cy", pt.y);
    });
    if (playing) {
      stageElapsed += dt;
      if (stageElapsed >= STAGE_MS / speed) setStage(idx + 1);
    }
    raf = requestAnimationFrame(frame);
  }

  function startLoop() { if (!raf) { lastTs = 0; raf = requestAnimationFrame(frame); } }
  function stopLoop() { if (raf) { cancelAnimationFrame(raf); raf = null; } }

  function setPlaying(p) {
    playing = p;
    const b = document.getElementById("arch-play");
    if (b) b.textContent = p ? "⏸ 暂停" : "▶ 播放";
  }

  function renderCore(container) {
    container.querySelector("#arch-stage").innerHTML = CORE_SVG;
    svg = container.querySelector("#arch-svg");
    setStage(0);
    startLoop();
  }

  function renderChip(container) {
    stopLoop(); packets = [];
    container.querySelector("#arch-stage").innerHTML = chipSVG();
    svg = container.querySelector("#arch-svg");
    const nb = document.getElementById("arch-narr");
    if (nb) nb.innerHTML = `<span class="arch-step">芯片 / 超节点</span> 看完单个 AI Core,再放大一层:芯片如何拼成集群。`;
  }

  function setView(v, container) {
    view = v;
    container.querySelectorAll(".arch-viewbtn").forEach((b) => b.classList.toggle("active", b.dataset.view === v));
    container.querySelector("#arch-controls").style.visibility = (v === "core") ? "visible" : "hidden";
    if (v === "core") renderCore(container); else renderChip(container);
  }

  function wireArch() {
    const container = document.getElementById("arch-body");
    if (!container || container.dataset.wired) return;
    container.dataset.wired = "1";
    container.innerHTML = `
      <div class="arch-bar">
        <div class="arch-views">
          <button class="arch-viewbtn active" data-view="core">AI Core 内部</button>
          <button class="arch-viewbtn" data-view="chip">芯片 / 超节点</button>
        </div>
        <div class="arch-controls" id="arch-controls">
          <button class="ghost-btn" id="arch-prev">◀ 上一步</button>
          <button class="ghost-btn" id="arch-play">▶ 播放</button>
          <button class="ghost-btn" id="arch-next">下一步 ▶</button>
          <label class="dim">速度 <input type="range" id="arch-speed" min="0.5" max="2" step="0.25" value="1"></label>
          <span class="arch-dots" id="arch-dots">${STAGES.map(() => `<i></i>`).join("")}</span>
        </div>
      </div>
      <div id="arch-stage" class="arch-stage"></div>
      <div id="arch-narr" class="arch-narr"></div>
      <div class="arch-legend">
        <span><i class="lg mem"></i>存储 (HBM/L1/L0/UB)</span>
        <span><i class="lg compute"></i>计算 (Cube/Vector)</span>
        <span><i class="lg ctrl"></i>控制 (Scalar)</span>
        <span><i class="lg dash"></i>控制流</span>
        <span><i class="lg pkt"></i>数据流动</span>
      </div>`;

    container.querySelectorAll(".arch-viewbtn").forEach((b) =>
      b.addEventListener("click", () => setView(b.dataset.view, container)));
    container.querySelector("#arch-play").addEventListener("click", () => { setPlaying(!playing); });
    container.querySelector("#arch-next").addEventListener("click", () => { setPlaying(false); setStage(idx + 1); });
    container.querySelector("#arch-prev").addEventListener("click", () => { setPlaying(false); setStage(idx - 1); });
    container.querySelector("#arch-speed").addEventListener("input", (e) => { speed = +e.target.value; });

    setView("core", container);
    setPlaying(true);

    // re-render when the tab is opened (recomputes path lengths now that it's visible)
    const tabBtn = document.querySelector('.tab[data-tab="arch"]');
    if (tabBtn) tabBtn.addEventListener("click", () => {
      requestAnimationFrame(() => {
        if (view === "core") { setStage(idx); setPlaying(true); }
      });
    });
  }

  window.wireArch = wireArch;
})();
