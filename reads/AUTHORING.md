# 精读版页面(reads/<post-id>.html)编写规范

每篇 Dispatch 对应一个独立的长文精读页,与 `docs/blog/<file>.md` 内容同源,但以"逐点拆解 + 对照 + 原图解析"的结构重写。参考实现:`reads/dispatch-35-frontier-scan.html`(通用格式)与用户提供的 K3 后训练精读页(单报告精读格式)。

## 文件与资源

- 页面:`reads/<post-id>.html`(post-id 即 `data/blog.json` 中的 `id`)
- 样式:仅引用 `read.css`(`<link rel="stylesheet" href="read.css">`),**不得内联 `<style>`**,不得引入外部字体或脚本
- 图片目录:`reads/img/<post-id>/`
  - `fig-<n>.svg`:由 `scripts/render-mermaid-svgs.mjs` 从原文第 n 个 mermaid 图预渲染而来(已生成,直接引用),称为**本站图**
  - `orig-<n>.png|jpg`:从原始来源抓取的**原图**(论文/仓库/官方博客配图)。网络代理只放行 `raw.githubusercontent.com`,因此原图只能来自原文引用的 GitHub 仓库 README(`curl -sS -o reads/img/<id>/orig-1.png https://raw.githubusercontent.com/<owner>/<repo>/main/<path>`);arXiv/HF/厂商站点均不可达。抓不到原图时用本站图,不要留空
- 引用路径均相对页面:`img/<post-id>/fig-1.svg`

## 页面骨架(严格遵守)

```html
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Dispatch NN 精读版 · 短标题</title>
<meta name="description" content="一句话摘要">
<link rel="canonical" href="https://lloo099.github.io/mission44/reads/<post-id>.html">
<meta property="og:title" content="…"><meta property="og:description" content="…"><meta property="og:type" content="article">
<meta property="og:url" content="https://lloo099.github.io/mission44/reads/<post-id>.html">
<meta property="og:image" content="https://lloo099.github.io/mission44/assets/og/<post-id>.png">
<meta name="twitter:card" content="summary_large_image">
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Crect width='32' height='32' rx='7' fill='%2310141c'/%3E%3Ctext x='16' y='22' font-size='18' text-anchor='middle' fill='%236aa6ff'%3E%E2%97%88%3C/text%3E%3C/svg%3E">
<script>try{if(localStorage.getItem('theme')==='dark')document.documentElement.setAttribute('data-theme','dark');}catch(e){}</script>
<link rel="stylesheet" href="read.css">
</head>
<body>
<div class="wrap">

<div class="readbar">
  <a class="brand" href="../index.html">RL on NPU · Research Dashboard</a>
  <a href="../index.html#blog/<post-id>">看板内阅读原文</a>
  <a href="../docs/blog/<file>.md">Markdown 源</a>
  <span class="spacer"></span>
  <button id="theme-toggle" type="button" aria-label="切换深浅色主题">切换深浅色</button>
</div>

<header class="masthead">
  <div class="kicker">NPU Frontier Dispatch NN · 精读版 · YYYY-MM-DD</div>
  <h1>标题(可与原文不同,但同义)</h1>
  <p class="standfirst">两到三句:这篇拆什么、怎么拆、读者能带走什么。</p>
  <div class="byline"><span>来源 A</span><span>覆盖范围</span><span>原图 N 张 · 本站图 M 张</span></div>
</header>
<nav class="toc"><ol>
  <li><a href="#s0"><span class="tnum">00</span>全景与读法</a></li>
  …每个 h2 一项,id 为 s0…sN…
</ol></nav>

<main>
<h2 id="s0"><span class="hnum">00</span>全景与读法</h2>
…
</main>

<footer>来源与声明…</footer>
</div>
<script>
(function(){var b=document.getElementById('theme-toggle');if(!b)return;b.addEventListener('click',function(){var r=document.documentElement;var dark=r.getAttribute('data-theme')==='dark'||(!r.getAttribute('data-theme')&&matchMedia('(prefers-color-scheme: dark)').matches);var next=dark?'light':'dark';r.setAttribute('data-theme',next);try{localStorage.setItem('theme',next);}catch(e){}});})();
</script>
</body>
</html>
```

## 内容结构

1. **00 全景与读法**:`<p class="lede">` 一段总览 + `<div class="note">` 说明"本文的读法"(固定三段结构 + 芯片含义)+ 可选 `<div class="formula">` 用等宽块画出主线/流程。
2. **正文各节**(每节一个 `<h2 id="sN"><span class="hnum">NN</span>标题</h2>`),按原文小节重组,每节固定三段:
   - 一段正文(来自原文,保留全部数字、URL 链接与限定语)
   - `<div class="vs">` 双栏:左 `.base`(灰)与右 `.real`(橙)。**双栏语义按篇选一种并全文统一**:
     - 单报告/单系统精读(如 K3、V4、dsh、slime、verl、OLMo、LongCat、GLM):左 = **朴素基线**(不看论文的直觉实现),右 = **它的做法**
     - 全景/方法学/调研篇:左 = **叙事怎么说**,右 = **证据是什么**
     - 昇腾落地篇:左 = **CUDA 世界的默认做法**,右 = **昇腾上的现状**
   - `<p class="delta"><span class="dl">差在哪</span>…</p>`(或 `本站判断`):说清差异、原文是否给了理由/证据、对 RL-on-NPU 的含义
3. **图**:每张 `<figure>` = `<img>` + `<figcaption><span class="fnum">图 N</span>说明。<span class="chip c-vr|c-tp|c-inf">来源类型</span> 来源(仓库/论文/本站)。</figcaption>` + `<div class="readout"><h5>这张图怎么读</h5><p>…</p></div>`。本站图加 `class="diagram"`。**每页至少 3 张图**,原图优先、本站图补足;读法段要指出坐标轴/因果链/最值得看的位置,而非复述图题。
4. **总表**:`<div class="tbl"><table>…<caption>…</caption></table></div>`,一页汇总各节的双栏对照。
5. **原文没给的东西 / 未能核实的东西**:`<ul>` 列出复现必须自定的数值、未给理由的设计、未经独立复现的自报数字。
6. **可搬走的判断**:`<div class="note">` 三条;**下一步看什么**:`<div class="note flag">`。

## 芯片(来源类型标注)

- `<span class="chip c-tp">第三方</span>` 独立测量、审稿记录、多方报道一致
- `<span class="chip c-vr">自报</span>` 厂商/作者口径
- `<span class="chip c-inf">推断</span>` 本站分析
所有数字首次出现处应能判断来源类型;原文已标"自报/provisional/推断"的必须保留。

## 文风与纪律

- 专业简洁、无口语(禁用:崩法、洼地、占坑、黑箱、动物园、押注、期房、刺眼 等;分别改为 失稳模式、空白领域、已有公开工作、未公开、谱系、投入、未兑现承诺、显著的负向)
- **不得新增原文没有的事实或数字**;可以重组、压缩、给出对照与判断,但判断要标为本站判断
- 保留原文全部外链(以 `<a href>` 形式)
- 中文全文使用中文标点;英文术语与数字前后留一个空格
- 表格、代码、宽图放在带 `overflow-x:auto` 的容器内(`.tbl`、`.formula` 已具备)
- 不使用 emoji 作为节标记;编号只用于确有顺序的内容(节号本身是顺序,允许)

## 完成后自检

- `node scripts/check-reads.mjs <post-id>`:检查骨架、TOC 与 h2 的 id 一致、图片文件存在、无内联 style、无外部资源、无 `{{` 占位
- 在 `data/blog.json` 对应条目加 `"read": "reads/<post-id>.html"`(由汇总者统一添加,单篇作者不要改 blog.json)
