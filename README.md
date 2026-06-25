# RL-on-NPU Research Dashboard

A lightweight, zero-build **research dashboard** for tracking the intersection of
three fast-moving areas, so you can pick a concrete project to work on:

1. **Reinforcement Learning for LLMs** — RLHF, GRPO, RLVR, DPO and successors, reasoning models, training frameworks.
2. **Ascend / NPU ecosystem** — hardware (910B/910C), CANN, MindSpore/MindSpeed, vLLM-Ascend, and the state of RL-on-Ascend.
3. **LLM modeling advances** — MoE, attention variants, long context, low-precision training that matter on memory-constrained NPUs.

Plus a synthesized **Project Ideas** section with impact / difficulty / novelty ratings,
and a **Live Papers** feed pulled from arXiv.

## Quick start

It's plain HTML/CSS/JS — no build step.

```bash
# from the repo root, serve the folder (any static server works):
python3 -m http.server 8080
# then open http://localhost:8080
```

> Open `index.html` via `http://` (not `file://`) so the browser can `fetch()` the JSON data files.

## Refreshing the Live Papers feed

The browser can't fetch arXiv directly (no CORS headers), so use the included script:

```bash
python3 scripts/fetch_arxiv.py --max 15
```

This writes `data/feed.json` = **curated highlights** (`data/feed_pinned.json`, always kept,
with hand-written summaries / official links / Ascend-readiness badges) **+ fresh arXiv papers**
(deduped, tagged `auto`). No third-party packages required (standard library only).

- To curate the feed, edit **`data/feed_pinned.json`** — never edit `data/feed.json` by hand
  (it is regenerated).
- A GitHub Actions workflow (`.github/workflows/refresh-feed.yml`) runs this **daily** and commits
  `data/feed.json` if it changed; you can also trigger it manually from the Actions tab.

### Ascend-readiness badges

Any card (in `modeling.json`, `feed_pinned.json`, etc.) may carry an `"ascend"` field rendered as
a badge and a filter chip: `"ready"` ✅, `"partial"` ⚠️, or `"none"` ❌. Add an optional
`"ascendNote"` for the hover tooltip.

## Project layout

```
index.html            # the dashboard shell
css/styles.css        # dark theme
js/app.js             # data loading, tabs, search, filters, rendering
data/
  rl.json             # RL-for-LLMs entries (curated)
  ascend.json         # Ascend/NPU entries (curated)
  modeling.json       # LLM modeling entries (curated)
  ideas.json          # project ideas (synthesized)
  feed_pinned.json    # Live Papers — curated highlights (edit this)
  feed.json           # Live Papers — generated: pinned + fresh arXiv (do not hand-edit)
scripts/fetch_arxiv.py
.github/workflows/refresh-feed.yml   # daily auto-refresh of feed.json
```

## Editing content

Each curated `data/*.json` file has the shape:

```json
{
  "updated": "2026-06-15",
  "items": [
    {
      "title": "DeepSeek-R1",
      "org": "DeepSeek-AI",
      "year": "2025-01",
      "category": "model",
      "innovation": "Pure-RL reasoning training (GRPO) with verifiable rewards",
      "summary": "Two sentences of context…",
      "url": "https://arxiv.org/abs/2501.12948",
      "tags": ["reasoning", "GRPO", "open-weights"]
    }
  ]
}
```

Optional per-item fields the dashboard understands:

- `confidence` — one of `confirmed` / `secondary` / `self-reported` (rendered as a 确证/二手/自报 badge).
- `ascend` — `ready` / `partial` / `none` (Ascend-readiness badge) with optional `ascendNote`.
- `analysis` — a deeper write-up (marked ★ analyzed; counts toward the ledger).
- `source` / `verified` — provenance shown on each card as `source: arXiv · verified 2026-06-23`.
  If `source` is omitted it is inferred from the URL host; if `verified` is omitted the file's
  top-level `updated` date is used.

`ideas.json` entries additionally use `pitch`, `impact`/`difficulty`/`novelty` (0–5), `why`,
a `steps` array, and — to keep ideas executable — **`minimumExperiment`** (the smallest run that
tests the thesis) and **`successMetric`** (how you know it worked).

To add an entry, append an object to the relevant `items` array — the dashboard picks up
new `category`/`tags` values automatically as filter chips.

## Data quality & provenance

The **Overview → Data Quality Ledger** summarizes the catalog (RL · Ascend · Modeling) on four
axes (hover any cell for its definition):

- **Primary sources** — % whose source host is primary/official (arXiv · HuggingFace · GitHub ·
  official docs/vendor pages) rather than media/aggregator.
- **Confidence labelled** — % carrying a credibility label, with the `confirmed / secondary /
  self-reported` breakdown. (Labels are auto-assigned by source type: primary host → confirmed,
  media → secondary, a 2026 vendor model-report → self-reported.)
- **Ascend readiness** — % of **model cards** (not algorithm/framework papers) with an explicit
  `ready / partial / none` Ascend call.
- **Deep analysis** — % with a ▸ analysis paragraph beyond the one-line summary.

Every card also shows a `source: … · verified …` line so the provenance of each claim is visible
inline. Benchmark numbers for unreleased/early models are labelled **provisional**.

### Training-curve provenance

`data/curves.json` carries a top-level `synthetic` flag and a per-experiment `meta` block
(`model`, `dataset`, `hardware`, `framework`, `precision`, `seed`, `synthetic`). The **Training
Curves** tab renders this as an experiment-metadata panel with a clear **synthetic-demo vs
real-logs** banner. Generate curves with:

```bash
# synthetic demo (no run needed) — clearly flagged as synthetic in the UI
python3 ascend-rl-bench/tools/logs_to_dashboard.py --synthetic

# real run — record full provenance
python3 ascend-rl-bench/tools/logs_to_dashboard.py \
  --log logs/run/train.log --name qwen0.5b_gsm8k_grpo --device npu \
  --model Qwen2.5-0.5B-Instruct --dataset GSM8K \
  --hardware "1× Ascend 910B 64GB" --framework "MindSpeed-RL + vLLM-Ascend" \
  --precision bf16 --seed 42
```

## Local validation

Before committing, run the same checks CI runs:

```bash
node --check js/*.js                         # JS syntax
python3 -m py_compile scripts/*.py           # Python syntax
python3 scripts/validate_data.py             # data + curve-metadata schema
git diff --check                             # whitespace/conflict markers
```

`validate_data.py` checks item titles/URLs/confidence values **and** that every
`curves.json` experiment carries a complete `meta` block — so missing curve provenance fails CI.
Accessibility: tabs use a standard `tablist`/`tab`/`tabpanel` pattern (arrow-key navigation,
`aria-selected`), and inactive panels are truly `hidden` so screen readers and automation don't
read offscreen content.

## Notes

- Content is **hand-curated from primary sources** (arXiv, official repos/blogs) and is a
  point-in-time snapshot — always verify against the linked source before citing.
- Many Ascend repositories live on **gitee.com / gitcode**, not GitHub.
