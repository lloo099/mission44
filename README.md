# RL-on-NPU Research Dashboard

A lightweight, zero-build **research dashboard** for tracking the intersection of
three fast-moving areas, so you can pick a concrete project to work on:

1. **Reinforcement Learning for LLMs** — RLHF, GRPO, RLVR, DPO and successors, reasoning models, training frameworks.
2. **Ascend / NPU ecosystem** — hardware (910B/910C), CANN, MindSpore/MindSpeed, vLLM-Ascend, and the state of RL-on-Ascend.
3. **LLM modeling advances** — MoE, attention variants, long context, low-precision training that matter on memory-constrained NPUs.

Plus a synthesized **Project Ideas** section with impact / difficulty / novelty ratings,
and a **Live Papers** feed pulled from arXiv.

## What changed in this upgraded version

- **Data Quality Ledger** on the Overview page summarizes source coverage, confidence labels,
  Ascend-readiness notes, and analysis coverage.
- Each card now shows a compact **source provenance line** (source type + snapshot/verified date)
  so provisional model claims are easier to separate from confirmed entries.
- Tabs use proper `tablist` / `tab` / `tabpanel` semantics; inactive panels are hidden from
  screen readers and automated checks.
- Training curves now carry explicit experiment metadata (`model`, `dataset`, `hardware`,
  `framework`, `precision`, `seed`, `runType`) so demo curves cannot be mistaken for a benchmark.
- Project ideas include a **minimum experiment** and **success signal** field to turn directions
  into executable first experiments.

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

### Source confidence fields

For fast-moving model and hardware entries, prefer adding:

```json
{
  "confidence": "confirmed",
  "verified": "2026-06-24",
  "sourceType": "official report",
  "url": "https://..."
}
```

Allowed confidence values are `confirmed`, `secondary`, and `self-reported` (Chinese aliases
`确证`, `二手`, `自报` are also supported). If `verified` is omitted, the dashboard falls back to
the parent data file's `updated` date.

## Training curve provenance

Training curves live in `data/curves.json` and should include run metadata for each experiment.
Generate demo curves:

```bash
python3 ascend-rl-bench/tools/logs_to_dashboard.py --synthetic
```

Parse a real run:

```bash
python3 ascend-rl-bench/tools/logs_to_dashboard.py \
  --log ascend-rl-bench/logs/qwen0.5b_gsm8k_grpo-gpu/train.log \
  --name qwen0.5b_gsm8k_grpo \
  --device gpu \
  --model Qwen2.5-0.5B \
  --dataset GSM8K \
  --hardware "8x H100" \
  --framework "verl@<commit>" \
  --precision bf16 \
  --seed 1
```

The dashboard displays this metadata above the chart; CI warns when curve experiments omit it.

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
  curves.json         # training curves + run metadata
  feed_pinned.json    # Live Papers — curated highlights (edit this)
  feed.json           # Live Papers — generated: pinned + fresh arXiv (do not hand-edit)
scripts/fetch_arxiv.py
scripts/validate_data.py
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

`ideas.json` entries additionally use `pitch`, `impact`/`difficulty`/`novelty` (0–5),
`why`, `minimumExperiment`, `successMetric`, and a `steps` array.

To add an entry, append an object to the relevant `items` array — the dashboard picks up
new `category`/`tags` values automatically as filter chips.

## Local checks

```bash
for f in js/*.js; do node --check "$f"; done
python3 -m py_compile scripts/*.py ascend-rl-bench/tools/logs_to_dashboard.py
python3 scripts/validate_data.py
```

## Notes

- Content is **hand-curated from primary sources** (arXiv, official repos/blogs) and is a
  point-in-time snapshot — always verify against the linked source before citing.
- Many Ascend repositories live on **gitee.com / gitcode**, not GitHub.
