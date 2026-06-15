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

This writes `data/feed.json`, which the dashboard reads. No third-party packages required
(standard library only). Re-run it whenever you want fresh papers, then commit the updated JSON.

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
  feed.json           # Live Papers (script-refreshable)
scripts/fetch_arxiv.py
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
`why`, and a `steps` array.

To add an entry, append an object to the relevant `items` array — the dashboard picks up
new `category`/`tags` values automatically as filter chips.

## Notes

- Content is **hand-curated from primary sources** (arXiv, official repos/blogs) and is a
  point-in-time snapshot — always verify against the linked source before citing.
- Many Ascend repositories live on **gitee.com / gitcode**, not GitHub.
