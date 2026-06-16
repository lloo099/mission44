# Paper-text proxy (optional)

The dashboard's **✨ Analyze** feature can optionally feed the linked paper's
text to Claude. Because the dashboard is a static site, the browser can't fetch
arXiv directly (no CORS) — this tiny proxy does it server-side and returns the
abstract as JSON with CORS enabled.

It's **optional**: without a proxy, Analyze still works using the dashboard's
own summary. With one, the model also sees the paper's title + abstract.

## What it does

`GET /?url=https://arxiv.org/abs/2402.03300`
→ `{ "id": "2402.03300", "title": "...", "text": "Title: …\n\nAbstract:\n…", "full": false }`

`GET /?url=https://arxiv.org/abs/2402.03300&full=1`
→ also includes the paper body (via ar5iv), `"full": true` — capped at ~45k chars.

- Only `arxiv.org` / `export.arxiv.org` input URLs are accepted (not an open proxy).
- Abstract comes from the official arXiv Atom API; full text from ar5iv HTML
  (falls back to abstract-only if ar5iv has no render).
- Output is length-capped (20k abstract / 45k full).
- The dashboard requests `full=1` by default; full text costs more tokens.

## Deploy on Cloudflare Workers (free tier)

```bash
npm i -g wrangler
wrangler login
# in this proxy/ dir:
wrangler deploy arxiv-worker.js --name rl-npu-paper-proxy --compatibility-date 2024-01-01
```

Wrangler prints a URL like `https://rl-npu-paper-proxy.<you>.workers.dev`.

### Alternative: Vercel / Deno Deploy

The handler is a standard `fetch(request)` export — adapt it to your platform's
entrypoint if you prefer. The logic (allow-list + arXiv API + CORS) is unchanged.

## Wire it into the dashboard

1. Open the dashboard, click **✨ Analyze** on any card → **⚙**.
2. Paste the worker URL when prompted for the **paper-text proxy URL**.

It's stored only in your browser (`localStorage` key `analyst_proxy_url`).
Leave it blank to disable.

## Notes / hardening

- The allow-list keeps this from becoming a general-purpose open proxy. Keep it.
- To support non-arXiv links later, add hosts to `ALLOWED_HOSTS` and a matching
  text extractor — but be deliberate about which domains you proxy.
- No API keys are involved here; the Anthropic key stays in the browser and goes
  straight to Anthropic, never through this proxy.
