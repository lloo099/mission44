/**
 * Paper-text proxy for the dashboard's "Analyze" feature (Cloudflare Worker).
 *
 * The dashboard is a static site, so the browser cannot fetch arXiv directly
 * (no CORS). This worker fetches an arXiv paper's title + abstract (and, with
 * ?full=1, the full body via ar5iv) and returns it as JSON with permissive CORS,
 * so the Analyst can feed the paper text to Claude.
 *
 * Deploy:  see proxy/README.md
 * Use:     GET /?url=https://arxiv.org/abs/2402.03300        -> title + abstract
 *          GET /?url=https://arxiv.org/abs/2402.03300&full=1 -> + full body text
 *          -> { "id": "...", "title": "...", "text": "...", "full": true|false }
 *
 * Safety: only arxiv.org / export.arxiv.org input URLs are accepted, and output
 * is length-capped. It is not a general-purpose open proxy.
 */

const ALLOWED_HOSTS = new Set(["arxiv.org", "www.arxiv.org", "export.arxiv.org"]);
const MAX_ABSTRACT = 20000;
const MAX_FULL = 45000;

const CORS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "GET, OPTIONS",
  "Access-Control-Allow-Headers": "content-type",
};

function json(obj, status = 200) {
  return new Response(JSON.stringify(obj), {
    status,
    headers: { "content-type": "application/json", ...CORS },
  });
}

function arxivId(url) {
  const m = url.match(/(\d{4}\.\d{4,5})(v\d+)?/) || url.match(/([a-z-]+\/\d{7})/i);
  return m ? m[1] : null;
}

function strip(xml, tag) {
  const m = xml.match(new RegExp(`<${tag}[^>]*>([\\s\\S]*?)</${tag}>`, "i"));
  return m ? m[1].replace(/\s+/g, " ").trim() : "";
}

// crude but dependency-free HTML → text
function htmlToText(html) {
  // drop the heavy/irrelevant chunks first
  let s = html
    .replace(/<script[\s\S]*?<\/script>/gi, " ")
    .replace(/<style[\s\S]*?<\/style>/gi, " ")
    .replace(/<head[\s\S]*?<\/head>/gi, " ")
    .replace(/<nav[\s\S]*?<\/nav>/gi, " ")
    .replace(/<footer[\s\S]*?<\/footer>/gi, " ");
  // try to keep just the article body if present
  const art = s.match(/<article[\s\S]*?<\/article>/i);
  if (art) s = art[0];
  return s
    .replace(/<[^>]+>/g, " ")
    .replace(/&nbsp;/g, " ")
    .replace(/&amp;/g, "&").replace(/&lt;/g, "<").replace(/&gt;/g, ">")
    .replace(/&#?\w+;/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

async function fetchAbstract(id) {
  const apiUrl = `http://export.arxiv.org/api/query?id_list=${encodeURIComponent(id)}`;
  const r = await fetch(apiUrl, { headers: { "User-Agent": "rl-npu-dashboard-proxy/1.0" } });
  const xml = await r.text();
  const entry = xml.match(/<entry>([\s\S]*?)<\/entry>/i);
  const body = entry ? entry[1] : xml;
  return { title: strip(body, "title"), abstract: strip(body, "summary") };
}

async function fetchFull(id) {
  // ar5iv renders arXiv LaTeX as HTML; fall back gracefully if unavailable
  for (const u of [`https://ar5iv.org/html/${id}`, `https://ar5iv.labs.arxiv.org/html/${id}`]) {
    try {
      const r = await fetch(u, { headers: { "User-Agent": "rl-npu-dashboard-proxy/1.0" }, redirect: "follow" });
      if (!r.ok) continue;
      const text = htmlToText(await r.text());
      if (text && text.length > 400) return text;
    } catch (_) { /* try next */ }
  }
  return "";
}

export default {
  async fetch(request) {
    if (request.method === "OPTIONS") return new Response(null, { headers: CORS });
    if (request.method !== "GET") return json({ error: "GET only" }, 405);

    const params = new URL(request.url).searchParams;
    const target = params.get("url");
    const wantFull = params.get("full") === "1";
    if (!target) return json({ error: "missing ?url=" }, 400);

    let host;
    try { host = new URL(target).hostname; } catch (_) { return json({ error: "bad url" }, 400); }
    if (!ALLOWED_HOSTS.has(host)) return json({ error: "only arxiv.org URLs are allowed" }, 403);

    const id = arxivId(target);
    if (!id) return json({ error: "could not parse arXiv id" }, 422);

    let title = "", abstract = "";
    try { ({ title, abstract } = await fetchAbstract(id)); }
    catch (e) { return json({ error: "arxiv api fetch failed: " + e.message }, 502); }

    let body = "", full = false;
    if (wantFull) {
      body = await fetchFull(id);
      full = !!body;
    }

    const text = full
      ? `Title: ${title}\n\nAbstract:\n${abstract}\n\nFull text (truncated):\n${body}`.slice(0, MAX_FULL)
      : `Title: ${title}\n\nAbstract:\n${abstract}`.slice(0, MAX_ABSTRACT);

    return json({ id, title, text, full });
  },
};
