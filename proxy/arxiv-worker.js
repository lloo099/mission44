/**
 * Paper-text proxy for the dashboard's "Analyze" feature (Cloudflare Worker).
 *
 * The dashboard is a static site, so the browser cannot fetch arXiv directly
 * (no CORS). This tiny worker fetches the abstract (+ optional HTML body) for an
 * arXiv URL and returns it as JSON with permissive CORS, so the Analyst can feed
 * the paper text to Claude.
 *
 * Deploy:  see proxy/README.md
 * Use:     GET https://<your-worker>/?url=https://arxiv.org/abs/2402.03300
 *          -> { "id": "2402.03300", "title": "...", "text": "..." }
 *
 * Safety: only arxiv.org / export.arxiv.org URLs are accepted, and output is
 * length-capped. It does not proxy arbitrary URLs.
 */

const ALLOWED_HOSTS = new Set(["arxiv.org", "www.arxiv.org", "export.arxiv.org"]);
const MAX_CHARS = 20000;

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
  // matches /abs/2402.03300, /pdf/2402.03300v2, bare 2402.03300, old-style hep-th/0101001
  const m = url.match(/(\d{4}\.\d{4,5})(v\d+)?/) || url.match(/([a-z-]+\/\d{7})/i);
  return m ? m[1] : null;
}

function strip(xml, tag) {
  const m = xml.match(new RegExp(`<${tag}[^>]*>([\\s\\S]*?)</${tag}>`, "i"));
  return m ? m[1].replace(/\s+/g, " ").trim() : "";
}

export default {
  async fetch(request) {
    if (request.method === "OPTIONS") return new Response(null, { headers: CORS });
    if (request.method !== "GET") return json({ error: "GET only" }, 405);

    const target = new URL(request.url).searchParams.get("url");
    if (!target) return json({ error: "missing ?url=" }, 400);

    let host;
    try { host = new URL(target).hostname; } catch (_) { return json({ error: "bad url" }, 400); }
    if (!ALLOWED_HOSTS.has(host)) return json({ error: "only arxiv.org URLs are allowed" }, 403);

    const id = arxivId(target);
    if (!id) return json({ error: "could not parse arXiv id" }, 422);

    // Use the arXiv Atom API for a clean title + abstract (stable, no scraping).
    const apiUrl = `http://export.arxiv.org/api/query?id_list=${encodeURIComponent(id)}`;
    let title = "", abstract = "";
    try {
      const r = await fetch(apiUrl, { headers: { "User-Agent": "rl-npu-dashboard-proxy/1.0" } });
      const xml = await r.text();
      const entry = xml.match(/<entry>([\s\S]*?)<\/entry>/i);
      const body = entry ? entry[1] : xml;
      title = strip(body, "title");
      abstract = strip(body, "summary");
    } catch (e) {
      return json({ error: "arxiv api fetch failed: " + e.message }, 502);
    }

    const text = (`Title: ${title}\n\nAbstract:\n${abstract}`).slice(0, MAX_CHARS);
    return json({ id, title, text });
  },
};
