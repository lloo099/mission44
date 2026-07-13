#!/usr/bin/env python3
"""Auto-refresh the ACADEMIC side of data/agentic.json from arXiv.

The Agentic RL tab is curated (industry items + trend analysis are editorial),
but the academic frontier moves fast, so this script pulls recent agentic-RL
papers and merges them as track:"academic" items marked "auto": true.

- Hand-curated items (no "auto" flag) are never touched.
- Auto items are deduped by URL (vs the file, and vs rl/modeling/ascend cards),
  capped to the most recent ones, and refreshed each run.
- trends and all other fields are preserved verbatim.

No third-party deps (stdlib only). Run weekly via .github/workflows/refresh-agentic.yml.
"""
import datetime as dt
import json
import os
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

ARXIV_API = "http://export.arxiv.org/api/query"
NS = {"a": "http://www.w3.org/2005/Atom"}
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "data", "agentic.json")
MAX_AUTO = 12          # cap auto-added academic items
MAX_PER_QUERY = 8

QUERIES = {
    "agentic-rl": '(abs:agentic OR abs:"LLM agent" OR abs:"tool use" OR abs:"multi-turn") AND abs:"reinforcement learning"',
    "long-horizon-rl": '(abs:"credit assignment" OR abs:"long-horizon" OR abs:"long horizon") AND abs:"reinforcement learning" AND (abs:agent OR abs:LLM)',
}


def fetch(query, n):
    params = {"search_query": query, "sortBy": "submittedDate",
              "sortOrder": "descending", "max_results": str(n)}
    url = f"{ARXIV_API}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": "rl-npu-dashboard/1.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read()


def parse(xml_bytes, bucket):
    root = ET.fromstring(xml_bytes)
    out = []
    for e in root.findall("a:entry", NS):
        title = (e.findtext("a:title", default="", namespaces=NS) or "").strip().replace("\n", " ")
        summary = (e.findtext("a:summary", default="", namespaces=NS) or "").strip().replace("\n", " ")
        published = (e.findtext("a:published", default="", namespaces=NS) or "")[:7]
        link = e.findtext("a:id", default="", namespaces=NS) or ""
        authors = [a.findtext("a:name", default="", namespaces=NS) for a in e.findall("a:author", NS)]
        org = authors[0] if authors else "research"
        if len(authors) > 1:
            org += f" +{len(authors) - 1}"
        if not (title and link):
            continue
        out.append({
            "title": title,
            "org": org,
            "year": published,
            "category": "algorithm",
            "track": "academic",
            "summary": (summary[:280] + "…") if len(summary) > 280 else summary,
            "url": link,
            "tags": ["academic", "agentic-rl", bucket],
            "confidence": "confirmed",
            "auto": True,
        })
    return out


def other_curated_urls():
    s = set()
    for fn in ("rl.json", "modeling.json", "ascend.json"):
        try:
            with open(os.path.join(HERE, "..", "data", fn), encoding="utf-8") as f:
                for it in (json.load(f).get("items") or []):
                    if it.get("url"):
                        s.add(it["url"])
        except (OSError, json.JSONDecodeError):
            pass
    return s


def main():
    with open(OUT, encoding="utf-8") as f:
        data = json.load(f)
    items = data.get("items", [])
    manual = [it for it in items if not it.get("auto")]
    manual_urls = {it.get("url") for it in manual}
    skip = manual_urls | other_curated_urls()

    # gather candidates: keep existing auto items + newly fetched, dedup by url
    pool, seen = [], set()
    for it in items:
        if it.get("auto") and it.get("url") and it["url"] not in seen and it["url"] not in skip:
            pool.append(it); seen.add(it["url"])
    for bucket, q in QUERIES.items():
        try:
            for it in parse(fetch(q, MAX_PER_QUERY), bucket):
                if it["url"] not in seen and it["url"] not in skip:
                    pool.append(it); seen.add(it["url"])
        except Exception as e:  # network/parse — don't fail the run
            print(f"[agentic] {bucket} failed: {e}")

    pool.sort(key=lambda it: it.get("year", ""), reverse=True)
    pool = pool[:MAX_AUTO]

    data["items"] = manual + pool
    data["updated"] = dt.date.today().isoformat()
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"[agentic] {len(manual)} curated + {len(pool)} auto academic items")


if __name__ == "__main__":
    main()
