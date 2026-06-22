#!/usr/bin/env python3
"""Refresh data/feed.json with the latest arXiv papers for the dashboard.

This is the reliable "live" path: the browser cannot fetch arXiv directly
(no CORS headers), so run this script to pull fresh results into a JSON file
that the dashboard reads.

Usage:
    python3 scripts/fetch_arxiv.py
    python3 scripts/fetch_arxiv.py --max 40

No third-party dependencies — uses only the Python standard library.
"""
import argparse
import datetime as dt
import json
import os
import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

ARXIV_API = "http://export.arxiv.org/api/query"

# Each query becomes a tag-able bucket in the feed.
QUERIES = {
    "rl-llm": 'abs:"reinforcement learning" AND (abs:"large language model" OR abs:LLM OR abs:RLHF OR abs:GRPO)',
    "reasoning-rl": 'abs:"verifiable reward" OR abs:RLVR OR (abs:reasoning AND abs:"reinforcement learning")',
    "agentic-rl": '(abs:agentic OR abs:"LLM agent" OR abs:"multi-turn") AND abs:"reinforcement learning"',
    "ascend-npu": 'abs:Ascend OR abs:"NPU" OR abs:CANN OR abs:MindSpore OR abs:"Da Vinci"',
    "sparse-attention": '(abs:"sparse attention" OR abs:"KV cache" OR abs:"long context") AND (abs:LLM OR abs:transformer)',
    "fp8-lowprec": '(abs:"FP8" OR abs:"low-precision" OR abs:"low precision" OR abs:"MXFP4") AND (abs:training OR abs:LLM)',
    "efficient-llm": 'abs:"mixture of experts" OR (abs:quantization AND abs:LLM)',
}

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "data", "feed.json")
PINNED = os.path.join(HERE, "..", "data", "feed_pinned.json")
NS = {"a": "http://www.w3.org/2005/Atom"}


def load_pinned():
    """Curated highlights that must always stay in the feed (with their
    hand-written summaries, official links, and Ascend-readiness badges)."""
    try:
        with open(PINNED, encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return []
    return data.get("items", []) if isinstance(data, dict) else (data or [])


def fetch(query: str, max_results: int):
    params = {
        "search_query": query,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
        "max_results": str(max_results),
    }
    url = f"{ARXIV_API}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": "rl-npu-dashboard/1.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read()


def parse(xml_bytes: bytes, bucket: str):
    root = ET.fromstring(xml_bytes)
    items = []
    for e in root.findall("a:entry", NS):
        title = (e.findtext("a:title", default="", namespaces=NS) or "").strip().replace("\n", " ")
        summary = (e.findtext("a:summary", default="", namespaces=NS) or "").strip().replace("\n", " ")
        published = (e.findtext("a:published", default="", namespaces=NS) or "")[:10]
        link = e.findtext("a:id", default="", namespaces=NS) or ""
        authors = [a.findtext("a:name", default="", namespaces=NS) for a in e.findall("a:author", NS)]
        org = authors[0] if authors else ""
        if len(authors) > 1:
            org += f" +{len(authors) - 1}"
        items.append({
            "title": title,
            "org": org,
            "year": published,
            "summary": (summary[:300] + "…") if len(summary) > 300 else summary,
            "url": link,
            "category": bucket,
            "tags": ["live", bucket],
        })
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=12, help="max results per query bucket")
    args = ap.parse_args()

    pinned = load_pinned()
    pinned_urls = {it.get("url") for it in pinned if it.get("url")}

    live, seen = [], set()
    for bucket, q in QUERIES.items():
        try:
            print(f"[fetch] {bucket} …", file=sys.stderr)
            data = fetch(q, args.max)
            for it in parse(data, bucket):
                key = it["url"]
                if key and key not in seen and key not in pinned_urls:
                    seen.add(key)
                    it.setdefault("tags", []).append("auto")
                    live.append(it)
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] {bucket} failed: {exc}", file=sys.stderr)

    live.sort(key=lambda x: x["year"], reverse=True)
    items = pinned + live  # curated highlights first, then freshest arXiv
    payload = {
        "updated": dt.datetime.utcnow().isoformat() + "Z",
        "source": "curated (data/feed_pinned.json) + arXiv API",
        "count": len(items),
        "pinned": len(pinned),
        "items": items,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"[done] wrote {len(items)} items ({len(pinned)} pinned + {len(live)} live) -> {os.path.relpath(OUT)}", file=sys.stderr)


if __name__ == "__main__":
    main()
