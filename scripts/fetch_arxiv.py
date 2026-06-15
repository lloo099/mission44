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
    "ascend-npu": 'abs:Ascend OR abs:"NPU" OR abs:CANN OR abs:MindSpore',
    "efficient-llm": 'abs:"mixture of experts" OR abs:"FP8 training" OR abs:"long context" OR abs:quantization AND abs:LLM',
}

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "data", "feed.json")
NS = {"a": "http://www.w3.org/2005/Atom"}


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

    all_items, seen = [], set()
    for bucket, q in QUERIES.items():
        try:
            print(f"[fetch] {bucket} …", file=sys.stderr)
            data = fetch(q, args.max)
            for it in parse(data, bucket):
                key = it["url"]
                if key and key not in seen:
                    seen.add(key)
                    all_items.append(it)
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] {bucket} failed: {exc}", file=sys.stderr)

    all_items.sort(key=lambda x: x["year"], reverse=True)
    payload = {
        "updated": dt.datetime.utcnow().isoformat() + "Z",
        "source": "arXiv API",
        "count": len(all_items),
        "items": all_items,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"[done] wrote {len(all_items)} items -> {os.path.relpath(OUT)}", file=sys.stderr)


if __name__ == "__main__":
    main()
