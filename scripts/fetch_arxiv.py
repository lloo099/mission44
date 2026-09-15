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
import re
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

BUCKET_TERMS = {
    "rl-llm": ("language model", "rlhf", "rlvr", "llm", "post-training", "reasoning model"),
    "reasoning-rl": ("language model", "rlvr", "verifiable reward", "reasoning model", "post-training"),
    "agentic-rl": ("language model", "llm", "agentic", "tool use", "tool-use", "tool calling", "multi-turn", "web agent", "software agent"),
    "ascend-npu": ("ascend", "cann", "mindspore", "da vinci", "huawei npu", "910b"),
    "sparse-attention": ("sparse attention", "kv cache", "kv-cache", "long context", "long-context"),
    "fp8-lowprec": ("fp8", "mxfp4", "fp4", "low-precision", "low precision"),
}


def has_term(text: str, term: str) -> bool:
    if term == "llm":
        return bool(re.search(r"\bllms?\b", text))
    if term == "ascend":
        return bool(re.search(r"\bascend\b", text))
    return term in text


def relevant(bucket: str, title: str, summary: str) -> bool:
    """Reject broad arXiv boolean matches that do not fit the dashboard domain."""
    text = f"{title} {summary[:360]}".lower()
    if bucket == "efficient-llm":
        efficient = ("mixture of experts", "mixture-of-experts", "quantization", "quantized", "moe")
        model = ("language model", "llm", "transformer")
        return any(has_term(text, t) for t in efficient) and any(has_term(text, t) for t in model)
    if bucket == "fp8-lowprec":
        precision = ("fp8", "mxfp4", "fp4", "low-precision", "low precision")
        model = ("language model", "llm", "transformer", "model training")
        return any(has_term(text, t) for t in precision) and any(has_term(text, t) for t in model)
    return any(has_term(text, term) for term in BUCKET_TERMS.get(bucket, ()))


def load_pinned():
    """Curated highlights that must always stay in the feed (with their
    hand-written summaries, official links, and Ascend-readiness badges)."""
    try:
        with open(PINNED, encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return []
    return data.get("items", []) if isinstance(data, dict) else (data or [])


def curated_urls():
    """URLs already shown as curated cards in the domain tabs — skip them in the
    auto feed so the same paper doesn't appear twice."""
    s = set()
    for fn in ("rl.json", "ascend.json", "modeling.json"):
        try:
            with open(os.path.join(HERE, "..", "data", fn), encoding="utf-8") as f:
                for it in (json.load(f).get("items") or []):
                    if it.get("url"):
                        s.add(it["url"])
        except (OSError, json.JSONDecodeError):
            pass
    return s


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
        if not relevant(bucket, title, summary):
            continue
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


def load_previous():
    """The last written feed, so a failed refresh can fall back to it."""
    try:
        with open(OUT, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except (FileNotFoundError, ValueError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=12, help="max results per query bucket")
    args = ap.parse_args()

    pinned = load_pinned()
    prev = load_previous()
    prev_auto = [it for it in (prev or {}).get("items", [])
                 if isinstance(it, dict) and "auto" in (it.get("tags") or [])]
    pinned_urls = {it.get("url") for it in pinned if it.get("url")}
    skip_urls = pinned_urls | curated_urls()  # avoid duplicating curated cards

    live, seen = [], set()
    ok_buckets, failed_buckets = [], []
    for bucket, q in QUERIES.items():
        try:
            print(f"[fetch] {bucket} …", file=sys.stderr)
            data = fetch(q, args.max)
            for it in parse(data, bucket):
                key = it["url"]
                if key and key not in seen and key not in skip_urls:
                    seen.add(key)
                    it.setdefault("tags", []).append("auto")
                    live.append(it)
            ok_buckets.append(bucket)
        except Exception as exc:  # noqa: BLE001
            failed_buckets.append(bucket)
            print(f"[warn] {bucket} failed: {exc}", file=sys.stderr)

    # A failed refresh must never silently empty the feed: fall back to the last
    # good auto items and mark the payload stale instead of publishing a hole.
    reused = 0
    if not live and prev_auto:
        live = prev_auto
        reused = len(prev_auto)
        print(f"[warn] no live results — keeping {reused} auto item(s) from the last successful run",
              file=sys.stderr)

    live.sort(key=lambda x: x["year"], reverse=True)
    items = pinned + live  # curated highlights first, then freshest arXiv

    now = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    # `updated` = last run that actually produced live data; `checked` = last attempt.
    # They diverge exactly when a refresh fails, which is what the UI surfaces.
    fresh = bool(ok_buckets) and reused == 0
    updated = now if fresh else ((prev or {}).get("updated") or now)

    payload = {
        "updated": updated,
        "checked": now,
        "stale": not fresh,
        "source": "curated (data/feed_pinned.json) + arXiv API",
        "count": len(items),
        "pinned": len(pinned),
        "live": len(live),
        "buckets_ok": ok_buckets,
        "buckets_failed": failed_buckets,
        "items": items,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    state = "fresh" if fresh else f"STALE (reused {reused} auto item(s))"
    print(f"[done] wrote {len(items)} items ({len(pinned)} pinned + {len(live)} live) — {state}; "
          f"{len(ok_buckets)}/{len(QUERIES)} buckets ok -> {os.path.relpath(OUT)}", file=sys.stderr)

    if failed_buckets and not ok_buckets:
        print("[error] every bucket failed — feed left at its last good state", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
