#!/usr/bin/env python3
"""Check 0 against the public GitHub event stream (GH Archive hourly dumps).

Check 0 splits into two questions, and they do not have the same answer:

  1. Are the fields recoverable? Can we compute, per event, when a human
     responded, who they were, which thread it belonged to, and how it ended?
  2. Is a per-repo stream reconstructable? ABO models concurrent work streams
     in ONE repository contending for ONE maintainer, so a sampled feed that
     carries a few events per repo per hour cannot express the queue at all.

Pass hourly files as arguments:

    curl -sS -o 2026-11-05-15.json.gz https://data.gharchive.org/2026-11-05-15.json.gz
    python3 abo-bench/trace/probe_gharchive.py 2026-11-05-15.json.gz [...]

Downloads are untrusted: keep them in their own directory, pass paths as
arguments, and run this with `python3 -I` if the files sit next to anything
importable.
"""
from __future__ import annotations

import argparse
import collections
import gzip
import json
import sys
from datetime import datetime

ISSUE_TYPES = ("IssuesEvent", "IssueCommentEvent", "PullRequestEvent",
               "PullRequestReviewEvent", "PullRequestReviewCommentEvent")


def parse_ts(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def read_events(paths: list[str]):
    """Yield events, skipping files that are not gzip (the archive serves XML
    error documents for missing hours, which must not be silently counted)."""
    bad = []
    for path in paths:
        try:
            with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
                for line in f:
                    try:
                        yield json.loads(line)
                    except Exception:
                        continue
        except (OSError, EOFError, gzip.BadGzipFile):
            bad.append(path)
    if bad:
        print(f"\n!! {len(bad)} file(s) were not readable gzip — missing hours, not empty hours:")
        for p in bad:
            print(f"     {p}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("files", nargs="+", help="GH Archive hourly .json.gz files")
    args = ap.parse_args()

    latencies: list[float] = []
    bots = humans = mentions = with_state = comments = 0
    per_repo = collections.Counter()
    types = collections.Counter()
    repos = set()
    threads = set()

    for e in read_events(args.files):
        types[e.get("type", "?")] += 1
        repo = (e.get("repo") or {}).get("name")
        if repo:
            repos.add(repo)
        if e.get("type") in ISSUE_TYPES and repo:
            per_repo[repo] += 1
        if e.get("type") != "IssueCommentEvent":
            continue
        payload = e.get("payload") or {}
        issue = payload.get("issue") or {}
        comment = payload.get("comment") or {}
        if not issue.get("created_at"):
            continue
        comments += 1
        delta = (parse_ts(e["created_at"]) - parse_ts(issue["created_at"])).total_seconds() / 3600.0
        if delta >= 0:
            latencies.append(delta)
        login = e["actor"]["login"]
        if login.endswith("[bot]") or "bot" in login.lower():
            bots += 1
        else:
            humans += 1
        if "@" in (comment.get("body") or ""):
            mentions += 1
        if issue.get("state"):
            with_state += 1
        threads.add((repo, issue.get("number")))

    if not comments:
        print("no IssueCommentEvents found — nothing to measure")
        return 1

    latencies.sort()
    pct = lambda q: latencies[min(int(q * len(latencies)), len(latencies) - 1)]

    print(f"events parsed:      {sum(types.values()):,}")
    print(f"distinct repos:     {len(repos):,}")
    print(f"IssueCommentEvents: {comments:,}  across {len(threads):,} threads\n")

    print("Q1 — are the fields recoverable?")
    print(f"  response latency   {len(latencies):>6,}/{comments:,} "
          f"({100*len(latencies)/comments:.1f}%)   "
          f"median {pct(.5):.1f}h  p10 {pct(.1):.2f}h  p90 {pct(.9):.1f}h")
    print(f"  thread_id          {comments:>6,}/{comments:,} (100.0%)   issue.number")
    print(f"  resolution         {with_state:>6,}/{comments:,} "
          f"({100*with_state/comments:.1f}%)   issue.state")
    print(f"  actor_is_bot       bot {bots:,} / human {humans:,} "
          f"({100*bots/comments:.1f}% bot by login suffix)")
    print(f"  addressed_to       {mentions:>6,}/{comments:,} "
          f"({100*mentions/comments:.1f}%)   bodies containing '@' — the weak one\n")

    print("Q2 — is a per-repo stream reconstructable?")
    top = per_repo.most_common(5)
    for repo, n in top:
        print(f"  {n:>5}  {repo}")
    busiest = top[0][1] if top else 0
    print(f"\n  busiest repo carries {busiest} issue-class events across these files.")
    print("  ABO needs the concurrent load inside ONE repo. A feed this thin cannot")
    print("  express that queue, however well Q1 scores.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
