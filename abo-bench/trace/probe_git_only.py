#!/usr/bin/env python3
"""How much of Check 0 can a git clone answer, with no GitHub API access?

Check 0 asks whether a trace lets us recover when a human responded and what
they did. The API route (issues, pull requests, comments) is unavailable to
this session, so this measures what git history alone supplies, field by field,
against the 14 fields declared in schema.py.

This is a PARTIAL probe. It cannot settle Check 0, and it says so in its
output rather than implying a pass.

Usage:
    python3 abo-bench/trace/probe_git_only.py /path/to/clone [--days 7]
"""
from __future__ import annotations

import argparse
import collections
import re
import subprocess
import sys
from datetime import datetime, timedelta, timezone

# Squash-merge and merge-commit conventions that link a commit back to a thread.
PR_SQUASH = re.compile(r"\(#(\d+)\)\s*$")
PR_MERGE = re.compile(r"^Merge pull request #(\d+)")
BOT_HINTS = ("[bot]", "github-actions", "dependabot", "renovate", "copilot")

SEP = "\x1f"


def git_log(repo: str, since: datetime) -> list[dict]:
    fmt = SEP.join(["%H", "%an", "%ae", "%cn", "%ce", "%aI", "%cI", "%s"])
    out = subprocess.run(
        ["git", "-C", repo, "log", f"--since={since.isoformat()}", f"--pretty=format:{fmt}"],
        capture_output=True, text=True, check=True,
    ).stdout
    rows = []
    for line in out.splitlines():
        if not line.strip():
            continue
        parts = line.split(SEP)
        if len(parts) != 8:
            continue
        sha, an, ae, cn, ce, ad, cd, subj = parts
        rows.append({
            "sha": sha, "author": an, "author_email": ae,
            "committer": cn, "committer_email": ce,
            "authored_at": datetime.fromisoformat(ad),
            "committed_at": datetime.fromisoformat(cd),
            "subject": subj,
        })
    return rows


def is_bot(name: str, email: str) -> bool:
    blob = f"{name} {email}".lower()
    return any(h in blob for h in BOT_HINTS)


def pr_number(subject: str) -> int | None:
    for rx in (PR_SQUASH, PR_MERGE):
        m = rx.search(subject)
        if m:
            return int(m.group(1))
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("repo", help="path to a git clone")
    ap.add_argument("--days", type=int, default=7)
    args = ap.parse_args()

    since = datetime.now(timezone.utc) - timedelta(days=args.days)
    rows = git_log(args.repo, since)
    if not rows:
        print(f"no commits in the last {args.days} days")
        return 1

    bots = [r for r in rows if is_bot(r["author"], r["author_email"])]
    linked = [r for r in rows if pr_number(r["subject"]) is not None]
    humans = [r for r in rows if not is_bot(r["author"], r["author_email"])]
    committers = collections.Counter(r["committer"] for r in rows)
    authors = collections.Counter(r["author"] for r in humans)

    # authored -> committed lag: for a squash-merged PR this is roughly the time
    # between the contributor's work and a maintainer landing it. It is a proxy
    # with real limits, stated below.
    lags = [(r["committed_at"] - r["authored_at"]).total_seconds() / 3600.0 for r in rows]
    lags_sorted = sorted(lags)
    median_lag = lags_sorted[len(lags_sorted) // 2]

    print(f"repo: {args.repo}")
    print(f"window: last {args.days} days (since {since.date()})\n")
    print(f"commits:            {len(rows)}")
    print(f"  bot-authored:     {len(bots)} ({100*len(bots)/len(rows):.1f}%)")
    print(f"  linked to a PR #: {len(linked)} ({100*len(linked)/len(rows):.1f}%)")
    print(f"distinct human authors:    {len(authors)}")
    print(f"distinct committers:       {len(committers)}")
    print(f"median authored->committed lag: {median_lag:.1f} h\n")

    print("top committers (the maintainers who land work):")
    for name, n in committers.most_common(5):
        print(f"  {n:>4}  {name}")
    print()

    # Field-by-field verdict against schema.py's 14 required fields.
    verdict = {
        "event_id":        ("YES", "commit sha"),
        "event_type":      ("PARTIAL", "commit only; no issue/PR/comment event types"),
        "created_at":      ("YES", "author/committer dates"),
        "repo":            ("YES", "known from the clone"),
        "thread_id":       ("PARTIAL", f"PR # recoverable for {100*len(linked)/len(rows):.0f}% of commits, "
                                       "only for merged work; open and abandoned threads invisible"),
        "actor_login":     ("PARTIAL", "git name/email, not the GitHub login"),
        "actor_is_bot":    ("PARTIAL", "heuristic on name/email; bots posting as users are missed"),
        "addressed_to":    ("NO", "needs comment bodies"),
        "parent_event_id": ("NO", "needs thread structure"),
        "body_len":        ("NO", "needs comment bodies"),
        "diff_added":      ("YES", "git diff --numstat"),
        "diff_removed":    ("YES", "git diff --numstat"),
        "resolution":      ("PARTIAL", "merged work is visible; closed-without-merge and stale are not"),
        "resolved_at":     ("PARTIAL", "merge time only, for merged work"),
    }
    counts = collections.Counter(v[0] for v in verdict.values())
    print("schema.py fields recoverable from git alone:")
    for field, (status, why) in verdict.items():
        print(f"  [{status:>7}] {field:<16} {why}")
    print(f"\n  YES {counts['YES']}   PARTIAL {counts['PARTIAL']}   NO {counts['NO']}  (of {len(verdict)})\n")

    print("What this does NOT settle, and why Check 0 stays open:")
    print("  - The decisive question is human RESPONSE timing to a request. Git shows")
    print("    when work was landed, never when a maintainer first replied, asked a")
    print("    question, or declined. Those live in issue and PR comments.")
    print("  - Survivorship: only merged work leaves a commit. Requests that were")
    print("    ignored or closed unmerged are exactly the cases an attention ledger")
    print("    must count, and they are invisible here.")
    print("  - authored->committed lag is not response latency. It mixes the")
    print("    contributor's own delay with review time, and squash-merges reset it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
