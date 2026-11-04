#!/usr/bin/env python3
"""Link health check: extract URLs from data/*.json + docs/**/*.md + top-level docs,
HEAD/GET each, and fail only on hard 404s (timeouts/anti-bot are reported, not fatal).

The docs glob is recursive, so the dispatches under docs/blog/ and anything under
docs/proposals/ are covered — they used to be skipped by a flat docs/*.md glob
while holding most of the site's links. That takes the URL set from ~360 to ~920,
which is many hours probed one at a time at a 25s timeout, so probes run in a
small thread pool. Output stays sorted by URL regardless of completion order.

Run weekly in CI or manually."""
import concurrent.futures, glob, re, sys, urllib.request, urllib.error

URL_RE = re.compile(r'https?://[^\s"<>)\]]+')
# Kept low on purpose: arxiv.org alone supplies ~300 of the URLs and answers
# bursts with 429. A 429 is reported, not fatal, but hammering is still rude.
WORKERS = 12
urls = set()
# "docs/**/*.md" with recursive=True also matches the flat docs/*.md files,
# since ** matches zero or more path segments.
files = (glob.glob("data/*.json")
         + glob.glob("docs/**/*.md", recursive=True)
         + ["README.md", "PLAN.md", "REPORT.md"])
for f in files:
    try:
        txt = open(f, encoding="utf-8").read()
    except OSError:
        continue
    for m in URL_RE.findall(txt):
        urls.add(m.rstrip('.,'))

def probe(u):
    for method, ua in (("HEAD", "mission44-linkcheck/1.0"), ("GET", "Mozilla/5.0")):
        try:
            req = urllib.request.Request(u, method=method, headers={"User-Agent": ua})
            return urllib.request.urlopen(req, timeout=25).status
        except urllib.error.HTTPError as e:
            if e.code in (403, 405, 429) and method == "HEAD":
                continue  # retry as GET
            return e.code
        except Exception:
            return 0  # network/timeout — non-fatal
    return 0

ordered = sorted(urls)
with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as pool:
    codes = list(pool.map(probe, ordered))

bad = []
for u, code in zip(ordered, codes):
    print(f"{code:>3}  {u}")
    if code == 404:
        bad.append(u)
print(f"\n[links] checked {len(urls)} urls across {len(files)} files")
if bad:
    print("404s:")
    for u in bad:
        print("  -", u)
    sys.exit(1)
print("[links] OK")
