#!/usr/bin/env python3
"""Link health check: extract URLs from data/*.json + docs/*.md + top-level docs,
HEAD/GET each, and fail only on hard 404s (timeouts/anti-bot are reported, not fatal).
Run weekly in CI or manually."""
import glob, re, sys, urllib.request, urllib.error

URL_RE = re.compile(r'https?://[^\s"<>)\]]+')
urls = set()
files = glob.glob("data/*.json") + glob.glob("docs/*.md") + ["README.md", "PLAN.md", "REPORT.md"]
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

bad = []
for u in sorted(urls):
    code = probe(u)
    print(f"{code:>3}  {u}")
    if code == 404:
        bad.append(u)
print(f"\n[links] checked {len(urls)} urls")
if bad:
    print("404s:")
    for u in bad:
        print("  -", u)
    sys.exit(1)
print("[links] OK")
