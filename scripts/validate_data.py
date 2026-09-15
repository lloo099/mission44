#!/usr/bin/env python3
"""Validate data/*.json for the dashboard. Run in CI to block broken commits.
Only files that have an "items" list are item-validated (compare/curves skipped)."""
import glob, hashlib, json, re, sys

# 可信度分级(与 js/app.js 的 confBadge 一一对应,改动时必须同步):
#   confirmed    确证    多方独立证实
#   secondary    二手    单一转述/媒体报道
#   self-reported 自报   厂商或作者口径
#   provisional  待核实  已收录但尚未核验,后续需回补证据
OK_CONF = {
    "confirmed", "secondary", "self-reported", "provisional",
    "确证", "二手", "自报", "待核实",
    None,
}
errs = []
for f in sorted(glob.glob("data/*.json")):
    try:
        d = json.load(open(f, encoding="utf-8"))
    except Exception as e:
        errs.append(f"{f}: invalid JSON: {e}"); continue
    items = d.get("items") if isinstance(d, dict) else d
    if items is None:
        continue  # compare.json / curves.json etc. — not item lists
    if not isinstance(items, list):
        errs.append(f"{f}: 'items' is not a list"); continue
    for i, it in enumerate(items):
        if not isinstance(it, dict):
            errs.append(f"{f}[{i}]: not an object"); continue
        if not it.get("title"):
            errs.append(f"{f}[{i}]: missing title")
        u = it.get("url")
        if u is not None and not (isinstance(u, str) and u.startswith(("http://", "https://", "docs/", "data/"))):
            errs.append(f"{f}[{i}] ({it.get('title','?')[:30]}): bad url {u!r}")
        if it.get("confidence") not in OK_CONF:
            errs.append(f"{f}[{i}] ({it.get('title','?')[:30]}): bad confidence {it.get('confidence')!r}")

# ---- curves.json: validate experiment metadata (provenance) ----
CURVE_META_KEYS = ("model", "dataset", "hardware", "framework", "precision", "seed")
import os
if os.path.exists("data/curves.json"):
    try:
        cv = json.load(open("data/curves.json", encoding="utf-8"))
    except Exception as e:
        errs.append(f"data/curves.json: invalid JSON: {e}"); cv = None
    if isinstance(cv, dict):
        exps = cv.get("experiments")
        if not isinstance(exps, list) or not exps:
            errs.append("data/curves.json: 'experiments' missing or empty")
        else:
            for i, e in enumerate(exps):
                tag = f"data/curves.json[{i}] ({e.get('name','?')}·{e.get('device','?')})"
                if not e.get("name") or not e.get("device"):
                    errs.append(f"{tag}: missing name/device")
                if not isinstance(e.get("metrics"), dict) or not e.get("metrics"):
                    errs.append(f"{tag}: missing metrics")
                meta = e.get("meta")
                if not isinstance(meta, dict):
                    errs.append(f"{tag}: missing meta block (model/dataset/hardware/framework/precision/seed)")
                else:
                    missing = [k for k in CURVE_META_KEYS if meta.get(k) in (None, "")]
                    if missing:
                        errs.append(f"{tag}: meta missing {missing}")
                    # A run presented as measured must carry evidence it can be
                    # traced back to: which log file, its hash, how many steps.
                    if meta.get("synthetic") is False:
                        src = e.get("source")
                        if not isinstance(src, dict):
                            errs.append(f"{tag}: non-synthetic run needs a 'source' block "
                                        f"(logPath/logSha256/steps) — regenerate with logs_to_dashboard.py")
                        else:
                            for k in ("logPath", "logSha256", "steps"):
                                if src.get(k) in (None, ""):
                                    errs.append(f"{tag}: source missing {k!r}")
                            sha = src.get("logSha256")
                            if sha is not None and not re.fullmatch(r"[0-9a-f]{64}", str(sha)):
                                errs.append(f"{tag}: source.logSha256 is not a sha256 hex digest")

        # The key must be present, not merely valid-if-present: a tool that
        # rewrites this file used to drop it entirely, and a default of [] made
        # that loss invisible here.
        if "publishedEvidence" not in cv:
            errs.append("data/curves.json: 'publishedEvidence' key is missing "
                        "(a writer dropped it — cited real-world runs must survive rewrites)")
        published = cv.get("publishedEvidence", [])
        if not isinstance(published, list):
            errs.append("data/curves.json: 'publishedEvidence' is not a list")
        else:
            required = (
                "id", "name", "kind", "status", "reported", "verified", "model",
                "algorithm", "hardware", "framework", "sourceTitle", "sourceUrl",
                "sourceType", "rawSamples", "caveat", "charts",
            )
            for i, run in enumerate(published):
                tag = f"data/curves.json.publishedEvidence[{i}]"
                if not isinstance(run, dict):
                    errs.append(f"{tag}: not an object"); continue
                missing = [k for k in required if k not in run or run.get(k) in (None, "")]
                if missing:
                    errs.append(f"{tag}: missing {missing}")
                if run.get("status") != "real-published":
                    errs.append(f"{tag}: status must be 'real-published'")
                if not isinstance(run.get("rawSamples"), bool):
                    errs.append(f"{tag}: rawSamples must be boolean")
                source_url = run.get("sourceUrl", "")
                if not isinstance(source_url, str) or not source_url.startswith("https://"):
                    errs.append(f"{tag}: sourceUrl must be https")
                charts = run.get("charts")
                if not isinstance(charts, list) or not charts:
                    errs.append(f"{tag}: charts missing or empty")
                    continue
                for j, chart in enumerate(charts):
                    ctag = f"{tag}.charts[{j}]"
                    if not isinstance(chart, dict) or not chart.get("metric") or not chart.get("asset"):
                        errs.append(f"{ctag}: missing metric/asset"); continue
                    asset = chart["asset"]
                    if not isinstance(asset, str) or not asset.startswith("assets/curves/"):
                        errs.append(f"{ctag}: asset must live under assets/curves/")
                    elif not os.path.exists(asset):
                        errs.append(f"{ctag}: asset not found: {asset}")
                    else:
                        expected_hash = chart.get("sha256", "")
                        with open(asset, "rb") as f:
                            actual_hash = hashlib.sha256(f.read()).hexdigest()
                        if expected_hash != actual_hash:
                            errs.append(f"{ctag}: sha256 mismatch")
                    original = chart.get("originalUrl", "")
                    if not isinstance(original, str) or not original.startswith("https://"):
                        errs.append(f"{ctag}: originalUrl must be https")

print(f"[validate] checked {len(glob.glob('data/*.json'))} files")
if errs:
    print("VALIDATION ERRORS:")
    for e in errs:
        print("  -", e)
    sys.exit(1)
print("[validate] data OK")
