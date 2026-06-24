#!/usr/bin/env python3
"""Validate data/*.json for the dashboard. Run in CI to block broken commits.
Only files that have an "items" list are item-validated (compare/curves skipped)."""
import glob, json, sys

OK_CONF = {"confirmed", "secondary", "self-reported", "确证", "二手", "自报", None}
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

print(f"[validate] checked {len(glob.glob('data/*.json'))} files")
if errs:
    print("VALIDATION ERRORS:")
    for e in errs:
        print("  -", e)
    sys.exit(1)
print("[validate] data OK")
