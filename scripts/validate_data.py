#!/usr/bin/env python3
"""Validate data/*.json for the dashboard. Run in CI to block broken commits.
Only files that have an "items" list are item-validated (compare/curves skipped)."""
import glob, json, sys

OK_CONF = {"confirmed", "secondary", "self-reported", "确证", "二手", "自报", None}
CURVE_REQUIRED = {"name", "device", "metrics"}
CURVE_META_HINTS = {"model", "dataset", "framework", "hardware", "precision", "seed", "runType"}
errs = []
for f in sorted(glob.glob("data/*.json")):
    try:
        d = json.load(open(f, encoding="utf-8"))
    except Exception as e:
        errs.append(f"{f}: invalid JSON: {e}"); continue
    if f.endswith("curves.json"):
        exps = d.get("experiments", []) if isinstance(d, dict) else []
        if not isinstance(exps, list):
            errs.append(f"{f}: 'experiments' is not a list"); continue
        for i, it in enumerate(exps):
            if not isinstance(it, dict):
                errs.append(f"{f}[{i}]: experiment is not an object"); continue
            missing = sorted(k for k in CURVE_REQUIRED if k not in it)
            if missing:
                errs.append(f"{f}[{i}] ({it.get('name','?')}): missing {', '.join(missing)}")
            if not any(k in it for k in CURVE_META_HINTS):
                errs.append(f"{f}[{i}] ({it.get('name','?')}): add run metadata such as model/dataset/hardware/seed")
            if not isinstance(it.get("metrics", {}), dict):
                errs.append(f"{f}[{i}] ({it.get('name','?')}): metrics is not an object")
        continue
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

print(f"[validate] checked {len(glob.glob('data/*.json'))} files")
if errs:
    print("VALIDATION ERRORS:")
    for e in errs:
        print("  -", e)
    sys.exit(1)
print("[validate] data OK")
