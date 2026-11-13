# Model comparison maintenance

The Overview comparison uses `data/compare.json` (schema 2). This is a
curated, dated snapshot, not a live or exhaustive global leaderboard.

## Evidence boundaries

- Default: independent evaluation, currently Artificial Analysis v5.0.1.
- Vendor reports: separate by report publisher, benchmark version and harness.
  Baseline scores in a vendor's table remain vendor-reported evidence.
- History: `data/archive/compare-2026-10-26.json` preserves the previous file.
  Its scores, prices and specifications are not silently promoted to current data.
- Pending models: no placeholder score, no substitution from an older release.

Only independent groups with a declared common evaluation protocol are sorted.
This is not a claim of identical compute budgets. Model names preserve release
identifiers and reasoning effort is a separate field. A configuration is never
silently replaced with the model's best score on another benchmark.

## Refresh checklist

1. Read the evaluator's current leaderboard and methodology, not a vendor's
   quotation of an older index. Record the exact benchmark version.
2. Create a new group when evaluator, benchmark version or harness changes.
   Do not merge FrontierCode and FrontierCode-AA, or index v4.3/v5.0/v5.0.1.
3. Preserve the source's numerical precision and exact model/configuration.
   Unknown settings must remain explicitly unknown. Do not infer a model's
   access mode, open-weight status or hardware compatibility from its score.
4. Record `capturedAt` separately from `publishedAt` and `evaluationDate`.
   Use null for undisclosed dates; capture date is not the experiment date.
5. Provide HTTPS evidence for every row, group methodology, sample/repetition
   details when verified, and explicit budget gaps. Differing harnesses require
   `harness: "mixed"`, row-level harness names and `comparable: false`.
6. For missing results, use the pending list, or null with `missingReason`.
   A real zero is distinct from an absent result. Avoid unsupported significance
   claims and do not infer capability trends across index revisions.

## Verification

```sh
python3 scripts/validate_data.py
python3 -m unittest discover -s scripts -p 'test_*.py'
node --test scripts/test-compare.mjs
node --check js/app.js
node scripts/check-reads.mjs
git diff --check
```

Preview Overview at desktop and mobile sizes. Check both evidence views, every
metric selector, historical loading and keyboard navigation. Source dates and
settings must remain readable without truncated model names.
