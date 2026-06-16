#!/usr/bin/env python3
"""align-probe / compare: diff two captures (e.g. GPU vs Ascend) and report where
the training-forward numbers drift.

    python3 align_probe/compare.py captures/gpu.npz captures/npu.npz --out data/align_drift.json
    python3 align_probe/compare.py --selftest
"""
import argparse
import datetime as dt
import json
import sys

import numpy as np


def _metrics(a: np.ndarray, b: np.ndarray) -> dict:
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]
    if a.size == 0:
        return {"n": 0}
    diff = np.abs(a - b)
    denom = np.maximum(np.abs(a), 1e-12)
    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    return {
        "n": int(a.size),
        "mae": float(diff.mean()),
        "max_abs_diff": float(diff.max()),
        "mean_rel_err": float((diff / denom).mean()),
        "cosine": cos,
    }


def _load(path):
    z = np.load(path, allow_pickle=True)
    meta = json.loads(str(z["_meta"])) if "_meta" in z else {}
    arrays = {k: z[k] for k in z.files if k != "_meta"}
    return meta, arrays


def compare(path_a, path_b):
    meta_a, arr_a = _load(path_a)
    meta_b, arr_b = _load(path_b)
    keys = [k for k in arr_a if k in arr_b]
    report = {k: _metrics(arr_a[k], arr_b[k]) for k in keys}
    return meta_a, meta_b, report


def _print(meta_a, meta_b, report):
    print(f"A = {meta_a.get('device','?')} ({meta_a.get('model','?')})")
    print(f"B = {meta_b.get('device','?')} ({meta_b.get('model','?')})")
    print(f"{'quantity':<16}{'MAE':>12}{'max|Δ|':>12}{'rel.err':>12}{'cosine':>10}")
    print("-" * 62)
    for k, m in report.items():
        if m.get("n", 0) == 0:
            print(f"{k:<16}{'(empty)':>46}")
            continue
        print(f"{k:<16}{m['mae']:>12.3e}{m['max_abs_diff']:>12.3e}"
              f"{m['mean_rel_err']:>12.3e}{m['cosine']:>10.5f}")
    # crude "where does it concentrate" hint
    ranked = sorted((m for m in report.values() if m.get("n")),
                    key=lambda m: m["mae"], reverse=True)
    if ranked:
        worst = max(report.items(), key=lambda kv: kv[1].get("mae", 0))
        print(f"\nlargest drift: {worst[0]} (MAE {worst[1]['mae']:.3e})")


def _selftest():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(4, 16))
    b = a + rng.normal(0, 1e-3, a.shape)
    m = _metrics(a, b)
    ok = m["mae"] < 5e-3 and m["cosine"] > 0.999 and m["n"] == 64
    # identical arrays => zero drift, cosine 1
    m0 = _metrics(a, a)
    ok = ok and m0["mae"] == 0.0 and abs(m0["cosine"] - 1.0) < 1e-9
    # nan handling
    a2 = np.array([1.0, np.nan, 3.0]); b2 = np.array([1.0, 5.0, 3.0])
    ok = ok and _metrics(a2, b2)["n"] == 2
    print("[selftest]", "PASS" if ok else "FAIL", json.dumps(m))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a", nargs="?", help="capture A (e.g. gpu.npz)")
    ap.add_argument("b", nargs="?", help="capture B (e.g. npu.npz)")
    ap.add_argument("--out", help="write JSON drift report here")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        sys.exit(_selftest())
    if not (args.a and args.b):
        ap.error("need two capture files (or --selftest)")

    meta_a, meta_b, report = compare(args.a, args.b)
    _print(meta_a, meta_b, report)

    if args.out:
        payload = {
            "updated": dt.datetime.utcnow().isoformat() + "Z",
            "a": meta_a, "b": meta_b, "report": report,
        }
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"\n[compare] wrote {args.out}")


if __name__ == "__main__":
    main()
