#!/usr/bin/env python3
"""Turn verl training logs into data/curves.json so the dashboard can plot
reward / KL / entropy / response-length curves.

Parse a real run:
    python3 tools/logs_to_dashboard.py \
        --log logs/qwen0.5b_gsm8k_grpo-gpu/train.log \
        --name qwen0.5b_gsm8k_grpo --device gpu

Generate a demo curve set (no run needed) so the chart has something to show:
    python3 tools/logs_to_dashboard.py --synthetic

Both modes MERGE into data/curves.json by (name, device), so you can add GPU and
NPU runs incrementally and compare them on the same axes.
"""
import argparse
import datetime as dt
import hashlib
import json
import math
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(HERE, "..", "..", "data", "curves.json")

# metric -> list of verl key aliases to look for in a log line
ALIASES = {
    "reward_mean": ["critic/rewards/mean", "reward/mean", "critic/score/mean", "val/reward", "reward"],
    "kl": ["actor/kl_loss", "actor/kl", "kl"],
    "entropy": ["actor/entropy", "actor/entropy_loss", "entropy"],
    "response_length": ["response_length/mean", "response/length/mean", "response_length"],
}
STEP_RE = re.compile(r"(?:global_)?step['\":\s]+(\d+)")


def _num_after(line, key):
    # match  key:0.12  key=0.12  'key': 0.12
    m = re.search(re.escape(key) + r"['\"]?\s*[:=]\s*(-?\d+\.?\d*(?:[eE][-+]?\d+)?)", line)
    return float(m.group(1)) if m else None


def _dedupe(series):
    """Keep one point per step (the last seen), sorted — verl can log a metric
    more than once per step and the chart should not show a zigzag for that."""
    out = {}
    for metric, pts in series.items():
        by_step = {}
        for step, val in pts:
            by_step[step] = val
        if by_step:
            out[metric] = [[s, by_step[s]] for s in sorted(by_step)]
    return out


def parse_console(path):
    """verl/MindSpeed-RL console output: 'step:12 - critic/rewards/mean:0.43 …'."""
    series = {m: [] for m in ALIASES}
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            sm = STEP_RE.search(line)
            if not sm:
                continue
            step = int(sm.group(1))
            for metric, keys in ALIASES.items():
                for k in keys:
                    v = _num_after(line, k)
                    if v is not None:
                        series[metric].append([step, v])
                        break
    return _dedupe(series)


def parse_jsonl(path):
    """One JSON object per line ({"step": 12, "critic/rewards/mean": 0.43, …}).

    Preferred over console scraping: no regex ambiguity, and it survives logger
    format changes between framework releases.
    """
    series = {m: [] for m in ALIASES}
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            step = rec.get("step", rec.get("global_step"))
            if step is None:
                continue
            for metric, keys in ALIASES.items():
                for k in keys:
                    if isinstance(rec.get(k), (int, float)):
                        series[metric].append([int(step), float(rec[k])])
                        break
    return _dedupe(series)


def parse_log(path, fmt="auto"):
    if fmt == "jsonl":
        return parse_jsonl(path)
    if fmt == "console":
        return parse_console(path)
    # auto: prefer jsonl when the file looks like it, fall back to console
    jsonl = parse_jsonl(path)
    return jsonl if jsonl else parse_console(path)


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_env(path):
    """Provenance record from `check_env.py --json` (see env/check_env.py)."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


SYNTH_META = {
    "gpu": {"model": "Qwen2.5-0.5B-Instruct", "dataset": "GSM8K", "hardware": "1× NVIDIA A100 80GB",
            "framework": "verl + vLLM", "precision": "bf16", "seed": 1, "synthetic": True},
    "npu": {"model": "Qwen2.5-0.5B-Instruct", "dataset": "GSM8K", "hardware": "1× Ascend 910B 64GB",
            "framework": "MindSpeed-RL + vLLM-Ascend", "precision": "bf16", "seed": 2, "synthetic": True},
}


def synthetic(name="qwen0.5b_gsm8k_grpo"):
    """Two plausible runs (gpu baseline + npu with slight drift)."""
    import random
    exps = []
    for device, seed, scale in (("gpu", 1, 1.0), ("npu", 2, 0.97)):
        rng = random.Random(seed)
        metrics = {"reward_mean": [], "kl": [], "entropy": [], "response_length": []}
        for step in range(0, 201, 10):
            t = step / 200
            reward = (0.08 + 0.78 * (1 / (1 + math.exp(-6 * (t - 0.45))))) * scale
            reward += rng.uniform(-0.02, 0.02)
            metrics["reward_mean"].append([step, round(reward, 4)])
            metrics["kl"].append([step, round(0.0005 + 0.004 * t + rng.uniform(0, 5e-4), 5)])
            metrics["entropy"].append([step, round(1.25 - 0.5 * t + rng.uniform(-0.03, 0.03), 4)])
            metrics["response_length"].append([step, round(120 + 80 * t + rng.uniform(-6, 6), 1)])
        exps.append({"name": name, "device": device, "metrics": metrics, "meta": dict(SYNTH_META[device])})
    return exps


def build_meta(args, synthetic_run, env=None):
    """Experiment metadata: measured values from the env record where available,
    with explicit CLI flags taking precedence."""
    meta = {}
    if env:
        for k in ("hardware", "framework"):
            if env.get(k):
                meta[k] = env[k]
    for k in ("model", "dataset", "hardware", "framework", "precision"):
        v = getattr(args, k, None)
        if v:
            meta[k] = v
    if getattr(args, "seed", None) is not None:
        meta["seed"] = args.seed
    meta["synthetic"] = bool(synthetic_run)
    return meta


def load_existing(out):
    """Return the whole payload, not just experiments.

    Earlier this returned only `experiments`, and main() then rebuilt the file
    from scratch — silently deleting `publishedEvidence` (the cited real-world
    runs) on every write. Keep the full document and merge into it.
    """
    try:
        with open(out, encoding="utf-8") as f:
            payload = json.load(f)
        return payload if isinstance(payload, dict) else {}
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def merge(existing, new):
    by_key = {(e["name"], e["device"]): e for e in existing}
    for e in new:
        by_key[(e["name"], e["device"])] = e
    return list(by_key.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", help="verl train.log to parse")
    ap.add_argument("--name", default="run", help="experiment name")
    ap.add_argument("--device", default="gpu", help="gpu|npu (legend grouping)")
    ap.add_argument("--synthetic", action="store_true", help="emit a demo curve set")
    # experiment metadata (recorded so the dashboard can show run provenance)
    ap.add_argument("--model", help="model id, e.g. Qwen2.5-0.5B-Instruct")
    ap.add_argument("--dataset", help="dataset, e.g. GSM8K")
    ap.add_argument("--hardware", help="hardware, e.g. '1× Ascend 910B 64GB'")
    ap.add_argument("--framework", help="framework, e.g. 'MindSpeed-RL + vLLM-Ascend'")
    ap.add_argument("--precision", help="precision, e.g. bf16 / fp8")
    ap.add_argument("--seed", type=int, help="random seed")
    ap.add_argument("--env", help="provenance record from `check_env.py --json` "
                                  "(fills hardware/framework from the actual host)")
    ap.add_argument("--format", choices=("auto", "jsonl", "console"), default="auto",
                    help="log format (default: auto-detect)")
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    if args.synthetic:
        new = synthetic()
    elif args.log:
        metrics = parse_log(args.log, args.format)
        if not metrics:
            print(f"[curves] no metrics parsed from {args.log} — check key names in ALIASES")
        env = load_env(args.env) if args.env else None
        meta = build_meta(args, synthetic_run=False, env=env)
        missing = [k for k in ("model", "dataset", "hardware", "framework", "precision", "seed")
                   if meta.get(k) in (None, "")]
        if missing:
            print(f"[curves] WARN: meta missing {missing} — validate_data.py will reject this. "
                  f"Pass --env and/or the matching flags.")
        steps = sorted({s for pts in metrics.values() for s, _ in pts})
        # An unreproducible curve is worse than no curve: anchor it to the exact
        # log bytes, the step range, and the host it came off.
        source = {
            "logPath": os.path.relpath(os.path.abspath(args.log), os.path.dirname(os.path.abspath(args.out))),
            "logSha256": sha256_of(args.log),
            "steps": len(steps),
            "stepRange": [steps[0], steps[-1]] if steps else None,
            "parsedAt": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        }
        if env:
            source["env"] = env
        new = [{"name": args.name, "device": args.device, "metrics": metrics,
                "meta": meta, "source": source}]
    else:
        ap.error("need --log <file> or --synthetic")

    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)

    # Merge into the existing document so sibling blocks (publishedEvidence and
    # anything added later) survive.
    payload = load_existing(out)
    experiments = merge(payload.get("experiments", []), new)
    payload["experiments"] = experiments
    payload["updated"] = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    payload["synthetic"] = any((e.get("meta") or {}).get("synthetic") for e in experiments)
    payload.setdefault("publishedEvidence", [])

    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")
    total = sum(len(next(iter(e["metrics"].values()), [])) for e in experiments)
    kept = len(payload.get("publishedEvidence", []))
    print(f"[curves] wrote {len(experiments)} experiment(s), ~{total} points "
          f"({kept} publishedEvidence entr{'y' if kept == 1 else 'ies'} preserved) -> {os.path.relpath(out)}")


if __name__ == "__main__":
    main()
