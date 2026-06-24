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


def parse_log(path):
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
    return {m: pts for m, pts in series.items() if pts}


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


def build_meta(args, synthetic_run):
    """Assemble experiment metadata from CLI flags (only keys that were given)."""
    meta = {}
    for k in ("model", "dataset", "hardware", "framework", "precision"):
        v = getattr(args, k, None)
        if v:
            meta[k] = v
    if getattr(args, "seed", None) is not None:
        meta["seed"] = args.seed
    meta["synthetic"] = bool(synthetic_run)
    return meta


def load_existing(out):
    try:
        with open(out, encoding="utf-8") as f:
            return json.load(f).get("experiments", [])
    except (FileNotFoundError, json.JSONDecodeError):
        return []


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
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    if args.synthetic:
        new = synthetic()
    elif args.log:
        metrics = parse_log(args.log)
        if not metrics:
            print(f"[curves] no metrics parsed from {args.log} — check key names in ALIASES")
        new = [{"name": args.name, "device": args.device, "metrics": metrics,
                "meta": build_meta(args, synthetic_run=False)}]
    else:
        ap.error("need --log <file> or --synthetic")

    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    experiments = merge(load_existing(out), new)
    payload = {
        "updated": dt.datetime.utcnow().isoformat() + "Z",
        "synthetic": any((e.get("meta") or {}).get("synthetic") for e in experiments),
        "experiments": experiments,
    }
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    total = sum(len(next(iter(e["metrics"].values()), [])) for e in experiments)
    print(f"[curves] wrote {len(experiments)} experiment(s), ~{total} points -> {os.path.relpath(out)}")


if __name__ == "__main__":
    main()
