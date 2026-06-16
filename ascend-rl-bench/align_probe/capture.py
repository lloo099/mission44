#!/usr/bin/env python3
"""align-probe / capture: run the training forward pass over FIXED rollouts and
dump the numeric quantities that drive GRPO, tagged with the device.

Run the identical command on GPU and on Ascend, then diff with compare.py.

Real:
    python3 align_probe/capture.py --model Qwen/Qwen2.5-0.5B-Instruct \
        --rollouts align_probe/rollouts.sample.jsonl --out captures/gpu.npz

Pipeline check without hardware:
    python3 align_probe/capture.py --synthetic --device gpu --out captures/gpu.npz
    python3 align_probe/capture.py --synthetic --device npu --out captures/npu.npz
"""
import argparse
import json
import os

import numpy as np

EPS = 1e-6


def grpo_advantages(rewards: np.ndarray, group_ids: np.ndarray) -> np.ndarray:
    """GRPO advantage = within-group standardized reward. Pure-numpy, identical
    on every platform, so any cross-device diff here is pure upstream drift."""
    adv = np.zeros_like(rewards, dtype=np.float64)
    for g in np.unique(group_ids):
        m = group_ids == g
        r = rewards[m]
        adv[m] = (r - r.mean()) / (r.std() + EPS)
    return adv


def load_rollouts(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ----------------------------------------------------------------- synthetic
def capture_synthetic(device, n_tokens=64, n_roll=8, seed=0):
    """Deterministic fake capture. Base numbers are identical across devices;
    `npu` gets a tiny injected perturbation to emulate numeric drift."""
    base = np.random.default_rng(seed)
    token_logprobs = -np.abs(base.normal(0.7, 0.3, size=(n_roll, n_tokens)))
    entropy = np.abs(base.normal(1.2, 0.2, size=(n_roll, n_tokens)))
    logits_summary = np.stack([base.normal(0, 5, n_roll),        # mean
                               base.normal(12, 1, n_roll)], 1)   # abs-max

    if device == "npu":
        drift = np.random.default_rng(seed + 1)
        token_logprobs = token_logprobs + drift.normal(0, 2e-3, token_logprobs.shape)
        entropy = entropy + drift.normal(0, 1e-3, entropy.shape)
        logits_summary = logits_summary + drift.normal(0, 5e-3, logits_summary.shape)

    rewards = np.array([1, 0, 1, 0, 1, 1, 0, 1][:n_roll], dtype=np.float64)
    group_ids = np.array([0, 0, 0, 0, 1, 1, 1, 1][:n_roll])
    return {
        "token_logprobs": token_logprobs,
        "seq_logprob": token_logprobs.sum(1),
        "entropy": entropy,
        "logits_summary": logits_summary,
        "advantages": grpo_advantages(rewards, group_ids),
    }


# ---------------------------------------------------------------------- real
def capture_real(model_id, rollouts, seed=0):
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(seed)
    np.random.seed(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:  # noqa: BLE001
        pass

    # device
    device = "cpu"
    try:
        import torch_npu  # noqa: F401
        if torch.npu.is_available():
            device = "npu"
    except Exception:  # noqa: BLE001
        pass
    if device == "cpu" and torch.cuda.is_available():
        device = "cuda"

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16 if device != "cpu" else torch.float32
    ).to(device).eval()

    tlp, ent, lsum = [], [], []
    for r in rollouts:
        msgs = [{"role": "user", "content": r["prompt"]}]
        prompt_ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
        resp_ids = tok(r["response"], return_tensors="pt", add_special_tokens=False).input_ids
        ids = torch.cat([prompt_ids, resp_ids], dim=1).to(device)
        with torch.no_grad():
            logits = model(ids).logits.float()           # [1, T, V]
        # log-probs of the response tokens (shifted)
        logp = F.log_softmax(logits[:, :-1], dim=-1)
        targets = ids[:, 1:]
        tok_logp = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)[0]
        start = prompt_ids.shape[1] - 1                   # response region
        resp_logp = tok_logp[start:]
        p = logp.exp()
        tok_ent = -(p * logp).sum(-1)[0][start:]
        tlp.append(resp_logp.cpu().numpy())
        ent.append(tok_ent.cpu().numpy())
        lsum.append([float(logits.mean()), float(logits.abs().max())])

    # pad ragged token arrays to a rectangle for storage
    width = max(len(x) for x in tlp)
    def pad(a):
        out = np.full((len(a), width), np.nan)
        for i, x in enumerate(a):
            out[i, : len(x)] = x
        return out

    rewards = np.array([r["reward"] for r in rollouts], dtype=np.float64)
    group_ids = np.array([r.get("group_id", 0) for r in rollouts])
    token_logprobs = pad(tlp)
    return device, {
        "token_logprobs": token_logprobs,
        "seq_logprob": np.array([np.nansum(x) for x in tlp]),
        "entropy": pad(ent),
        "logits_summary": np.array(lsum),
        "advantages": grpo_advantages(rewards, group_ids),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--rollouts", default="align_probe/rollouts.sample.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--device", default=None, help="tag for synthetic mode: gpu|npu")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)

    if args.synthetic:
        device = args.device or "gpu"
        arrays = capture_synthetic(device, seed=args.seed)
        model_id = "synthetic"
    else:
        rollouts = load_rollouts(args.rollouts)
        device, arrays = capture_real(args.model, rollouts, seed=args.seed)
        model_id = args.model

    meta = json.dumps({"device": device, "model": model_id, "seed": args.seed,
                       "n_rollouts": int(arrays["seq_logprob"].shape[0])})
    np.savez(args.out, _meta=np.array(meta), **arrays)
    print(f"[capture] device={device} model={model_id} -> {args.out}")
    print(f"[capture] arrays: {', '.join(f'{k}{v.shape}' for k, v in arrays.items())}")


if __name__ == "__main__":
    main()
