# align-probe

The project's core contribution: **pin everything that can be pinned, then measure
where the numbers diverge** between NVIDIA and Ascend.

The idea: rollout generation is stochastic and hard to reproduce across platforms, so
we **fix the rollouts** (same prompts + same response token ids + same rewards) and only
compare the **training forward pass**. That isolates pure numerical drift (fused
attention, mixed-precision accumulation, RNG, optimizer) from sampling differences.

## Workflow

```bash
# on the GPU box
python3 align_probe/capture.py --model Qwen/Qwen2.5-0.5B-Instruct \
    --rollouts align_probe/rollouts.sample.jsonl --out captures/gpu.npz

# on the Ascend box (identical command, different hardware)
python3 align_probe/capture.py --model Qwen/Qwen2.5-0.5B-Instruct \
    --rollouts align_probe/rollouts.sample.jsonl --out captures/npu.npz

# anywhere: diff the two captures
python3 align_probe/compare.py captures/gpu.npz captures/npu.npz \
    --out data/align_drift.json
```

`compare.py` reports, per quantity, **MAE / max-abs-diff / mean relative error / cosine
similarity** — so you can say *"log-probs drift by 3e-3 MAE, advantages by 1e-2, and the
divergence concentrates in the attention block"* instead of hand-waving.

## What gets captured (per fixed rollout)

| array | why it matters for GRPO |
|---|---|
| `logits_summary` | mean / abs-max of the full logits (cheap proxy for raw forward drift) |
| `token_logprobs` | per-token log-prob of the response — the policy term in the loss |
| `seq_logprob`    | sequence log-prob sum — what importance ratios are built from |
| `entropy`        | per-token entropy — exploration / collapse signal |
| `advantages`     | GRPO group-normalized advantages from the **fixed** rewards |

`advantages` use only the pinned rewards + group ids, so any cross-platform difference
there comes purely from upstream numeric drift, not from different sampling.

## No hardware? Validate the pipeline

```bash
# fabricate two deterministic captures (npu = gpu + tiny injected drift)
python3 align_probe/capture.py --synthetic --device gpu --out captures/gpu.npz
python3 align_probe/capture.py --synthetic --device npu --out captures/npu.npz
python3 align_probe/compare.py captures/gpu.npz captures/npu.npz --out data/align_drift.json
python3 align_probe/compare.py --selftest
```
