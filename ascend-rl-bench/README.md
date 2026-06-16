# AscendRL-Bench — M0/M1 scaffold

Minimal, runnable scaffold for the first two milestones of the
[project plan](../PLAN.md):

- **M0 — environment**: install the stack (CANN + torch_npu + vLLM-Ascend + verl on NPU,
  or CUDA + vLLM + verl on GPU) and pass a smoke test.
- **M1 — single-platform baseline**: run GRPO on `Qwen2.5-0.5B/1.5B-Instruct` over GSM8K
  with verifiable rewards and get a first reward curve.

> ⚠️ This is a **scaffold**, not a finished experiment. It is written against the public
> verl / vLLM-Ascend docs and is meant to be run on a machine that actually has GPUs or
> 910B NPUs. Exact verl flags shift between releases — pin a version and check
> [verl docs](https://verl.readthedocs.io/) if an override is renamed.

## Layout

```
ascend-rl-bench/
├── env/
│   ├── setup_ascend.sh     # install stack on Ascend 910B (CANN/torch_npu/vllm-ascend/verl)
│   ├── setup_gpu.sh        # install stack on NVIDIA (CUDA/vllm/verl) for the baseline/cross-check
│   ├── requirements.txt    # shared python deps
│   └── check_env.py        # SMOKE TEST: device count + a tiny generation
├── data/
│   └── prepare_gsm8k.py    # download GSM8K -> verl parquet (train/test)
├── rewards/
│   └── gsm8k_verifier.py   # verifiable reward (answer extraction + exact match); verl-compatible
├── configs/
│   ├── qwen0.5b_gsm8k_grpo.sh   # hyperparameters (sourced by run_grpo.sh)
│   └── qwen1.5b_gsm8k_grpo.sh
├── train/
│   └── run_grpo.sh         # one command to launch GRPO; DEVICE=gpu|npu switch
├── eval/
│   └── eval_gsm8k.py       # post-training pass@1 on the test set
└── logs/                   # training curves / outputs land here
```

## Quick start

```bash
# 0) install (pick your hardware)
bash env/setup_gpu.sh        # or: bash env/setup_ascend.sh

# 1) smoke test — confirms devices are visible and a model can generate
python3 env/check_env.py --model Qwen/Qwen2.5-0.5B-Instruct

# 2) prepare data (writes data/gsm8k/{train,test}.parquet)
python3 data/prepare_gsm8k.py --out data/gsm8k

# 3) sanity-check the reward function (no GPU needed)
python3 rewards/gsm8k_verifier.py --selftest

# 4) train — 0.5B GRPO on GSM8K
DEVICE=gpu CONFIG=configs/qwen0.5b_gsm8k_grpo.sh bash train/run_grpo.sh
#   on Ascend:  DEVICE=npu CONFIG=configs/qwen0.5b_gsm8k_grpo.sh bash train/run_grpo.sh

# 5) evaluate the trained checkpoint
python3 eval/eval_gsm8k.py --model <path-to-checkpoint> --data data/gsm8k/test.parquet
```

## What to expect (so you can tell "working" from "broken")

- **check_env.py**: prints `device backend: cuda|npu`, a device count > 0, and one short
  generated completion. If it prints `cpu`, the accelerator libs aren't picked up.
- **prepare_gsm8k.py**: writes ~7.47k train / ~1.32k test rows; each row has a chat-format
  `prompt` and `reward_model.ground_truth`.
- **run_grpo.sh**: verl logs `step`, `critic/rewards/mean`, `actor/kl`, `response_length`.
  On a 0.5B model the mean reward should start near chance and trend up within tens of steps.
- **logs/**: point the dashboard at these later (export to the `data/feed.json` JSON shape).

## GPU ↔ NPU switch

`run_grpo.sh` keeps one config and flips only the device-specific bits:

| | GPU | Ascend NPU |
|---|---|---|
| visible-device env | `CUDA_VISIBLE_DEVICES` | `ASCEND_RT_VISIBLE_DEVICES` |
| rollout backend | `vllm` | `vllm` (via **vllm-ascend** plugin) |
| install script | `setup_gpu.sh` | `setup_ascend.sh` |

This is exactly the cross-platform comparison M2 needs: validate the pipeline on GPU,
then flip `DEVICE=npu` to run the same config on 910B.
