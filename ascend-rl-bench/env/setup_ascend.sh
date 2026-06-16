#!/usr/bin/env bash
# M0 — environment setup on Ascend 910B (Atlas 800T A2).
# Assumes the CANN toolkit + kernels are already installed at /usr/local/Ascend
# (driver/firmware come with the box). See:
#   https://www.hiascend.com/  and  https://github.com/vllm-project/vllm-ascend
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 0) source CANN env (path may differ per install)
CANN_ENV="${CANN_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}"
if [ -f "$CANN_ENV" ]; then
  # shellcheck disable=SC1090
  source "$CANN_ENV"
  echo "[setup-ascend] sourced CANN: $CANN_ENV"
else
  echo "[setup-ascend] WARN: $CANN_ENV not found — set CANN_ENV to your set_env.sh"
fi

echo "[setup-ascend] python: $(python3 --version)"
python3 -m pip install --upgrade pip

# 1) PyTorch (CPU/aarch64 build) + torch_npu matching your CANN version.
#    Check the support matrix before pinning: torch_npu must match torch + CANN.
python3 -m pip install "torch>=2.4"
python3 -m pip install torch_npu        # pin the version that matches your CANN release

# 2) Shared deps
python3 -m pip install -r "$HERE/requirements.txt"

# 3) Rollout engine on NPU + RL framework
python3 -m pip install vllm-ascend      # provides the Ascend backend for vLLM
python3 -m pip install "verl"           # or build from source: pip install -e verl

# quick visibility check
python3 - <<'PY'
try:
    import torch, torch_npu  # noqa
    print("[setup-ascend] torch_npu OK, npu available:", torch.npu.is_available(),
          "count:", torch.npu.device_count())
except Exception as e:  # noqa
    print("[setup-ascend] torch_npu import failed:", e)
PY

echo "[setup-ascend] done. Next: python3 env/check_env.py --model Qwen/Qwen2.5-0.5B-Instruct"
