#!/usr/bin/env bash
# M0 — environment setup on NVIDIA GPU (baseline + cross-check platform).
# Run on a machine with CUDA drivers already installed.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[setup-gpu] python: $(python3 --version)"
python3 -m pip install --upgrade pip

# 1) PyTorch (CUDA build). Adjust the index URL to your CUDA version if needed.
python3 -m pip install "torch>=2.4" --index-url https://download.pytorch.org/whl/cu121

# 2) Shared deps
python3 -m pip install -r "$HERE/requirements.txt"

# 3) Rollout engine + RL framework
python3 -m pip install "vllm>=0.6.3"
# verl from source (recommended — APIs move fast):
python3 -m pip install "verl"      # or: git clone https://github.com/volcengine/verl && pip install -e verl

echo "[setup-gpu] done. Next: python3 env/check_env.py --model Qwen/Qwen2.5-0.5B-Instruct"
