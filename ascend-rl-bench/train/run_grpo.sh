#!/usr/bin/env bash
# M1 — launch GRPO on GSM8K with verl. One config, GPU/NPU switch.
#
#   DEVICE=gpu CONFIG=configs/qwen0.5b_gsm8k_grpo.sh bash train/run_grpo.sh
#   DEVICE=npu CONFIG=configs/qwen0.5b_gsm8k_grpo.sh bash train/run_grpo.sh
#
# Override anything inline, e.g.  N_DEVICES=4 DEVICE=gpu CONFIG=... bash train/run_grpo.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DEVICE="${DEVICE:-gpu}"
CONFIG="${CONFIG:-configs/qwen0.5b_gsm8k_grpo.sh}"
N_DEVICES="${N_DEVICES:-8}"
REWARD_PATH="${REWARD_PATH:-rewards/gsm8k_verifier.py}"
OUTPUT_DIR="${OUTPUT_DIR:-logs/$(basename "${CONFIG%.sh}")-${DEVICE}}"

# shellcheck disable=SC1090
source "$CONFIG"
mkdir -p "$OUTPUT_DIR"

# --- device-specific bits --------------------------------------------------
if [ "$DEVICE" = "npu" ]; then
  export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-$(seq -s, 0 $((N_DEVICES-1)))}"
  echo "[train] NPU run — ASCEND_RT_VISIBLE_DEVICES=$ASCEND_RT_VISIBLE_DEVICES"
  echo "[train] (ensure vllm-ascend + torch_npu are installed; verl auto-detects npu)"
else
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$(seq -s, 0 $((N_DEVICES-1)))}"
  echo "[train] GPU run — CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
fi

echo "[train] config=$CONFIG model=$MODEL_PATH n_devices=$N_DEVICES out=$OUTPUT_DIR"

# --- verl GRPO -------------------------------------------------------------
# NOTE: flags follow verl's GRPO example; if a key was renamed in your verl
# version, check `python3 -m verl.trainer.main_ppo --help` / the docs.
python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files="$DATA_DIR/train.parquet" \
  data.val_files="$DATA_DIR/test.parquet" \
  data.train_batch_size="$TRAIN_BATCH_SIZE" \
  data.max_prompt_length="$MAX_PROMPT_LEN" \
  data.max_response_length="$MAX_RESP_LEN" \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.actor.optim.lr="$LR" \
  actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI_BATCH" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$MICRO_BATCH" \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef="$KL_COEF" \
  actor_rollout_ref.rollout.name="$ROLLOUT_BACKEND" \
  actor_rollout_ref.rollout.gpu_memory_utilization="$GPU_MEM_UTIL" \
  actor_rollout_ref.rollout.n="$ROLLOUT_N" \
  custom_reward_function.path="$REWARD_PATH" \
  custom_reward_function.name=compute_score \
  trainer.n_gpus_per_node="$N_DEVICES" \
  trainer.nnodes=1 \
  trainer.total_epochs="$EPOCHS" \
  trainer.save_freq="$SAVE_FREQ" \
  trainer.test_freq="$TEST_FREQ" \
  trainer.project_name=ascend-rl-bench \
  trainer.experiment_name="${EXP_NAME}-${DEVICE}" \
  trainer.logger=['console'] \
  trainer.default_local_dir="$OUTPUT_DIR" \
  2>&1 | tee "$OUTPUT_DIR/train.log"

echo "[train] done -> $OUTPUT_DIR (see train.log for the reward curve)"
