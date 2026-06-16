# Hyperparameters for 0.5B GRPO on GSM8K (sourced by train/run_grpo.sh).
# Small + fast: good for getting the pipeline green and a first reward curve.

MODEL_PATH="Qwen/Qwen2.5-0.5B-Instruct"
EXP_NAME="qwen0.5b_gsm8k_grpo"

# data
DATA_DIR="data/gsm8k"
TRAIN_BATCH_SIZE=256
MAX_PROMPT_LEN=512
MAX_RESP_LEN=512

# GRPO / actor
LR=1e-6
PPO_MINI_BATCH=64
MICRO_BATCH=8
KL_COEF=0.001
ROLLOUT_N=8                 # samples per prompt (the "group" in GRPO)

# rollout engine
ROLLOUT_BACKEND="vllm"      # vllm on GPU; vllm via vllm-ascend on NPU
GPU_MEM_UTIL=0.6

# trainer
EPOCHS=1
SAVE_FREQ=20
TEST_FREQ=10
