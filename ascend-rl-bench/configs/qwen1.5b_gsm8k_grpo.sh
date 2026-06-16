# Hyperparameters for 1.5B GRPO on GSM8K (sourced by train/run_grpo.sh).
# Same recipe as 0.5B, scaled batch/memory. Expect a clearer reward curve.

MODEL_PATH="Qwen/Qwen2.5-1.5B-Instruct"
EXP_NAME="qwen1.5b_gsm8k_grpo"

# data
DATA_DIR="data/gsm8k"
TRAIN_BATCH_SIZE=256
MAX_PROMPT_LEN=512
MAX_RESP_LEN=1024

# GRPO / actor
LR=1e-6
PPO_MINI_BATCH=64
MICRO_BATCH=4
KL_COEF=0.001
ROLLOUT_N=8

# rollout engine
ROLLOUT_BACKEND="vllm"
GPU_MEM_UTIL=0.6

# trainer
EPOCHS=1
SAVE_FREQ=20
TEST_FREQ=10
