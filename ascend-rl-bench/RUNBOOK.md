# RUNBOOK — 一次可复现的 910B 实验

这份文档是**上机时照着走的流程**。`README.md` 说明脚手架有什么,这里说明怎么跑出一条**别人能核验**的曲线。

判据只有一条:**任何发布到看板的曲线,都必须能从原始日志重建**。为此每次运行落三个文件——`train.log`(原始)、`env.json`(机器与软件栈)、`run.json`(超参),缺一个,`scripts/validate_data.py` 就会拒绝。

---

## 0 · 前置

| 项 | 要求 |
|---|---|
| 硬件 | Atlas 800T A2(910B)一节点,或任意 NVIDIA 节点做对照 |
| 栈 | CANN + torch_npu + vLLM-Ascend + verl(NPU);CUDA + vLLM + verl(GPU) |
| 磁盘 | 模型与 checkpoint 约 20 GB 起 |
| 时间 | 0.5B / 200 步,8 卡约 1 至 2 小时(首次含下载) |

```bash
cd ascend-rl-bench
bash env/setup_ascend.sh      # 或 env/setup_gpu.sh
```

---

## 1 · 冒烟:先证明栈是通的

```bash
python3 env/check_env.py --model Qwen/Qwen2.5-0.5B-Instruct
```

**期望**:`device backend: npu`、`device count: 8`、一段短补全。

**若打印 `cpu` 或 `none`**:torch_npu 没装上或 `ASCEND_RT_VISIBLE_DEVICES` 没生效,先解决这个,不要往下走——后面每一步都会以更难诊断的方式失败。

---

## 2 · 数据与奖励

```bash
python3 data/prepare_gsm8k.py --out data/gsm8k
python3 rewards/gsm8k_verifier.py --selftest      # 不需要加速卡
```

**期望**:约 7.47k 训练 / 1.32k 测试行;自检全绿。奖励函数错了,后面的曲线只是噪声,所以这一步必过。

---

## 3 · 训练

```bash
DEVICE=npu N_DEVICES=8 CONFIG=configs/qwen0.5b_gsm8k_grpo.sh bash train/run_grpo.sh
```

脚本在开训**之前**先写 `logs/<exp>-npu/env.json` 与 `run.json`。这是刻意的:事后凭记忆补的 provenance 没有价值。

**运行中要看什么**

| 现象 | 含义 |
|---|---|
| `critic/rewards/mean` 数十步内从接近随机开始上行 | 正常 |
| reward 卡在 0 不动 | 奖励函数没接上,或答案抽取失败 —— 回第 2 步 |
| reward 迅速冲到 1.0 | 多半是奖励被钻空子,查几条 rollout 原文 |
| `actor/kl_loss` 单调爆涨 | KL 系数太小,策略跑飞 |
| OOM | 降 `MICRO_BATCH` 或 `GPU_MEM_UTIL`;910B 上这是最常见的失败(见 D05 引用的 verl #2552) |

**种子**。`configs/*.sh` 里的 `SEED=42` 接到 verl 的两个真实键:`data.seed`(数据打散)与 `actor_rollout_ref.rollout.seed`(vLLM 采样)。要完全确定性还需 `actor_rollout_ref.rollout.full_determinism=true`,但会掉吞吐——**默认不开,因为本实验比较的是曲线形状而非逐 token 复现**。换独立运行时改 `SEED` 即可。

> ⚠️ **verl 的参数名会随版本漂移。** 如果某个 override 报 `unrecognized key`,用 `python3 -m verl.trainer.main_ppo --help` 对一下,并把版本记进 `run.json`。`env.json` 里已自动记录 `verl` 的包版本。

---

## 4 · 发布曲线

```bash
python3 tools/logs_to_dashboard.py \
  --log logs/qwen0.5b_gsm8k_grpo-npu/train.log \
  --env logs/qwen0.5b_gsm8k_grpo-npu/env.json \
  --name qwen0.5b_gsm8k_grpo --device npu \
  --model Qwen2.5-0.5B-Instruct --dataset GSM8K --precision bf16 --seed 42

python3 ../scripts/validate_data.py        # 必须通过才能提交
```

`--env` 会把**实测的** `hardware` 与 `framework` 填进去(例如 `8x Ascend910B4`、`verl + vLLM-Ascend`),而不是手敲。手敲的字符串会随时间和机器漂移,这是这套流程要防的第一件事。

工具还会写一个 `source` 块:日志相对路径、**日志的 sha256**、步数与步区间、解析时间,以及完整的 `env.json`。校验器对 `synthetic: false` 的运行**强制要求**这个块——没有它,一条"实测"曲线只是一个形状。

日志格式:`--format auto`(默认)优先按 JSONL 解析,失败回落到 console 正则。**能出 JSONL 就出 JSONL**,它不会随日志格式微调而失效。

**把证据一起提交。** `logs/` 默认只放行三类文件:

```bash
gzip -k logs/qwen0.5b_gsm8k_grpo-npu/train.log      # 压缩后才会被 git 跟踪
git add ascend-rl-bench/logs/qwen0.5b_gsm8k_grpo-npu/{env.json,run.json,train.log.gz}
```

原始 `train.log` 与 checkpoint 仍被忽略(体积);`env.json`、`run.json`、`train.log.gz` 会进仓库——**否则 `source.logSha256` 指向的东西别人拿不到,校验也就没有意义**。

---

## 5 · 配对对照(这才是结论所在)

单跑一条 NPU 曲线说明不了什么。有价值的是**同配置、同种子、只换设备**:

```bash
DEVICE=gpu N_DEVICES=8 CONFIG=configs/qwen0.5b_gsm8k_grpo.sh bash train/run_grpo.sh
python3 tools/logs_to_dashboard.py --log logs/qwen0.5b_gsm8k_grpo-gpu/train.log \
  --env logs/qwen0.5b_gsm8k_grpo-gpu/env.json \
  --name qwen0.5b_gsm8k_grpo --device gpu \
  --model Qwen2.5-0.5B-Instruct --dataset GSM8K --precision bf16 --seed 42
```

两条同名不同 `device` 的运行会画在同一坐标系上。**只有当 `model` / `dataset` / `precision` / `seed` 四项完全一致时,这个对比才成立**——曲线分叉才能归因到硬件与框架,否则归因不了。

值得记录的差异量:收敛到同一 reward 所需步数、每步墙钟时间、峰值 HBM、以及 rollout 与 update 的时间占比。

---

## 6 · 发布检查表

- [ ] `train.log`、`env.json`、`run.json` 三件齐全,且 `env.json` 的 `hardware` 不是 `null`
- [ ] `python3 ../scripts/validate_data.py` 通过
- [ ] `data/curves.json` 的 `synthetic` 变成 `false`,且 `publishedEvidence` **仍在**(工具会保留,校验器会拦截丢失)
- [ ] 看板的 **Training Curves** 标签页自动出现(它只在存在实测运行时显示)
- [ ] `env.json` / `run.json` / `train.log.gz` 三个证据文件已 `git add`(其余 `logs/` 内容被忽略)
- [ ] `git status` 里没有把 checkpoint 或数据集误加进来
- [ ] 曲线异常处在 dispatch 里写清楚,而不是只贴图

---

## 已知边界

- **`full_determinism` 默认关闭**,所以同种子重跑的曲线会有微小抖动。要主张"逐点可复现"必须先打开它并重跑。
- **`env.json` 在无 torch 的机器上会退化**(`hardware: null`),此时校验器会因 `meta` 缺字段而拒绝——这是有意的。
- **verl 的 flag 漂移**是这套脚本最脆的地方,已在第 3 步标注。
- **本仓库尚无任何实测 910B 运行**;`data/curves.json` 当前是标注为合成的演示数据,曲线页因此保持隐藏。
