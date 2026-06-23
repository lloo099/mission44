# Dispatch 02 · Rollout 是 RL 的瓶颈:FP8 量化与异步稳定化的双线突破

*2026-06-23 · NPU Frontier Dispatch · RL-systems / FP8 / async-RL / Ascend*

> **TL;DR** — 在 LLM 的 RL 后训练里,**rollout(生成样本)吃掉了超过 70% 的训练时间**。2026 上半年,一大批工作从两条线同时压这个瓶颈:**(1) 把 rollout 量化到 FP8/INT8**(Jet-RL、FP8-RL、Quantized Rollout,7B–32B 上吞吐 +20–80%),**(2) 异步解耦 + 稳定化**(GAC 梯度对齐、staleness 受限协调、周期性异步)。这两条线恰好都压在**昇腾的两个痛点**上:Ascend 950 的**原生 FP8** 和 910B 上**无 sleep-mode 的显存争用**。换句话说——RL 系统的研究风向,正在朝着对 NPU 最友好的方向走。

接上一期(Seed 2.1 与开源 1M 浪潮)。这期不看模型,看**训练系统**:RL 后训练这一年最热的工程主线是什么,以及它为什么对"在昇腾上做 RL"格外重要。

---

## 1 · 为什么所有人都在盯 rollout

一次 GRPO/RLVR 迭代分两段:**rollout**(用当前策略采样大量轨迹)和 **train**(用这些样本更新策略)。问题在于 rollout 是自回归生成,**慢、长尾、且占用大量显存**:

- **时间**:rollout 普遍占 RL 单步 **70% 以上**的 wall-clock——它才是真正的瓶颈,不是反向传播。
- **长尾**:同一 batch 里少数超长轨迹拖住整组(APRIL 等专门治这个)。
- **显存**:推理引擎的 KV cache + 权重要常驻;在 CUDA 上可以靠 vLLM 的 **sleep-mode** 在 train 阶段释放,**昇腾目前没有这个机制**(见 NPU 架构页的"RL 显存争用"视图)。

于是 2026 H1 的 RL 系统工作基本沿两条线展开。

## 2 · 线一:把 rollout 量化下去(FP8 / INT8)

核心直觉:rollout 是推理,**推理可以用低精度**。难点是 RL 里**策略每步都在变**——要反复量化、把权重同步进推理引擎,还得防止低精度 rollout 与高精度 trainer 之间的 **train-inference mismatch** 把训练带崩。

| 工作 | 机制 | 收益 / 定位 |
|---|---|---|
| **Jet-RL** | 统一训练与 rollout 的精度流,做**on-policy FP8** RL | 让 FP8 rollout 与 trainer 数值一致,消除 mismatch |
| **FP8-RL** | veRL 生态里的实用 FP8 栈(FSDP/Megatron + vLLM/SGLang) | 工程+算法手段稳住 FP8 RL 循环 |
| **Quantized Rollout** | 仅对 rollout 做 INT8/FP8 量化 | 7B/14B/32B 上吞吐 **+20–80%** |

> 这条线的意义:rollout 既然占 70% 时间,把它的精度砍一半、吞吐翻倍,就是对整条 RL 管线最直接的提速。而 **Ascend 950 原生支持 FP8/MXFP4**——这恰好是把这套量化 rollout 搬到 NPU 的硬件前提。

## 3 · 线二:异步解耦,再把它稳住

另一条线干脆**把 rollout 和 train 解耦**成异步流水,让生成器一直跑、训练器消费稍旧的样本——吞吐上去了,但样本"陈旧"(staleness)会让重要性权重出现重尾、把 GRPO/REINFORCE 带偏。所以 2026 H1 的重点从"做异步"转向了"**把异步做稳**":

- **GAC(Gradient Alignment Control)**:对齐异步产生的梯度方向,抑制不稳定。
- **Staleness-Constrained Rollout Coordination**:给样本陈旧度设上界,在高异步度下保持可控。
- **Periodic Asynchrony**:用"周期性"的近 on-policy 方式拿异步的吞吐、还原 on-policy 的稳定。
- **APRIL(Active Partial Rollouts)**:主动截断长尾生成,治 rollout 的长尾拖累。

(这些与看板 RL 标签下已有的 Stable Asynchrony、RollMux、AsyncFlow、ROLL Flash 是同一条系统脉络。)

## 4 · 这对 RL-on-NPU 意味着什么

把两条线叠在昇腾的现实约束上,结论很顺:

- **量化 rollout 直接缓解显存争用**。昇腾没有 sleep-mode、rollout 与 train 抢同一份 64GB HBM;**FP8/INT8 的 rollout 占用更小**,等于绕开了一部分"无法释放 KV cache"的痛。这是把 GPU 上的量化-rollout 配方移植到 NPU 的**最高性价比方向**。
- **Ascend 950 的 FP8 让"FP8 RL on Ascend"从设想变可行**。Jet-RL/FP8-RL 这类配方 + 950 原生 FP8,是一个**高新颖度、低人做过**的选题。
- **但要盯数值漂移**。FP8 rollout、以及自定义稀疏注意力(DSA/MSA/CSA)在 NPU 上重写后,train-inference mismatch 会更隐蔽——这正是看板里 **align-probe** 这个想法该干的活:在 NPU 上量化 train-inference 的 log-prob 偏差。
- **MindSpeed-RL 是现成底座**。华为已开源的 MindSpeed-RL(910B、384-NPU 上跑过 DeepSeek-R1-671B)用 vLLM-Ascend 做生成、MindSpeed 做训练——量化 rollout 与异步稳定化,正好可以接在它上面做实验。

## 5 · 下一步看什么

1. **FP8 rollout 会不会成为 RL 框架的默认项**(veRL/ROLL/MindSpeed-RL 是否内置)。
2. **异步稳定化的"够用阈值"**:staleness 上界 / 周期性异步,究竟容忍多旧的样本还不掉点。
3. **有没有人在昇腾 950 上跑通端到端 FP8 RL** 并公布 train-inference 一致性数据——这会是 RL-on-NPU 最有说服力的一块拼图。

---

*来源:arXiv 上 2026 H1 的 RL 系统工作(Jet-RL 2601.14243、FP8-RL 2601.18150、Quantized Rollout 2602.13953、GAC 2603.01501、Staleness-Constrained Coordination 2601.12784、Periodic Asynchrony 2511.18871、APRIL 2509.18521)与 MindSpeed-RL(2507.19017);数字为论文自报,provisional。*
