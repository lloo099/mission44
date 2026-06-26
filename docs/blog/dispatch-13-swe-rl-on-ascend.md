# Dispatch 13 · 在昇腾上搭一个 SWE Agentic RL 最小原型:方案设计

*2026-06-26 · NPU Frontier Dispatch · 方案设计 / SWE-RL / Ascend / 系统*

> **TL;DR** — 把 Dispatch 12 的 SWE agentic RL 配方落到昇腾上,做一个**单节点(Atlas 800T A2,8× 910B)、7B 量级、能跑通的最小原型**。栈:**环境** = R2E-Gym-Subset 任务 + 节点 CPU 上的 Docker 沙箱 + F2P/P2P 二元奖励;**rollout** = 轻量 scaffold(mini-swe-agent)+ vLLM-Ascend 推理 server(observation-token masking);**trainer** = MindSpeed-RL(Megatron 系)+ GRPO++;**异步** = 推理 server 与 trainer 解耦 + per-trajectory + staleness 有界。目标不是刷 SOTA,而是**量化并打磨昇腾特有的四个坑**:无 sleep-mode 的显存争用、train/infer 的 logprob 一致性(align-probe)、长 rollout 的显存、沙箱容器并行成本。里程碑:M0 单任务跑通奖励 → M1 SFT 冷启动 → M2 GRPO 7B 单节点 → M3 量化 FP8 rollout / 解耦异步 / 对 GPU 的差距。**provisional:本篇为方案设计(proposal),非已验证结果。**

承接 Dispatch 12(SWE agents 上手)。这期把"怎么搭"具体到昇腾,给一份能照着做的设计。

---

## 0. 目标与非目标

- **目标**:在**单节点 8× 910B** 上,用 GRPO 把一个 **7B 编码模型**在 SWE 任务上从基线训出可测量的提升;**把昇腾特有的瓶颈量化出来并给出缓解**;产出一张 GPU vs NPU 的吞吐/显存/一致性对照表。
- **非目标**:刷 SWE-bench 榜单、上 32B/MoE、多节点超大规模——这些是原型跑通后的扩展。
- **为什么从这做**:SWE rollout 又长又重(30–50 轮、分钟级容器),是昇腾显存+异步痛点的**放大器**,最能暴露问题、也最能体现前面各期招式的价值。

## 1. 总体架构(5 部件,标注昇腾选型)

```
[数据/任务池]        [环境=沙箱+奖励]         [Rollout=scaffold+推理]        [Trainer]
R2E-Gym-Subset  →   Docker(节点CPU)      →   mini-swe-agent             →  MindSpeed-RL
SWE-Gym-Lite        F2P/P2P → 0/1 reward     vLLM-Ascend server(NPU)       Megatron/GRPO++
(预过滤)            超时=0                   observation-token mask         周期性 weight sync
                        ▲                          │  trajectory(token+reward)  │
                        └──────────── 解耦异步:trainer 拉完成轨迹,staleness 有界 ──────────┘
```

关键原则:**推理(rollout)与训练分到不同 NPU 子集**,沙箱容器跑在**节点的 CPU**上(测试执行不需要 NPU)。

## 2. 环境与奖励

- **任务源**:首选 **R2E-Gym-Subset**(程序化生成、无需人写 issue/test、镜像规整),或 **SWE-Gym-Lite**(真实但小)。规模化阶段再上 **SWE-smith**(一 repo 一执行环境,适合批量)。
- **沙箱**:每个任务一个预构建 Docker 镜像,跑在节点 CPU 上;**禁网、固定 seed、超时**;维持 **warm pool** + 复用容器降 cold-start。
- **奖励**:二元——**所有 F2P 通过 且 P2P 不退化 → 1.0,否则 0.0**;format/parse 惩罚;超时=0。
- **预过滤(关键)**:对每个候选实例用 **golden patch 和 empty patch 各跑一遍**,丢掉"不能干净 FAIL→PASS"或不确定性高的任务——脏标签直接进 reward 会毁训练。

## 3. Rollout(scaffold + 推理引擎)

- **scaffold**:从 **mini-swe-agent**(bash-only、~100 行)起步,封装成 `(task, policy endpoint) → (token 级 trajectory, reward)`,走 OpenAI 兼容 API;成熟后可换 SWE-agent / R2E AgentHub。
- **推理引擎**:默认 **vLLM-Ascend**(910B 覆盖最广、最成熟);**备选 SGLang-Ascend**——它的 RadixAttention 能复用"一个任务采 N 条样本共享的 prompt 前缀"(见 Dispatch 10),对 group rollout 省 KV,但 Ascend 后端成熟度需先验证。
- **observation-token masking(必做)**:trajectory 里工具输出/观察 token **不参与 loss**,只对策略自己生成的 token 算——昇腾上重写注意力/采样算子后,这块若错会悄悄毁梯度。

## 4. Trainer

- **框架**:**MindSpeed-RL**(华为原生、Megatron 系、跑过 384-NPU 上的 GRPO,本看板 Ascend 标签有卡)为主;**verl-Ascend** 为备选。
- **算法**:**GRPO++**(clip-higher、无 KL、无 entropy bonus、长度归一化、组采样 8–16/任务)——对稀疏二元奖励最稳,无 value network。
- **引擎切分**:训练引擎(MindSpeed/Megatron 算梯度)与推理引擎(vLLM-Ascend 出 rollout)分离,周期性 **weight sync**(优先零拷贝/host 中转)。
- **冷启动**:先用 **SWE-smith-trajectories** 几千条专家轨迹做 SFT,再进 GRPO。

## 5. 异步 + 信用分配

- **解耦异步**:推理独立成 server,trainer 拉**已完成**的轨迹——避免同步批生成让 NPU 干等最慢/最长的那条(SWE 长尾极重)。这正是 Project Ideas 里"**异步 off-policy RL on Ascend**"那条的 SWE 版。
- **per-trajectory** 与沙箱交互(而非 per-batch),让快任务先回。
- **信用分配**:默认"结果奖励 + 长度归一化广播 + GRPO 组采样";太稀疏时上 **turn-level / GiGPO**(把每轮当 MDP step)。
- **staleness 修正**:in-flight 权重更新会让 rollout 由略旧策略产生 → 限制 staleness + **重要性采样修正(TIS/MIS)**。

## 6. 昇腾特有的四个坑 + 缓解(本原型的核心产出)

1. **无 vLLM sleep-mode 的显存争用** —— 910B 单卡 64GB,rollout 引擎无法在 train 阶段释放 KV/权重(见 NPU 架构页"RL 显存争用"视图)。
   - 缓解:**解耦**(推理/训练分到不同 NPU)+ **host offload** + **限轮/限上下文** + **FP8/量化 rollout**(Dispatch 02,直接缩 rollout 占用)。量化"省了多少"是 M3 的关键指标。
2. **train/infer logprob 不一致** —— vLLM-Ascend/SGLang-Ascend 的采样 logprob 与 MindSpeed/Megatron 的训练 logprob,因 NPU 算子数值差异可能漂移,悄悄毁梯度。
   - 缓解:**align-probe**——早期就测**逐 token logprob 的 MAE**;超阈值则 trainer 侧重算 logprob,或用 TIS/MIS 兜。**这是整个原型最该先做的健全性检查。**
3. **长 rollout 的显存/上下文** —— 30–50 轮会冲爆上下文。
   - 缓解:限 max turns/tokens、**observation masking**、**compaction**(摘要历史);截断/未 submit 的轨迹**给 reward 0 但不丢弃**(丢弃使梯度有偏)。
4. **沙箱容器并行/成本** —— 主导成本其实是 CPU/内存/编排,不是 NPU。
   - 缓解:预构建镜像 + warm pool + 安全复用;沙箱与 NPU 解耦扩缩;监控容器吞吐别成新瓶颈。

## 7. 里程碑路径

- **M0 · 环境跑通(最小闭环)**:1 个任务,golden/empty 双跑过滤,用**托管模型**跑出一条 trajectory + 正确 reward。**不训练**。验收:reward 计算正确、沙箱稳定。
- **M1 · SFT 冷启动**:SWE-smith-trajectories 几千条,910B 上 SFT 一个 7B(Qwen2.5-Coder-7B 一类、vLLM-Ascend 已支持)。验收:loss 正常、能产出合法 patch。
- **M2 · GRPO 单节点**:R2E-Gym-Subset 上跑 GRPO++,先**同步**打通,再切**解耦异步**。验收:reward 上升、**轨迹长度稳定(防 hacking)**、logprob MAE 在阈值内。
- **M3 · 量化昇腾杠杆**:分别测 **FP8/量化 rollout**、**解耦异步** 的显存/吞吐收益,并对一台 GPU 节点做同配置对照。产出 **吞吐 / MFU / 峰值 HBM / train-infer 一致性** 对照表——正是 Project Ideas 里"**GRPO-on-Ascend Benchmark**"那条的落地。

## 8. 评测与验收

- **能力**:SWE-bench-**Live**(post-cutoff,抗污染)小切片 + R2E held-out;别用被污染的 Verified 当唯一指标。
- **系统**:GPU vs NPU 的 throughput / MFU / 峰值 HBM;**logprob 一致性(MAE)**;rollout 占总时长比例;沙箱容器吞吐。
- **健康**:reward 与平均轨迹长度并行监控(长度骤降+reward 升 = hacking)。

## 9. 风险与回退

- **logprob 漂移太大** → 先 trainer 侧重算 logprob(慢但正确),再优化。
- **沙箱成为瓶颈** → 降并发 group size、加 CPU 节点、缩任务镜像。
- **解耦异步不稳** → 回退同步 GRPO 先拿到正确性,再逐步放开 staleness。
- **7B 信号太弱** → 用更易的任务子集(pass-rate 0.2–0.8 区间)保住梯度,而非直接上大模型。

## 10. 与本看板的联系

这套原型把多条线收口:**Project Ideas** 的「GRPO-on-Ascend Benchmark」「异步 off-policy RL」「FP8 RL on Ascend」在这里同时落地;**align-probe** 是坑 #2 的解法;**RL 显存争用**视图解释坑 #1;Dispatch 02(rollout 瓶颈/FP8)、08(agentic RL)、12(SWE 配方)是它的上游。**一句话:SWE agentic RL 是压测昇腾 RL 系统最严苛、也最有信息量的负载。**

---

*本篇为方案设计(proposal),非已验证结果;具体超参/吞吐需实跑确定。参考:本看板 Dispatch 02 / 08 / 10 / 12,MindSpeed-RL、vLLM-Ascend、SGLang、R2E-Gym / SWE-Gym / SWE-smith、DeepSWE(rLLM)。*
