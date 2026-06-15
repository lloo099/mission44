# RL Training on NPU (Ascend) — 研究报告

> 生成日期:2026-06-15 · 所有条目均来自一手来源(arXiv / 官方仓库 / 博客),链接见文末与 dashboard。
> 配套交互式 dashboard:`index.html`(6 个标签页,搜索 + 标签过滤;数据在 `data/*.json`)。

---

## 一、核心结论(Thesis)

三条主线正在交汇:

1. **RL post-training 已成为前沿能力的主要杠杆** —— 从 RLHF 到 RLVR/GRPO,2025 年的推理模型几乎都靠大规模 RL 训练出来。
2. **Ascend 软硬件栈已经成熟到"能做真 RL"** —— 但几乎所有工具链都默认 CUDA。
3. **机会窗口非常具体**:Ascend 上做 RL 的瓶颈是少数几个清晰、可攻克的工程/研究缺口,而非"完全不能做"。

> **一句话:在 NPU 上把高效 RL 训练真正跑起来、跑大、跑准,是目前开源生态最明确的空白。**

---

## 二、RL for LLMs:技术全景

### 1. 算法演进(从 PPO 到 2025 新变体)

| 阶段 | 代表 | 关键创新 | 来源 |
|---|---|---|---|
| 经典 RLHF | **PPO / InstructGPT** (2022) | SFT→RM→PPO 三段式 | arXiv:2203.02155 |
| 去 critic | **GRPO** (DeepSeekMath, 2024-02) | 组内相对优势,去掉 value network → 现在主流底座 | arXiv:2402.03300 |
| 去 RM | **RLVR** (Tülu 3, 2024-11) | 用可验证函数(对/错)替代奖励模型 | arXiv:2411.15124 |
| 去 critic(REINFORCE) | **RLOO / ReMax** | leave-one-out / 贪心 baseline,更省显存 | arXiv:2402.14740 / 2310.10505 |
| 离线偏好 | **DPO / KTO / SimPO / ORPO / IPO** | 把对齐变成分类损失,不采样不 RL | arXiv:2305.18290 等 |
| 2025 推理变体 | **DAPO** | Clip-Higher + 动态采样 + token 级 loss + 超长奖励整形 | arXiv:2503.14476 |
| | **Dr.GRPO** | 去掉长度/标准差归一化偏置(避免response越来越长) | arXiv:2503.20783 |
| | **GSPO** | 序列级重要性比与裁剪,**稳定 MoE RL**,用于 Qwen3 | arXiv:2507.18071 |
| | **CISPO** | 裁剪 IS 权重而非 token,保留关键低概率 token | arXiv:2506.13585 |
| | **VAPO** | value-based PPO 回归,长 CoT SOTA | arXiv:2504.05118 |
| | **ProRL** | 超长 RL 拓展推理边界(KL 控制 + 参考策略重置) | arXiv:2505.24864 |

**对 NPU 选型的启示**:GRPO 系(去 critic、内存友好)最适合显存/带宽受限的 NPU;GSPO 对 MoE RL 的稳定性尤其关键。

### 2. RL 训练出来的推理模型(2024–2026)

| 模型 | 机构 | RL 方法要点 | 来源 |
|---|---|---|---|
| **DeepSeek-R1 / R1-Zero** | DeepSeek | 纯 RL(无 SFT)即涌现 long-CoT;引爆 RLVR 浪潮 | arXiv:2501.12948 |
| **OpenAI o1 / o3 / o4-mini** | OpenAI | 大规模 RL-on-CoT + 训练内化 agentic 工具使用(闭源) | openai.com |
| **QwQ-32B** | Qwen | verifier-based RL,32B 追平 671B R1 | qwenlm.github.io |
| **Qwen3** | Qwen | 统一 thinking/non-thinking + thinking budget;GRPO | arXiv:2505.09388 |
| **Kimi k1.5** | Moonshot | 简化 RL,scale context length,无 MCTS/value/PRM | arXiv:2501.12599 |
| **MiniMax-M1** | MiniMax | CISPO + lightning attention,512×H800 三周完成 | arXiv:2506.13585 |
| **Magistral** | Mistral | 纯 RL,自建异步训练栈 | arXiv:2506.10910 |
| **GLM-4.5** | 智谱 | 难度课程 RL,用开源 slime 框架 | arXiv:2508.06471 |
| **Seed1.5-Thinking** | ByteDance | 可验证+通用+混合三引擎数据 | arXiv:2504.13914 |
| **AceReason-Nemotron** | NVIDIA | 先数学 RL 后代码 RL 的课程 | arXiv:2505.16400 |
| **Phi-4-reasoning** | Microsoft | 精选 SFT + outcome-based RL(小模型) | arXiv:2504.21233 |

### 3. 开源 RL 框架(含 Ascend 支持情况 ⭐重点)

| 框架 | 机构 | 算法/特点 | Ascend 支持 |
|---|---|---|---|
| **verl** (HybridFlow) | ByteDance | PPO/GRPO/GSPO/DAPO,FSDP+Megatron,可到 671B | ✅ **一方支持**(专门 Ascend 教程) |
| **TRL** | Hugging Face | GRPO/DPO/PPO trainer,易用 | ✅ 910B GRPO 经 vLLM-Ascend 验证 |
| **ROLL** | Alibaba | 多角色 Ray,RLHF/RLVR/agentic | ✅ 一方 Ascend 使用指南 |
| **AReaL** | 蚂蚁/清华 | 全异步(~2.77× 加速) | ✅ 专门 `ascend` 分支(~2026-01 稳定) |
| **slime** | 智谱/THUDM | Megatron↔SGLang,GLM 背后的栈 | 间接(经 SGLang rollout) |
| **OpenRLHF** | 社区 | 首个 Ray+vLLM RLHF | ❌ WIP(issue #852) |
| **NeMo-RL / Oat / open-instruct / Unsloth** | — | — | ❌ 主要 NVIDIA-only |

### 4. 2025–2026 研究趋势(精选)

- **Process Reward Models**:判别式 PRM → 生成式"会思考"的 ThinkPRM(数据效率高约 100×)。arXiv:2504.16828 / 2501.07301
- **异步 / off-policy RL**:Async RLHF(2410.18252)、AReaL(2505.24298)、**ExGRPO** 经验回放(2510.02245)—— 直接缓解 NPU 同步 colocated RL 的瓶颈。
- **数据效率惊人**:**1-shot RLVR**(一个样本接近上千,2504.20571);**Spurious Rewards**(随机奖励在 Qwen 也涨分但换模型失效 → RLVR 多是"激发"而非"教会",2506.10947)。
- **探索 / 稳定性**:熵坍缩机制 **Clip-Cov / KL-Cov**(2505.22617)。
- **零数据自博弈**:Absolute Zero(2505.03335)、R-Zero(2508.05004)。
- **Agentic RL**:Search-R1(2503.09516)、ReTool(2504.11536)、RAGEN/StarPO(2504.20073)。
- **Test-time RL**:TTRL 用多数投票做伪奖励,无标签也能 RL(2504.16084)。
- **奖励黑客缓解**:Reward Shaping / PAR(2502.18770)。

---

## 三、Ascend / NPU 生态

### 1. 硬件

| 产品 | 年份 | 关键规格 | 定位 |
|---|---|---|---|
| **Ascend 910B** | 2024 主力 | ~256–280 TFLOPS BF16,64GB HBM2e @ ~1.2TB/s | 实测约 A100 级,最易获得 |
| **Ascend 910C** | 2025 旗舰 | 双 die,~780 TFLOPS,128GB HBM | ~H100 dense FP16 的 80%,能效/带宽落后 |
| **CloudMatrix 384** | 2025-04 | 384×910C,~300 PFLOPS BF16,~49TB HBM | 对标 NVL72;靠规模堆叠,代价 ~2.3× 能耗/FLOP |

> 注:910C / CM384 规格多来自分析机构报道,视为近似值。

### 2. 软件栈(与 CUDA 的对应)

| Ascend | CUDA 对应 |
|---|---|
| **CANN**(全家桶) | CUDA + cuDNN + NCCL |
| **HCCL** | NCCL |
| **Ascend C**(kernel 语言) | 写 CUDA kernel |
| **AscendCL / pyACL** | CUDA Runtime/Driver API |
| **MindSpore / torch_npu** | 框架后端 |

**关键**:没有 CUDA 兼容层 —— 必须**移植**而非重编译。2025-08 华为宣布 Ascend 全栈开源。

### 3. RL-on-Ascend 现状(实测可行)

- **MindSpeed-RL**(华为一方,最稳):已在 **384-NPU 超节点**上对 **DeepSeek-R1-MoE-671B** 做 GRPO(arXiv:2507.19017)。
- **verl on Ascend**:验证了 7B 级 GRPO + R1-Zero 复现(8/16 NPU)。
- **单台 Atlas 800T A2(8×910B)今天就能跑 0.5B–7B 的 GRPO**。

### 4. 移植痛点(= 机会)

1. **缺 vLLM sleep mode / hybrid engine** → 无法显存高效地 colocate actor+rollout(**最大的 RL 专属瓶颈**)。
2. **无 FlashAttention**:只能用 `npu_fusion_attention`;FA3 未公开。算子自动转换覆盖率仅 ~65%,长尾算子需手写 Ascend C。
3. **精度/收敛漂移**:DAPO 在 Ascend reward MAE ~3.7–4.3%(vs GRPO 0.3–3.3%)。
4. **带宽瓶颈**:910B 算力约 H100 的 60%,但内存带宽仅 ~36% → 更 compute-bound,MFU 偏低(verl 实测吞吐比约 0.38–0.59)。
5. **生态成熟度**:中文为主、平行工具链(msprof/msDebug),HCCL 不如 NCCL 久经考验。(参考:DeepSeek R2 据报因 Ascend 训练不稳定回退 NVIDIA 训练、Ascend 仅做推理。)

---

## 四、LLM Modeling:对 NPU 最相关的技术

显存墙是 NPU 的核心约束,以下技术直接缓解:

1. **MLA(Multi-head Latent Attention)** —— KV cache 降 ~93%,DeepSeek-V2/V3/R1 已验证 → NPU 上**最值钱的一招**(arXiv:2405.04434)。
2. **FP8/FP4 训练** —— DeepSeek-V3 FP8(细粒度 tile/block 缩放,2412.19437)、微软 FP4(2501.17116)、NVIDIA NVFP4(2509.25149)。
3. **GQA + Sliding Window** —— 廉价、默认、稳(2305.13245 / 2310.06825)。
4. **muP / muTransfer** —— 小模型调参 → 大模型零样本迁移,避免在受限硬件反复全尺寸跑(2203.03466)。
5. **量化(AWQ/GPTQ)+ Compute-Optimal QAT** —— rollout/部署省显存(2306.00978 / 2509.22935)。
6. **长上下文** —— YaRN(2309.00071)、LongRoPE(2402.13753)、Ring Attention(2310.01889)、ChunkKV 语义块级 KV 压缩(2502.00299)、线性/SSM 混合(Mamba-2 2405.21060、MiniMax lightning attention)。

---

## 五、机会分析 & 项目建议

按"影响 / 难度 / 新颖度"打分(满分 5):

| 项目 | 影响 | 难度 | 新颖 | 一句话 |
|---|---|---|---|---|
| **GRPO-on-Ascend 基准 + 精度对齐套件** ⭐推荐起步 | 5 | 2 | 3 | 单节点即可,把零散的跨平台吞吐/精度数据做成自动化基准,补社区空白 |
| **Ascend 的 sleep-mode / colocated hybrid engine** | 5 | 5 | 5 | 攻克最大 RL 瓶颈,真正的系统级贡献 |
| **面向 NPU 约束的异步 off-policy RL** | 4 | 4 | 4 | 用 AReaL ascend 分支 + ExGRPO 回放绕开瓶颈 |
| **Ascend 上的 FP8 RL 训练** | 4 | 5 | 5 | FP8 用在 RL loop(而非仅预训练)几乎无人做 |
| **单节点 R1-Zero 复现 cookbook** ⭐易上手 | 4 | 2 | 2 | 社区急需的"一键上手"教程,可结合 1-shot RLVR |
| **verl-Ascend 熵坍缩缓解研究** | 3 | 3 | 4 | 研究 NPU 数值噪声是否改变 entropy collapse 动态 |
| **MLA + KV 压缩做长上下文 RL** | 4 | 4 | 4 | 针对 NPU 弱带宽,扩展可训练 context 长度 |
| **RL 算子覆盖/自动移植工具** | 5 | 5 | 4 | 系统化解决 ~65% 算子覆盖 + 手写 kernel 难题 |

**建议路径**:先做 **GRPO-on-Ascend 基准套件**(影响大、难度低、单节点可行,吃透整条栈),再选 **sleep-mode** 或 **FP8 RL** 作为高天花板课题。

---

## 附:数据规模

- `data/rl.json` —— 55 条(算法 / 推理模型 / 框架 / 趋势)
- `data/ascend.json` —— 32 条(硬件 / 软件栈 / 训练&推理框架 / RL-on-Ascend / 移植挑战)
- `data/modeling.json` —— 32 条(MoE / attention / 长上下文 / 低精度 / 模型报告)
- `data/ideas.json` —— 8 条项目想法(含 impact/difficulty/novelty 评分与首步)
- `data/feed.json` —— Live Papers,可用 `scripts/fetch_arxiv.py` 刷新

> 免责声明:内容为一手来源人工整理的某时刻快照;部分 arXiv 页面对自动抓取返回 403,日期/ID 经多源交叉验证,引用前建议手动点开核对。
