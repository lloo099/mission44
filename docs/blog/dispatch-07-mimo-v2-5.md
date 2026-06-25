# Dispatch 07 · 详解 MiMo-V2.5-Pro:滑窗×全局 6:1 混合注意力 + MiMo Code

*2026-06-25 · NPU Frontier Dispatch · model / MiMo-V2.5 / hybrid-attention / RL-on-NPU*

> **TL;DR** — MiMo-V2.5-Pro(小米,2026-04-28,**MIT 开源**)是一记"硬件厂商杀进模型赛道"的重拳:**1.02T / 42B 激活 MoE**,但长上下文的省法和别家都不同——不靠块稀疏(MSA/DSA),而是**交替堆叠滑动窗口注意力(SWA,128-token 窗)与全局注意力,比例 6:1**,把 KV cache 砍掉约 **7×**,撑起 1M 上下文。真正的杀手锏是配套的 **MiMo Code** 智能体编码 harness:厂商基准里 MiMo Code + V2.5-Pro 在 **SWE-bench Verified 82 / Pro 62 / Terminal-Bench 2 73** 上分别压过 Claude Code + Sonnet 4.6,且在 **200+ 步超长任务**上更稳。对 RL-on-NPU:SWA 解码极省、7× KV cut 直接缓解 rollout 显存,MIT 许可适合做 RL 基座——**但目前没有 Ascend 移植**,且跑分与 harness 强绑定,要打折看。

接 Dispatch 06(GLM-5.2)。应要求把这次研究动态里最值得单讲的新模型 **MiMo-V2.5-Pro** 拆开。所有数字均**厂商/媒体口径,provisional**。

---

## 1 · 身份与定位

- **厂商**:小米(Xiaomi);MiMo 团队。
- **发布**:2026-04-28 正式开源(4-23 起公测),**MIT 许可**(可商用推理 + 二次训练,无需额外授权)。
- **家族**:
  - **MiMo-V2.5-Pro** —— 旗舰,**1.02T 总 / 42B 激活** MoE,面向编码 agent、复杂软件工程、超长工具链。
  - **MiMo-V2.5**(标准)—— **310B 总 / 15B 激活**,约 **48T token** 预训练。
- **定位**:agentic coding + 长程推理 + 多模态;主打"硬件厂商把端侧/Agent 能力做进开源大模型"。

## 2 · 核心:滑窗 × 全局 6:1 的混合注意力

MiMo 走的不是"块稀疏选择"那条路(MSA 选块、DSA token 级 top-k、GLM-5.2 IndexShare 复用索引器),而是经典但有效的 **局部 / 全局交替**:

- **SWA(Sliding-Window Attention)**:大多数层只看 **128 token 的局部窗口**——极省、解码友好。
- **Global Attention**:每隔几层插一层**全局注意力**,补"长程视野"。
- **比例 6:1**:每 6 层 SWA 配 1 层全局。绝大多数层是廉价的局部注意力,少数全局层兜住长依赖。
- **效果**:长上下文下 **KV cache ~7× 降**,质量不塌,支撑 **1M** 上下文。

和 2026 几条长上下文路线对照:

| 方案 | 机制 | 省的方式 |
|---|---|---|
| **MiMo-V2.5(SWA×Global 6:1)** | 局部 128-窗 + 周期性全局层 | 大多数层只看局部 → KV ~7×↓ |
| MiniMax **MSA** | 真实 KV 上块选 top-k | 每 query 固定看 2048 token |
| DeepSeek **CSA+HCA** | KV 压 4× / 128× + 选/全看 | 压缩 KV,1M 下 FLOPs↓~4× |
| GLM-5.2 **DSA + IndexShare** | token 级稀疏 + 索引器每 4 层复用 | 省索引重算 |

一句话:**MiMo 用"局部窗 + 稀疏全局层"换便宜,别家用"在压缩/真实 KV 上选块"换便宜。** 局部窗的工程最成熟、kernel 最现成。

## 3 · 杀手锏:MiMo Code 智能体 harness

MiMo 的卖点不只是模型,还有配套的 **MiMo Code** —— 一个开源 agentic 编码 harness。厂商基准(MiMo Code + V2.5-Pro vs Claude Code + Sonnet 4.6):

| 基准 | MiMo Code + V2.5-Pro | Claude Code + Sonnet 4.6 |
|---|---|---|
| SWE-bench Verified | **82** | 79 |
| SWE-bench Pro | **62** | 55 |
| Terminal-Bench 2 | **73** | 69 |

- 另:开源榜 **GDPVal-AA / ClawEval 第一**。
- 媒体强调它在**200+ 步的超长 agentic 任务**上比 Claude Code 更稳。
- **读榜须知**:这些是 **harness + 模型一起**测的(MiMo Code 这套脚手架本身在帮分),换个 harness 数字会变;且全是厂商口径。**和裸模型对比时要注意这层耦合**。SWE-bench Verified 82 若成立,会是当前开源最高。

## 4 · 价格

- 输入约 **$1.00 / 1M token**(token-efficient agent 定位)。输出价各源不一,这里从略。
- 叠加 MIT 开源可自托管,成本面对标 GLM-5.2 / DeepSeek-V4 那一档(便宜)。

## 5 · 对 RL-on-NPU 的意义

- **SWA 解码极省,正中 rollout 痛点**。RL 的 rollout 是 decode-heavy + memory-bound;128-token 滑窗让绝大多数层的 KV 访存几乎是常数级,**7× KV cut** 直接松绑昇腾"无 sleep-mode"的显存争用(见 NPU 架构页"RL 显存争用"视图)。这是比块稀疏更易拿到的工程收益。
- **kernel 最现成**。SWA + 周期性全局是成熟模式,NPU 上重写比 MSA/DSA/CSA 这类自定义稀疏注意力风险更低——**移植友好度高**。
- **MIT = 可做 RL 基座**。可商用 + 可二次训练,适合在昇腾上做 agentic RL 实验;MiMo Code 的多步 agent 场景本身就是 agentic RL 的训练场。
- **但有两个坎**:① **目前无 Ascend 移植**(未在 vLLM-Ascend 列出);② 跑分 harness 强绑定,需要裸模型复现来定真实水平。
- **数值一致性**:SWA 的窗口边界 + 全局层在 NPU 上重写,仍要用 **align-probe** 量化 train-inference 漂移。

## 6 · 下一步看什么

1. **裸模型(非 MiMo Code)复现**:SWE-bench Verified/Pro 在标准 harness 下还剩多少。
2. **MiMo-V2.5 上 vLLM-Ascend 的时间点**:SWA 混合注意力在 910B/950 上的吞吐与精度。
3. **SWA×Global vs 块稀疏的长检索质量**:6:1 的局部/全局配比在多跳/长检索上是否够稳。
4. **MiMo Code 当 agentic RL 环境**:把它的 200+ 步任务做成 RL 训练/评测环境。

---

*来源:小米 MiMo 官方(mimo.mi.com / mimo.xiaomi.com)、MiMo-V2.5 开源公告、VentureBeat(MiMo Code vs Claude Code)、BigGo/fonearena/Medium 等解析。规格与跑分均厂商/媒体口径,provisional;基准与 MiMo Code harness 耦合。相关卡片见本看板 LLM Modeling 标签页与 Overview 对比组件。*
