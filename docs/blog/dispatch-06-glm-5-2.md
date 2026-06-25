# Dispatch 06 · GLM-5.2 全面评测与技术分析

*2026-06-23 · NPU Frontier Dispatch · model / GLM-5.2 / benchmarks / RL-on-NPU*

> **TL;DR** — GLM-5.2(智谱 Zhipu / Z.ai,2026-06-13,**MIT 开源**)是当前**最强的开源权重编程模型**之一:**~753B MoE / ~40B 激活**,配新的 **IndexShare 稀疏注意力**撑起**可用的 1M 上下文**、131K 输出。编程上 **SWE-bench Pro 62.1**(超 GPT-5.5 的 ~58.6)、**FrontierSWE 74.4**、**Terminal-Bench 2.1 81.0**(逼近 Opus 4.8 的 85);数学 **AIME 2026 99.2**、科学 **GPQA Diamond 91.2**。价格 **$1.40 / $4.40 每 1M(入/出)**,约为 GPT-5.5 的 **1/6**。短板在**智能体工具**(Tool-Decathlon 明显落后)且**没公布** SWE-bench Verified / LiveCodeBench / Aider 这三项社区常用基准——读榜要留个心眼。对 RL-on-NPU:**MIT 许可 + 强编程 = 极佳的开源 RL 后训练基座**;GLM-5 已在 vLLM-Ascend(W8A8C8),5.2 尚未按名列入。

接 Dispatch 05(DeepSeek-V4)。这期应要求,把 **GLM-5.2** 的评测和技术做一次尽量全的梳理。所有数字均为**厂商/媒体口径,provisional**,以官方报告与第三方复现为准。

---

## 1 · 身份与定位

- **厂商**:智谱 AI(国际品牌 **Z.ai**)。
- **发布**:2026-06-13,**MIT 开源权重**(HuggingFace `zai-org/GLM-5.2`)。
- **定位**:coding-first 的旗舰 MoE,主打**智能体工程(agentic engineering)** 与长上下文编码;延续 GLM-5《from Vibe Coding to Agentic Engineering》(arXiv 2602.15763)与 GLM-4.5 ARC(Agentic/Reasoning/Coding,arXiv 2508.06471)的路线。
- 快速迭代:GLM-5 → 5.1 → 5.2,几个月一代,是对出口管制的"开源回应"。

## 2 · 架构详解

| 维度 | GLM-5.2 | 备注 |
|---|---|---|
| 结构 | 稀疏 **MoE** | 承袭 GLM-5 的 744B 级 MoE |
| 总参数 / 激活 | **~753B / ~40B** | 每 token 仅激活 ~40B |
| 注意力 | DSA 式稀疏 + **IndexShare** | 索引器每 4 层一组复用,控 1M 推理成本 |
| 上下文 | **1M**(可用) | 比 GLM-5.1 的 ~200K **翻 5×** |
| 最大输出 | **131,072** token | 长代码 / 长报告 |
| 推理模式 | **High / Max** 双档 | 按难度切深思 |
| 预训练 | **28.5T** token | |
| 许可 | MIT | 可商用、可改、可自托管 |

几个要点:

- **IndexShare 稀疏注意力**是 5.2 的关键工程:为了把 200K→1M 的上下文成本压住,GLM 在 DSA(DeepSeek Sparse Attention)式的稀疏注意力之上加了 **IndexShare**——**把稀疏注意力的"选块索引器"在每 4 层一组里复用**,而不是每层都重算一遍索引,省下索引开销。与 DeepSeek 的 CSA/HCA、MiniMax 的 MSA 同属 2026 的"稀疏注意力潮"(见 Dispatch 04/05)。"可用的 1M"是卖点:号称在几十万 token 后**不塌**,而非纸面长度。
- **40B 激活 / 753B 总**:典型的"大稀疏、小激活"——推理算力≈40B 稠密,但要装下 753B 权重,**自托管门槛高**(多卡/多 NPU)。
- **双推理档(High/Max)**:把"快答"和"深思"分开,贴合编码 vs 智能体长任务。

## 3 · 评测全表(provisional)

**① 编程 / 软件工程**

| 基准 | GLM-5.2 | 对照 |
|---|---|---|
| SWE-bench Pro | **62.1** | GLM-5.1 58.4 · GPT-5.5 ~58.6 |
| FrontierSWE | **74.4** | GPT-5.5 72.6 · Opus 4.8 75.1 |
| Terminal-Bench 2.1 | **81.0**(最佳 harness 82.7) | Opus 4.8 85.0 |

**② 推理 / 数学 / 科学**

| 基准 | GLM-5.2 |
|---|---|
| AIME 2026 | **99.2** |
| GPQA Diamond | **91.2** |
| HLE(Humanity's Last Exam,带工具) | 54.7 |

**③ 智能体 / 工具使用**

| 基准 | GLM-5.2 | 对照 |
|---|---|---|
| MCP-Atlas | **76.8** | Opus 4.8 77.8(几乎打平) |
| Tool-Decathlon | 明显落后 | 落后 Opus 4.8 / GPT-5.5 |

**④ 横向定位**

- Artificial Analysis 上被列为**当前领先的开源权重模型**。
- 对 Opus 4.8:编码差距收到 ~1–4 分(FrontierSWE 74.4 vs 75.1、MCP-Atlas 76.8 vs 77.8、Terminal-Bench 81 vs 85),个别项在最佳 harness 下反超。
- 对 GPT-5.5:SWE-bench Pro(62.1 vs ~58.6)、FrontierSWE(74.4 vs 72.6)**领先**。

## 4 · 价格(API)

| 模型 | 输入 $/1M | 输出 $/1M | 相对 |
|---|---|---|---|
| **GLM-5.2** | **1.40** | **4.40** | 基准 |
| GPT-5.5 | 5 | 30 | ~6× 贵(综合) |
| Claude Opus 4.8 | 5 | 25 | ~5–7× 贵 |

开源 + 便宜 + 1M 上下文,是 GLM-5.2 最锋利的组合拳:**性能贴近闭源旗舰,价格约 1/6**。

## 5 · 强项与短板(读榜须知)

**强项**
- **编程/软工**是主战场:SWE-bench Pro、FrontierSWE、Terminal-Bench 都在第一梯队。
- **数学/科学**:AIME 2026 99.2、GPQA 91.2 非常高。
- **可用的 1M 上下文** + **MIT 开源** + **极低价**。

**短板 / 需要警惕**
- **智能体工具不均衡**:MCP-Atlas 接近 Opus,但 **Tool-Decathlon 明显落后**,HLE(带工具)54.7 一般。多步工具编排还不是它的强项。
- **关键基准缺席**:智谱**没有公布** SWE-bench **Verified**、**LiveCodeBench**、**Aider polyglot**——恰恰是开源社区最常用的三项 agentic-coding 基准。选了对自己有利的榜单,**对比时要补齐这块空白**。
- **API 数据风险**:有报道提示通过其 API 使用存在中国数据合规顾虑;**自托管(MIT 权重)可规避**这条。
- 所有数字仍是**厂商/媒体口径**,待第三方复现。

## 6 · 对 RL-on-NPU 的意义

为什么本看板把 GLM-5.2 看作**首选开源 RL 基座**:

- **MIT + 强编程 = 理想的 RL 后训练底座**。可商用、可改权重,适合在昇腾上做 GRPO/RLVR 的开源实验对象;编程/智能体方向恰是 agentic RL 的训练场。
- **昇腾就绪度:partial**。GLM-5 已在 **vLLM-Ascend**(性能优化 + **W8A8C8** 量化),**GLM-5.2 尚未按名列入**——是个明确、可做的移植缺口。
- **IndexShare 稀疏注意力 = 又一条要移植/对齐的路径**。decode 端省 KV 直接缓解昇腾"无 sleep-mode"的 rollout 显存争用(见 NPU 架构页"RL 显存争用"视图);但 NPU 上重写稀疏注意力会引入 train-inference 数值漂移——交给 **align-probe** 量化。
- **自托管成本高**:753B 总参数要多卡/多 NPU,RL 训练更需 384-NPU 级超节点(对口 MindSpeed-RL / Atlas 950,见 Dispatch 03)。

## 7 · 下一步看什么

1. **第三方复现 + 补齐缺席基准**:SWE-bench Verified / LiveCodeBench / Aider 的独立结果。
2. **GLM-5.2 进 vLLM-Ascend 的时间点**与量化方案(W8A8C8?)。
3. **IndexShare 的机制细节**:与 DSA / MSA / CSA-HCA 的异同——值得单开一期详解。
4. **拿 GLM-5.2 当基座在昇腾上做 RL** 的首个端到端工作。

---

*来源:GLM-5.2 评测/解析(theairankings、edenai、digitalapplied、apidog、lushbinary、bitsminds、techtimes、latent.space、Hacker News 等)、智谱/Z.ai 与 HuggingFace `zai-org/GLM-5.2`、GLM-5 技术报告(arXiv 2602.15763)、GLM-4.5 ARC(arXiv 2508.06471)、vLLM-Ascend 支持矩阵。数字均为厂商/媒体口径,provisional。相关卡片见本看板 LLM Modeling 标签页与 Overview 对比组件。*
