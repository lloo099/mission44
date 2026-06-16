# 2026 开放前沿模型对比:DeepSeek-V4 · MiniMax-M3 · GLM-5.2

> 视角:RL-on-NPU(Ascend)。重点看**架构创新(尤其稀疏注意力)**、**性能**、**Ascend 适配**与**对 RL 训练的影响**。
> ⚠️ 基准数字多来自厂商/媒体/第三方榜单,**属临时口径**,等独立复现校准。来源见文末。

---

## TL;DR 对比表

| | **DeepSeek-V4** | **MiniMax-M3** | **GLM-5.2** | (参照) DeepSeek-V3.2 |
|---|---|---|---|---|
| 发布 | 2026-04 | 2026-06 | 2026-06 | 2025-12 |
| 规模 | V4-Pro ~1.6T/49B active;V4-Flash 284B/13B | 109B(多模态) | 开放权重(MIT) | 671B-MoE |
| 上下文 | 1M | 1M | 1M(部分来源称 200K+) | 长上下文 |
| 注意力 | **CSA + HCA**(混合,逐层交替) | **MSA**(GQA 上块稀疏) | GLM-5 谱系 | **DSA**(MLA 上 token 级 top-k) |
| 底座 | DeepSeekMoE + MTP | GQA + 原生多模态 | coding-first | MLA + DeepSeekMoE |
| 许可 | 开放权重 | 开放权重 | **MIT** | 开放权重 |
| 价($/1M out) | ~$3.48(Pro)〜$0.87(不同来源) | **~$1.20**(促销价) | — | — |
| Ascend | ✅ 910B 已支持(vLLM-Ascend) | ⚠️ 家族量化支持,M3 未按名列入 | ⚠️ GLM-5 支持,5.2 待列入 | ✅ V3.x 已支持 |

---

## 一、2026 最大的架构主线:稀疏注意力分化

2025→2026 各家不再用同一种长上下文方案,而是**沿不同工程取舍分裂**。理解这条线,基本就理解了这批模型:

| 方案 | 底座 | 机制 | 取舍 |
|---|---|---|---|
| **DSA**(DeepSeek V3.2) | MLA | 轻量 "lightning indexer" 算 query↔token 分数 → **token 级 top-k** | 质量最稳,但 kernel 工程重 |
| **NSA**(DeepSeek 早期) | GQA | 三分支(压缩/选择/滑窗)+ 门控 | 质量天花板最高,实现最复杂 |
| **CSA + HCA**(DeepSeek V4) | — | **CSA**:把每 m 个 token 的 KV 压成 1 条 → 再做 DSA(query 只 attend top-k 压缩条目);**HCA** 更激进压缩;两者逐层交替 | 长上下文最省,牺牲一点细粒度 |
| **MSA**(MiniMax M3) | **GQA(非 MLA)** | 块级选择(像 CSA)**但 attention 在真实 KV 上做、不在压缩维度** | 不追理论最优,**追"现成 kernel 立刻能跑、跑得快"** |

**关键洞察(对 NPU 尤其重要):**
- **MiniMax 的设计哲学最务实** —— 基于 GQA(不是 MLA)、块选择在真实 KV 上做,目的就是**复用现有 kernel**。这正是 Ascend 这种"算子覆盖不全、FlashAttention 要换 npu_fusion_attention"的平台最吃香的特性 —— **越是标准 GQA、越容易移植**。
- DeepSeek 的 CSA/HCA/DSA 走 MLA + 重 kernel 路线,**质量/省内存更好,但移植成本高** —— 在 GPU 上 SGLang/vLLM 已适配,Ascend 上 V4 也已支持,但底层 kernel 是适配重点。
- 对 **RL rollout** 来说,decode 阶段的稀疏注意力加速 = 直接降低 rollout 时延和显存,这恰好缓解 Ascend 上"无 vLLM sleep-mode、rollout/train 抢显存"的痛点。

---

## 二、性能(临时口径,务必存疑)

**编码 / SWE-bench(变体口径混乱,注意):**
- 顶端非常接近:DeepSeek-V4-Pro-Max ~**80.6%** SWE-bench Verified、MiniMax-M3 ~**80.5%**、Qwen3.7-Max ~80.4% —— **差距 <0.2pp**。
- 但换变体对比时:V4-Pro **73.6%** vs M3 **80.5%** —— 说明**比的是哪个变体极其重要**,别只看一个数。

**综合 / agentic(某第三方汇总表,临时):**
- 综合分:M3 ~**79** vs V4 ~**68**
- 编码均分:M3 ~**67** vs ~58.8
- **agentic 任务 M3 优势最大**:~**71.9** vs 59.1

**价格:**
- V4-Pro ~$1.74 in / $3.48 out;M3 ~$0.30 in / $1.20 out(促销)—— **M3 输出便宜约 2.9×**。
- (注:不同来源给 V4 输出价 $0.87,口径/变体不一致,以官方为准。)

> 读法:**M3 在"性价比 + agentic + 编码"上很猛且便宜**;**V4 在绝对规模/长上下文/推理深度**上是另一档(1.6T MoE)。GLM-5.2 在这些 2026 SWE-bench 汇总里出现不多(榜单多还停在 GLM-5/5.1),但它的**MIT 开放权重**是独有杀手锏。

---

## 三、逐个看

### DeepSeek-V4(V4-Pro / V4-Flash)
- **创新**:沿用 DeepSeekMoE + 多 token 预测(MTP),换上 **CSA+HCA 混合注意力**让 1M 上下文可负担;32T+ token、多教师蒸馏。
- **性能**:开放权重前沿,长上下文/推理强;数字按厂商口径,待第三方复现。
- **RL-on-NPU**:✅ 910B 已支持(自定义算子、MTP KV-cache 分片做投机解码),稀疏注意力降 rollout 显存 —— 现成可做的 rollout/eval 目标。

### MiniMax-M3 + MSA ⭐(本轮性价比之王)
- **创新**:MSA = GQA 上块稀疏,index 分支选 per-group top-k 块,**在真实 KV 上算**。
- **性能**:109B 多模态,~28.4× 降 attention 计算(1M);**vs DeepSeek NSA 有 9.7× prefill / 15.6× decode 加速**(另一口径:vs full-attn 14.2× prefill / 7.6× decode @H800)。编码/agentic 榜单领先且**便宜约 3×**。
- **RL-on-NPU**:**最值得押注的移植对象** —— GQA 底座最易上 Ascend,decode 加速直接利好 rollout;只差 vLLM-Ascend 按名支持 M3(家族量化 W8A8/QuaRot 已有)。

### GLM-5.2
- **创新**:**MIT 全开放权重**、1M 上下文、coding-first;快速迭代(GLM-5→5.2 数月),明确对标出口管制下的开放路线。
- **性能**:称在多步推理/代码上对标闭源,数字多为厂商/媒体,独立 eval 待补;主流 SWE-bench 汇总里 5.2 尚未广泛出现。
- **RL-on-NPU**:GLM-5 已上 vLLM-Ascend(W8A8C8 量化),5.2 大概率近期跟进;**MIT 许可**使它成为**开放 RL 后训练实验的理想底座**。

---

## 四、对 RL-on-NPU 项目的结论

1. **首选移植/实验底座**:**MiniMax-M3**(GQA 最易上 Ascend、decode 快、便宜)+ **GLM-5.2**(MIT,适合做 RL 后训练 base)。DeepSeek-V4 适合做"已支持的大模型 rollout/eval 标杆"。
2. **稀疏注意力 = RL 显存解药**:这批模型的 decode 稀疏化,正面缓解 Ascend "无 sleep-mode、rollout/train 抢显存"的核心痛点 —— 值得在基准里专门量化"稀疏注意力对 rollout 显存/吞吐的影响"。
3. **精度对齐更重要了**:CSA/HCA/DSA/MSA 这些自定义注意力在 NPU 上重写 kernel,**数值漂移风险更高** —— 正是 `align-probe` 要诊断的对象;建议把这几种注意力的前向逐层 diff 纳入对齐套件。
4. **FP8 RL 窗口打开**:V4 已验证 FP8 训练、Ascend 950 原生 FP8 —— "在 Ascend 上做 FP8 RL"从空白变可行,是高新颖度课题。

---

## 来源(均为临时/二手,务必核对官方)
- MiniMax MSA 机制图解:https://huggingface.co/blog/AtlasCloud-AI/minimax-goes-sparse
- MSA vs DSA/CSA 对比(elie thread):https://x.com/eliebakouch/status/2059321928205156568
- DeepSeek V4 规格/基准:https://www.morphllm.com/deepseek-v4 · 官方权重 https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro
- 编码模型榜单(SWE-bench Pro):https://www.morphllm.com/best-ai-model-for-coding
- V4-Pro vs M3 对比:https://benchlm.ai/compare/deepseek-v4-pro-vs-minimax-m3
- DSA 实现(SGLang):https://shawnding.medium.com/deepseek-sparse-attention-and-its-implementation-in-sglang-b0bb907c375a
- MiniMax-M3 官方报告:https://arxiv.org/abs/2606.13392 · GLM-5 报告:https://arxiv.org/abs/2602.15763
