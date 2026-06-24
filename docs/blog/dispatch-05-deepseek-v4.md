# Dispatch 05 · 详解 DeepSeek-V4:混合注意力(CSA + HCA)与 1M 上下文

*2026-06-23 · NPU Frontier Dispatch · attention / MoE / DeepSeek-V4 / RL-on-NPU*

> **TL;DR** — DeepSeek-V4(2026-04-24,开源 MIT)是一对 MoE:**V4-Pro 1.6T / 49B 激活**、**V4-Flash 284B / 13B 激活**,都原生 **1M 上下文**。核心创新是**混合注意力**——逐层交替 **CSA(压缩稀疏注意力,KV 压 4× + 稀疏选块)** 与 **HCA(重压缩注意力,KV 压 128× + 不选、全看)**;再叠上 **mHC 超连接**、**Muon 优化器**、**MTP 多 token 预测**,以及 **FP4(专家)+ FP8(其余)混合精度**训练。效果:1M 上下文下,V4-Pro 单 token 推理只要 V3.2 的 **27% FLOPs、10% KV cache**;SWE-bench Verified **80.6%**(开源最强档)。对 RL-on-NPU 的意义:它**已经在 vLLM-Ascend 上跑**(910B,自定义算子 + MTP KV-cache 分片做投机解码),是昇腾上**现成的大模型 rollout/评测基线**;FP8/FP4 训练也正好印证了"FP8 RL on Ascend"这条线(见 Dispatch 02/03)。

接 Dispatch 04(MiniMax MSA)。同样应要求,把 4 月开源的 **DeepSeek-V4** 架构拆开讲——它和 MSA 是两条不同的长上下文路线,对比着看最清楚。

---

## 1 · 定位与规格

DeepSeek-V4 不是单个模型,是一对:

| | **V4-Pro** | **V4-Flash** |
|---|---|---|
| 总参数 / 激活 | 1.6T / **49B** | 284B / **13B** |
| 上下文 | 1M | 1M |
| 许可 | 开源 (MIT) | 开源 (MIT) |
| 输出价(参考) | ~$3.48 / 1M | **$0.87 / 1M** |
| 发布 | 2026-04-24 | 2026-04-24 |

延续 DeepSeekMoE + MLA 血统,但这一代真正的看点在**注意力**。

## 2 · 核心:混合注意力(Hybrid Attention)

V4 不再用单一注意力,而是**逐层交替**两种压缩注意力,各管一段职责:

**① CSA — Compressed Sparse Attention(压缩稀疏)**
- 沿序列维把 KV **压缩 4×**:用 **softmax-gated pooling(带可学习位置偏置)** 把相邻 KV 聚合成块。
- 然后做**稀疏选择**——query 只看选中的压缩块。
- 角色:在"还算精细"的粒度上保留检索/局部细节。

**② HCA — Heavily Compressed Attention(重压缩)**
- 把 KV **压缩 128×**——极度浓缩成很少的块。
- **完全放弃稀疏选择**:每个 query **稠密地看所有压缩块**。
- 角色:用极低成本提供"全局视野"——反正块已经少到可以全看。

**为什么要两种交替**:CSA 给"细节 + 选择性",HCA 给"便宜的全局上下文"。一层精挑、一层通览,合起来既省又不丢长程信息。这与 MSA(单一块稀疏、在真实未压缩 KV 上选 top-k)是**两条思路**:

| | **DeepSeek-V4(CSA+HCA)** | **MiniMax MSA** |
|---|---|---|
| KV | **压缩**(4× / 128×) | **不压**,真实 KV |
| 选择 | CSA 选块 / HCA 全看 | Index 分支选 top-k 块 |
| 风格 | 双机制交替、压得狠 | 单机制、复用标准 kernel |
| 取舍 | 长上下文内存最省;移植成本高 | 务实、易加速、易移植 |

## 3 · 其他架构件

- **mHC(Manifold-Constrained Hyper-Connections)**:对传统残差连接的增强版"超连接",约束在流形上以稳住深层信息流。
- **Muon 优化器**:更快收敛 + 更稳的训练(取代/补充 AdamW 一类)。
- **MTP(Multi-Token Prediction)**:保留多 token 预测模块——既提训练信号,又能在推理时做**投机解码**。
- **FP4 + FP8 混合精度**:**MoE 专家用 FP4、其余参数用 FP8** 训练——这是 V4 验证过的低精度训练栈,直接对应下一代昇腾 950 的原生 FP8/MXFP4。

## 4 · 训练 / 后训练:先分养专家,再蒸馏合并

V4 的后训练是**两阶段**:

1. **独立培养领域专家**:对不同领域分别做 **SFT + RL(GRPO)**,各练各的强项。
2. **统一合并**:用 **on-policy 蒸馏**把这些各有所长的专家**整合进一个模型**,跨领域能力收敛到单体。

(配合 32T+ token、多教师蒸馏的数据管线。)这套"分养—合并"思路,本身就是一个值得在昇腾上复现的 RL + 蒸馏流程。

## 5 · 效率与跑分(论文/厂商口径,provisional)

- **效率**:1M 上下文下,V4-Pro 单 token 推理仅需 V3.2 的 **27% FLOPs** 和 **10% KV cache**——混合注意力把长上下文成本砍到约 1/4~1/10。
- **质量**:V4-Pro(Max)**SWE-bench Verified 80.6%**,开源最强、与 Gemini 3.1 Pro 持平;LiveCodeBench ~93.5%。
- **价格**:V4-Flash 输出 **$0.87/1M** —— 开源里很有竞争力。

## 6 · 对 RL-on-NPU 的意义

为什么 V4 是本看板反复提到的"基线":

- **已经能在昇腾上跑**。vLLM-Ascend 的 2026 支持矩阵里 V4 在 **910B** 上有自定义算子 + **MTP 层 KV-cache 分片做投机解码**——这意味着它是**现成的大模型 rollout / 评测对象**,不用等移植。
- **FP8/FP4 训练 = "FP8 RL on Ascend"的实证**。V4 把低精度训练跑通了,叠上 950 的原生 FP8(Dispatch 03),让"在昇腾上做 FP8 RL"从设想更近一步。
- **KV cache 砍到 10% = 直接缓解显存争用**。这正是 910B 上 rollout/train 抢 64GB HBM 的痛点(NPU 架构页"RL 显存争用"视图)。
- **但混合注意力是移植风险点**。CSA/HCA 这种压缩 + 稀疏的自定义注意力,在 NPU 上重写后容易引入 train-inference 数值漂移——这正是 **align-probe** 想法该量化的。

## 7 · 下一步看什么

1. **CSA/HCA 的 Ascend kernel 成熟度**:压缩注意力在 910B/950 上的真实吞吐与数值一致性。
2. **MTP 投机解码在 NPU 上的加速比**:对 RL rollout 的端到端收益。
3. **V4 vs MSA 的长上下文质量对照**:压缩(V4)与不压缩选块(MSA)在多跳推理 / 长检索上谁更稳。

---

*来源:DeepSeek-V4 技术报告与解析(HuggingFace deepseek-ai/DeepSeek-V4-Pro、DeepSeek API Docs、latent.space、morphllm、techjacksolutions 等);vLLM-Ascend 2026 支持矩阵。规格 / 跑分为论文 / 厂商口径,provisional。相关卡片见本看板 LLM Modeling 标签页。*
