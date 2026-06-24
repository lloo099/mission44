# Dispatch 04 · 详解 MiniMax MSA:在真实 KV 上做"块稀疏"注意力

*2026-06-23 · NPU Frontier Dispatch · attention / sparse / MiniMax M3 / RL-on-NPU*

> **TL;DR** — MSA(MiniMax Sparse Attention,arXiv 2606.13392)是 MiniMax-M3 的核心:在标准 **GQA** 主干上做**块级稀疏**——不压缩 KV,而是用一个轻量 **Index 分支**为每个 query 组挑出 **top-k 个 KV 块**,再用 **Main 分支**只在这些块上跑精确 softmax。默认块大小 `Bk=128`、每组选 `k=16` 块 → **每个 query 固定只看 2048 个 KV token**,与上下文长度**无关**(把 O(n²) 压成近 O(n))。在 109B MoE 上**质量持平 GQA**,1M 上下文**每 token 注意力算力降 28.4×**,配套 kernel 后 H800 上 **prefill 14.2× / decode 7.6×** 提速。对 RL-on-NPU 的关键点:它跑在**未压缩的真实 KV** 上、复用标准注意力 kernel,是**最好往昇腾移植**的一档稀疏注意力,而且 decode 端的大提速正好压在 RL rollout 的痛点上。

接 Dispatch 03(昇腾 950)。这期应要求,把 6 月最受关注的注意力机制 **MSA** 拆开讲清楚:它解决什么、怎么工作、怎么训、和 DSA/MLA/NSA 有什么不同。

---

## 1 · 背景:MiniMax 为什么"绕了一圈又回来"

MiniMax 的注意力路线很有故事:

- **M1 / MiniMax-01**:押注 **Lightning Attention**(线性/次二次注意力),想用线性复杂度换长上下文。
- **M2**:规模做大后发现,**线性 / 滑窗注意力严重损伤"多跳推理"**——跨长文档把分散线索串起来的能力。团队只好**退回完整二次注意力**,硬扛算力成本来保住前沿智能。
- **M3 → MSA**:既不想要线性注意力的推理缺陷,又不想吃满二次注意力的成本,于是走**稀疏注意力**这条中间路——只在"该看的地方"做精确注意力。

一句话:**MSA 是 MiniMax 在"线性太笨、全注意力太贵"之间找的第三条路。**

## 2 · MSA 怎么工作:两个分支

MSA 把注意力拆成两段,跑在普通 **GQA** 主干上(不像 DeepSeek MLA 把 KV 压进低维潜空间——MSA 用的是**真实、未压缩的 KV**):

**① Index 分支(选块)**
- 把 KV 序列切成大小 `Bk=128` 的**块**。
- 对每个注意力**组(GQA group)**,用 **max-pooling 打分**给每个 KV 块算一个相关度,选 **top-k(默认 k=16)** 个块。
- **永远保留最近的那个块**(保证局部性 + 训练稳定)。

**② Main 分支(算注意力)**
- 只在 Index 分支选出的那 k 个块上,跑**精确的 softmax 注意力**。

**为什么这是 O(n) 而不是 O(n²)**:每个 query 的预算被**钉死**在 `k·Bk = 16·128 = 2048` 个 KV token——无论上下文是 8K 还是 1M,单 query 看的 KV 量不变。上下文越长,省得越多(1M 时 ~28×)。

| 参数 | 默认值 | 含义 |
|---|---|---|
| 块大小 `Bk` | 128 token | KV 按块切分的粒度 |
| 每组选块数 `k` | 16 | 每个 query/组保留的块数 |
| 每 query 预算 `k·Bk` | **2048 token** | 固定,与上下文长度无关 |
| 主干 | GQA | 在真实 KV 上选块(非 MLA 压缩) |
| 选择粒度 | 块级(非 token 级) | 复用块稀疏 kernel,更易加速 |

## 3 · 怎么训练:top-k 不可导,用 KL 对齐救

这是 MSA 最巧的一点。**top-k 块选择是不可导的**——语言建模损失的梯度传不到 Index 分支的投影参数上,Index 分支学不会"该选哪些块"。

MSA 的解法:**KL 对齐损失(KL alignment loss)**——让 **Index 分支打分出来的块分布**去对齐 **Main 分支真实的注意力分布**。也就是说,用 Main 分支"实际看了哪里"当老师,反过来监督 Index 分支"应该选哪里"。再加上"永远保留最近块"兜底,训练就稳了。

> 训练规模:在一个 **109B 参数的 MoE** 上做了**原生多模态**训练,token 预算约 **3T**。

## 4 · 和别家稀疏 / 压缩注意力比

2025–26 各家在"让长上下文变便宜"上各走各路,关键差异在**压不压 KV、选 token 还是选块**:

| 方案 | 主干 | 机制 | 取舍 |
|---|---|---|---|
| **MLA**(DeepSeek V3) | — | 把 KV 压成低维潜向量 | 省显存;但要专门 kernel |
| **DSA**(DeepSeek V3.2) | MLA | lightning indexer → **token 级** top-k | 质量稳;kernel 重 |
| **NSA**(早期) | GQA | 三分支(压缩/选择/滑窗)+ 门控 | 上限高;最复杂 |
| **MoBA** | — | **块级**选择 | 思路接近 MSA |
| **MSA**(MiniMax M3) | GQA | Index 分支选**块** + Main 分支在**真实 KV** 上精确注意力 | **务实**——复用现有 kernel、对齐损失可训、易加速 |

MSA 的差异化:**不压 KV、按块选、跑在真实 KV 上**。代价是块级比 token 级粗一点,但换来工程上的简单——这恰恰是它能快速落地、也最好移植的原因。

## 5 · 性能数字(论文口径)

- **质量**:109B 模型上 **与 GQA 持平**(没有线性注意力那种推理掉点)。
- **算力**:1M 上下文下,**每 token 注意力算力降 28.4×**。
- **墙钟提速**(配套 co-designed kernel,H800):**prefill 14.2× / decode 7.6×**。
- (厂商早期 teaser 给过 ~20× 算力、>9× prefill、>15× decode 的口径,以论文 28.4/14.2/7.6 为准;均 provisional。)

## 6 · 对 RL-on-NPU 的意义

为什么本看板特别看重 MSA:

- **最好移植到昇腾**。MSA 跑在**普通 GQA + 未压缩 KV** 上,Main 分支就是标准块稀疏注意力,能**复用现有 kernel**;NPU 上真正要新写的只是那个轻量 **Index 选块 kernel**。相比 MLA/DSA 要重写一整套压缩-注意力路径,MSA 的移植面小得多。
- **decode 大提速正中 RL 痛点**。RL 的 rollout 是 decode-heavy 且 memory-bound;MSA 的 **7.6× decode** 与"每 query 只看 2048 token"直接压低 KV 访存,缓解昇腾"无 sleep-mode"的显存争用(见 NPU 架构页的"RL 显存争用"视图)。
- **要盯数值一致性**。块选择 + KL 对齐 + 在 NPU 上重写 Index kernel,会引入新的 train-inference mismatch 风险——这正是看板 **align-probe** 想法该量化的:NPU 上 MSA 的选块是否和 GPU 训练时一致。
- **现状**:vLLM-Ascend 已有 MiniMax 系(M2.x)的 W8A8/QuaRot,但 **M3 尚未按名列入**——MSA 的 Ascend 落地是个明确、可做的工程缺口。

## 7 · 下一步看什么

1. **MSA 的 Ascend kernel**:谁先把 Index 选块 + 块稀疏 Main 分支在 910B/950 上跑通并公布吞吐。
2. **块级 vs token 级的质量差**:MSA(块)与 DSA(token)在长上下文检索 / 多跳推理上的真实差距。
3. **MSA + FP8**:把 Dispatch 02/03 的 FP8 rollout 叠到 MSA 上,decode 端还能再省多少。

---

*来源:MiniMax Sparse Attention(arXiv 2606.13392)及其解析(MarkTechPost、HuggingFace、Medium/artgor 等);MiniMax-01 / M1 Lightning Attention 背景(arXiv 2501.08313 / 2506.13585)。数字为论文/厂商口径,provisional。相关卡片见本看板 LLM Modeling 标签页。*
