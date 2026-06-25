# Dispatch 09 · RadixArk / Miles:SGLang 团队的 RL 框架,与投资视角

*2026-06-25 · NPU Frontier Dispatch · infra / SGLang / Miles / investment*

> **TL;DR** — RadixArk 是 **SGLang 核心团队**的商业化公司:2026 年 1 月从开源项目 SGLang spin-out,**~$400M 估值**(Accel 领投),5 月宣布 **$100M 种子轮**,天使含 **Intel CEO 陈立武(Lip-Bu Tan)**。两条产品线:推理引擎 **SGLang**(核心是 **RadixAttention** —— 基数树自动 KV 前缀复用,prefix-heavy/agent 场景比 vLLM 快约 29%~6×)+ RL 训练框架 **Miles**(从智谱 **slime** fork、共演化)。Miles 的杀手锏是 **统一 FP8 流水线**(首个端到端 FP8 采样+训练)和 **Rollout Routing Replay(R3)**——把 MoE 专家路由在 SGLang 推理时记录、Megatron 训练时回放,做到**逐比特对齐**,根治 MoE RL 的 train-inference mismatch。技术壁垒在**人才 + SGLang 分发量 + R3/FP8 的硬工程 + 开源飞轮**;主要风险是开源基础设施的变现未验证、vLLM(Inferact,$800M)生态更大。对本看板:**R3 正是"train-inference 一致性"在 GPU 上的标准答案**,而它能否移植到昇腾(SGLang-Ascend + MindSpeed),是 RL-on-NPU 的一道关键缺口。

> ⚠️ 以下为公开资料的研究综述,估值/融资为媒体报道口径(provisional),**非投资建议**。

应要求研究一下这家公司。它正好踩在本看板的核心主线(RL 后训练 + 训推一致 + FP8)上,所以单开一期。

---

## 1 · 公司:把 SGLang 变成生意

- **出身**:SGLang 源自 LMSYS / 伯克利系开源,是"日均处理万亿 token"的推理引擎。2026-01 项目 spin-out 成 **RadixArk**,~**$400M 估值**(TechCrunch)。
- **融资**:2026-05 宣布 **$100M 种子轮**,**Accel 领投**,天使含 **Intel CEO Lip-Bu Tan**。
- **使命**:"Democratize Frontier AI Infrastructure" / "Ship AI for All" —— 做**开放的前沿 AI 基础设施**。
- **时机**:HuggingFace 的 TGI 在 2025-12 进入维护模式,SGLang 顺势成为主要开源替代;推理市场被各方称为"battleground"。
- **两条腿**:**推理 = SGLang**,**RL 训练 = Miles**。一家公司同时卡住"推理"和"RL 后训练"两个基础设施入口。

## 2 · 技术优势

### SGLang(推理护城河)
- **RadixAttention**:用**基数树(radix tree)**维护 KV cache 的 LRU,新请求来时做**前缀匹配**,命中就复用已算的 KV,不重算。对 **prefix-heavy / RAG / 多轮 agent** 工作负载收益巨大。
- **性能**:整体吞吐比"充分优化的 vLLM"高约 **29%**(16.2k vs 12.5k tok/s);在唯一 prompt 批任务上差距趋零,在 prefix-heavy RAG 上可放大到 **6×**。
- 定位:vLLM 赢在生态广度/社区/易用;SGLang 赢在**前缀缓存、结构化输出、agent 工作流**。

### Miles(RL 训练框架)—— 本看板更关心的部分
栈 = **SGLang(rollout)+ Megatron-LM(训练)**,从 **slime**(智谱那套、跑过 GLM-4.6 与大 MoE)fork 而来。核心创新:

- **统一 FP8 流水线**:**首个端到端 FP8 采样 + 训练**。让训练和推理用**完全相同的 FP8 量化逻辑**,消除 MoE RL 里量化不一致导致的"RL 崩溃"。
- **Rollout Routing Replay(R3)**:**在 SGLang 推理时记录 MoE 的专家路由决策,在 Megatron 训练时回放** → 专家选择**逐比特对齐** → 根治 MoE 的 train-inference mismatch。这是 Miles 最硬的一招。
- **INT4 QAT**:量化感知训练,让 **1TB+ 模型**单机可部署。
- **在线 draft 模型投机解码**:rollout 提速 **25%+**。
- **零拷贝权重同步**(CUDA IPC)、**partial rollout**(多轮)、**截断+掩码重要性采样**(治 off-policy 偏差)、**多智能体协同训练**、**VLM+LLM 统一**。
- 支持 DeepSeek(R1/V3/V3.2)、Qwen、Llama、Gemma、GLM。

## 3 · 同行对比

**推理引擎**

| | **SGLang** | vLLM | TGI |
|---|---|---|---|
| 杀手锏 | RadixAttention 前缀复用 | 生态广、PagedAttention | 已进维护模式(2025-12) |
| 强项场景 | prefix-heavy / RAG / agent | 通用、社区最大 | — |
| 商业体 | **RadixArk ~$400M** | Inferact ~$800M | (HF) |

**RL 后训练框架**

| 框架 | 出身 | MoE/EP | FP8 | 差异点 |
|---|---|---|---|---|
| **Miles** | RadixArk(SGLang) | ✅ Megatron | ✅ 端到端 + **R3** | SGLang 原生 rollout、R3 逐比特对齐 |
| slime | 智谱 THUDM | ✅ | 部分 | Miles 的上游 |
| verl | 字节(HybridFlow) | ✅ | 进行中 | 最广用、引擎中立 |
| NeMo-RL | NVIDIA | ✅ | ✅(NeMo 生态) | 绑 NVIDIA 栈 |
| OpenRLHF | 社区 | 较弱 | — | 易用、起步早 |

> 关键判断:**只有 Megatron 系(verl/slime/Miles/ROLL/NeMo-RL)能正确做专家并行(EP)**——而前沿模型几乎都是稀疏 MoE(DeepSeek-V3/Qwen3-MoE/…),这让 Miles 的 MoE 专长正中靶心。**R3 + 统一 FP8 是 Miles 相对 verl/slime 的真差异化**;NeMo-RL 也有 FP8,但绑死 NVIDIA。

## 4 · 技术壁垒与风险

**壁垒**
1. **人才**:做出 RadixAttention 的 SGLang 核心团队,系统工程深度难复制。
2. **分发量 = 护城河**:SGLang 日均万亿 token、是众多 RL 栈的 rollout 引擎;Miles **SGLang 原生**的紧耦合是别家给不了的。
3. **R3 / 统一 FP8 的硬工程**:让 MoE 训推**逐比特一致**是真难的活,先发优势明显。
4. **开源飞轮**:开放 SGLang+Miles → 采用 → 企业版变现(Databricks / vLLM-Inferact 同款打法)。

**风险**
- **开源基础设施变现未验证**:能不能从"免费引擎"转成"赚钱企业产品"是关键问号。
- **vLLM 生态更大、融资更多**(Inferact $800M):社区与广度上 SGLang 仍追赶。
- **巨头夹击**:NVIDIA(NeMo-RL + 硬件捆绑)、各云厂自带推理服务、价格战导致推理商品化。
- **栈依赖**:深度绑 Megatron / NVIDIA;**对国产 NPU(昇腾)的支持是另一条战线**。
- 团队/项目的开源中国渊源(slime/THUDM 共演化),在部分企业/政府采购上可能成为考量。

## 5 · 投资视角(非建议)

- **看多**:推理是当下最炸的市场;SGLang 是开源前二的引擎;顶级团队 + Accel + Intel CEO 背书;**$400M 估值相对 vLLM-Inferact 的 $800M 显得不贵**;同时握住推理 + RL 训练两个入口;MoE 后训练是前沿方向而 Miles 在此最强。
- **看空**:基础设施开源变现难、毛利与议价权存疑;vLLM 生态/资金更厚;推理趋于商品化;高度依赖 NVIDIA 栈。
- **可比公司**:vLLM→Inferact($800M)、Together / Fireworks / Baseten(推理云)、Databricks(开源→企业的范式)。
- **观察指标**:企业版/托管收入起量、SGLang 相对 vLLM 的采用份额、Miles 被头部实验室用于真实 MoE 后训练的案例、对非 NVIDIA 硬件(昇腾/AMD)的支持进展。

## 6 · 对 RL-on-NPU 的意义

这家公司和本看板的主线高度重合,有两个直接启发:

- **R3 就是"训推一致"在 GPU 上的标准答案**。本看板反复提的 align-probe,想量化的正是 train-inference 的数值/路由漂移;Miles 的 R3 直接用"记录-回放路由"做到逐比特对齐——**这是昇腾上做 MoE RL 时最该借鉴的机制**(在 NPU 上重写 MoE 路由极易漂移)。
- **统一 FP8** 与 Dispatch 02/03 的 FP8 RL、昇腾 950 原生 FP8 完全同向。问题是:**SGLang 的 Ascend 后端 + MindSpeed,能否复刻 Miles 的 R3 / 端到端 FP8?** 这是 RL-on-NPU 一道明确、可做的工程缺口——谁先在昇腾上跑通"路由回放 + FP8 一致"的 MoE RL,谁就补上了最硬的一块。

---

*来源:RadixArk 官网/博客、TechCrunch / BusinessWire / theaiinsider(spinout、$400M、$100M 种子、Accel、Lip-Bu Tan)、LMSYS Miles 博客(2025-11-19)、github.com/radixark/miles 与 miles.radixark.com/docs、SGLang vs vLLM 对比(turion/particula/yottalabs 等)、HuggingFace async-RL 库综述。估值/融资为媒体报道口径,provisional;本篇为研究综述,非投资建议。*
