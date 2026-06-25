# Dispatch 11 · vLLM/Inferact vs SGLang/RadixArk:两家伯克利系推理公司的对决

*2026-06-25 · NPU Frontier Dispatch · inference / vLLM / SGLang / investment*

> **TL;DR** — 2026 年 1 月,两大开源推理引擎几乎同时商业化:**vLLM → Inferact**(**$800M 估值 / $150M 种子**,a16z + Lightspeed 领投,创始团队 Simon Mo、Woosuk Kwon、Kaichao You、Roger Wang + **Ion Stoica**,vLLM 跑在 **40 万+ GPU** 上);**SGLang → RadixArk**(**~$400M 估值 / $100M 种子**,Accel 领投,Intel CEO 陈立武天使)。两家都出自 **UC Berkeley** 系,瞄准同一个爆发的推理市场。差别:**vLLM 赢在生态广度、社区、硬件中立、资金**;**SGLang 赢在 prefix-heavy/agent 场景性能(RadixAttention)+ 自带 RL 训练框架 Miles**。对 RL-on-NPU:两个引擎都有 Ascend 后端,是 rollout 引擎选型的两个主选项;SGLang 的 RadixAttention + Miles 的 R3/FP8 让它在"RL 训练侧"叙事更完整,vLLM 则以 vLLM-Ascend 的广覆盖见长。

承接 Dispatch 09/10。应要求把 vLLM 这一侧也做进来,和 SGLang 正面对比。

> ⚠️ 估值/融资为媒体报道口径(provisional),非投资建议。

---

## 1 · 两家公司

| | **Inferact**(vLLM) | **RadixArk**(SGLang) |
|---|---|---|
| 引擎 | vLLM(40 万+ GPU) | SGLang(日均万亿 token) |
| 估值 / 轮次 | **$800M / $150M 种子** | **~$400M / $100M 种子** |
| 领投 | a16z + Lightspeed(+Sequoia/Altimeter/Redpoint/ZhenFund) | Accel(+Intel CEO Lip-Bu Tan 天使) |
| 创始 | Simon Mo、Woosuk Kwon、Kaichao You、Roger Wang + Ion Stoica | SGLang 核心团队(LMSYS 系) |
| 时间 | 2026-01 | 2026-01 spin-out / 2026-05 launch |
| 根 | UC Berkeley | UC Berkeley / LMSYS |

两者同源(伯克利)、同台(推理)、同期(2026 初),却走了**广度 vs 专精**两条路。

## 2 · 引擎之争:PagedAttention vs RadixAttention

承接 Dispatch 10:

- **vLLM / PagedAttention**:用**分页**管理 KV、解决显存碎片;前缀缓存是精确哈希命中(APC)。**通用、稳、生态最大**——支持的模型/硬件最广,是事实上的默认。
- **SGLang / RadixAttention**:用**基数树 + cache-aware 调度**做自动前缀复用,**多轮/agent/RAG/fork** 场景强。整体吞吐比充分优化的 vLLM 高 **~29%**,prefix-heavy 下可达 **6×**;唯一 prompt 批任务两者趋同。

一句话:**vLLM 是"什么都能跑得不错"的默认,SGLang 是"在前缀重度/agent 上更快"的专精。**

## 3 · 战略差异

| 维度 | vLLM / Inferact | SGLang / RadixArk |
|---|---|---|
| 核心叙事 | 最广的开源推理标准 | 最快的 agent/前缀场景 + **RL 训练(Miles)** |
| 硬件 | 中立、覆盖最广(含 vLLM-Ascend) | SGLang-Ascend 等,覆盖较窄但在追 |
| 训练侧 | 主打推理 | **同时握 RL 后训练(Miles:FP8 + R3)** |
| 生态/资金 | 更大、更厚 | 较小、更聚焦 |
| 团队牌面 | + Ion Stoica(Databricks 联创) | SGLang 原班 + Intel CEO 背书 |

最大的结构差异:**RadixArk 同时卡住"推理 + RL 训练"两个入口**(SGLang+Miles),而 Inferact 目前更纯粹是推理。这让 RadixArk 的故事在"前沿 MoE 后训练"这条线上更完整(见 Dispatch 09)。

## 4 · 谁赢哪块 / 会共存吗

- **生产部署默认**:vLLM(广度、社区、硬件)。
- **prefix-heavy / agent / 结构化输出 / RL rollout**:SGLang。
- **大多数团队 2026 年其实在两者之间二选一**——很多 RL 栈(AReaL、Miles)默认用 SGLang 做 rollout,而通用服务默认 vLLM。两者**长期共存**的概率高于一家通吃。

## 5 · 投资视角(非建议)

- **估值差**:Inferact $800M vs RadixArk $400M。前者反映更大社区/生态;后者**相对便宜**,且多一条 RL 训练腿。
- **看多 vLLM/Inferact**:事实标准、40 万 GPU 分发、a16z+Lightspeed+Ion Stoica 的牌面与渠道。
- **看多 SGLang/RadixArk**:性能领先的细分 + 推理&训练双入口 + 更低进入估值。
- **共同风险**:开源推理**商品化 + 变现难**;巨头(NVIDIA、云厂)夹击;价格战压毛利。
- **观察指标**:各自托管/企业版收入起量、相对采用份额、对非 NVIDIA 硬件(昇腾/AMD)的覆盖、SGLang 的 RL 训练(Miles)是否被头部实验室真实采用。

## 6 · 对 RL-on-NPU 的意义

- **rollout 引擎选型**:在昇腾上做 RL,rollout 引擎基本就是 **vLLM-Ascend** vs **SGLang-Ascend** 二选一。vLLM-Ascend 覆盖广、社区成熟;SGLang-Ascend 带 RadixAttention 的前缀复用红利(Dispatch 10)。
- **训练侧**:SGLang 这边有 **Miles 的 R3 + 端到端 FP8**(Dispatch 09)——这是"训推一致 + FP8 RL"在 GPU 上的完整答案;能否随 SGLang-Ascend 一起落到昇腾,是 RL-on-NPU 最值得追的工程线。
- 换句话说:**vLLM 给昇腾最广的 rollout 覆盖,SGLang/RadixArk 给昇腾最完整的 RL 训练蓝图**——两条都值得本看板持续跟踪。

---

*来源:TechCrunch / Bloomberg / SiliconANGLE / pulse2(Inferact $800M、$150M、a16z+Lightspeed、创始团队、Ion Stoica)、RadixArk 报道(Dispatch 09 来源)、SGLang vs vLLM 性能对比(Dispatch 10 来源)。估值/融资为媒体口径,provisional;非投资建议。*
