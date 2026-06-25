# Dispatch 10 · RadixAttention 详解:SGLang 怎么靠"前缀树"赢 KV 复用

*2026-06-25 · NPU Frontier Dispatch · inference / SGLang / KV-cache / RL-on-NPU*

> **TL;DR** — RadixAttention 是 SGLang 的核心(arXiv 2312.07104):把 KV cache 组织成一棵 **基数树(radix tree)**,对每个新请求做**前缀匹配**,命中就**复用已算的 KV**,不重算。它**自动、动态**地从真实流量里发现可共享的前缀(零配置),用 **LRU** 淘汰、用 **cache-aware scheduling** 调度请求以最大化命中,并能优雅处理**对话分叉(fork)**。对比 vLLM 的 PagedAttention(解决显存碎片 + 精确前缀哈希复用),RadixAttention 在**多轮、agent、RAG、共享 system prompt、树搜索**这些"前缀重度"场景上优势最大——prefix-heavy 下可比 vLLM 快约 **29%~6×**。对 RL:**GRPO 一个 prompt 采 N 条样本,它们共享同一前缀** → RadixAttention 直接复用,这是 rollout 引擎最实在的省法;移植到昇腾的 SGLang-Ascend 后,同样能缓解 910B 的 rollout 显存。

承接 Dispatch 09(RadixArk/Miles)。应要求把 SGLang 的看家本领 **RadixAttention** 拆开讲。

---

## 1 · 问题:前缀被反复重算

LLM 推理里,**很多请求共享前缀**:同一套 system prompt、同一段 few-shot、多轮对话的历史、RAG 里相同的文档、agent 群里相同的工具说明。朴素做法每来一个请求都从头算一遍这些前缀的 KV——**纯浪费**。

省法就是**前缀缓存(prefix caching)**:把算过的 KV 留着,前缀相同就复用。难点在于——**怎么高效地"发现 + 匹配 + 淘汰"任意的共享前缀**。

## 2 · RadixAttention 怎么做

核心是一棵 **基数树(radix tree,压缩前缀字典树)**:

- **节点 = 一段 token 序列 → 它的 KV**。从根到某节点的路径,就是一个被缓存的前缀。
- **新请求来 → 沿树做最长前缀匹配**:匹配到的部分**直接复用 KV**,只对剩下的新 token 算注意力。
- **自动 / 动态发现**:不需要你预先声明"哪些前缀会复用",树会**自然捕捉真实流量里实际存在的共享**,零配置。
- **LRU 淘汰**:显存不够时,按最近最少使用从树上驱逐叶子。
- **Cache-aware scheduling**:调度器**有意把"能命中缓存"的请求排在一起**,把命中率最大化(这是和单纯加个缓存的关键区别)。
- **优雅处理 fork**:对话/推理分叉时,多个分支**共享公共前缀**,各自只算分叉后的部分——对 **best-of-N、树搜索、agent 探索**极友好。

一句话:**它把"前缀复用"从'精确字符串命中'升级成'一棵会自我组织的树 + 会配合的调度器'。**

## 3 · vs PagedAttention(其实互补)

| | **RadixAttention(SGLang)** | **PagedAttention(vLLM)** |
|---|---|---|
| 解决的核心 | **前缀复用**(KV 共享) | **显存碎片**(分页管理 KV) |
| 结构 | 基数树 + cache-aware 调度 | 固定大小 KV 分页(类虚拟内存) |
| 前缀缓存 | 自动/动态发现任意前缀 | APC:精确前缀哈希命中 |
| 强项场景 | 多轮 / agent / RAG / 共享 prompt / fork | 通用、可预测、可配置负载 |

注意:**两者不是非此即彼**——SGLang 也做分页式显存管理,vLLM 也有自动前缀缓存(APC)。真正的差异在 **radix tree 的结构 + cache-aware 调度**,让它在"复杂、不可预测的前缀复用"上更强。所以基准上:唯一 prompt 批任务两者趋同;**prefix-heavy RAG 上 SGLang 可快到 6×**。

## 4 · 为什么对 RL / agent 特别重要

这是本看板最关心的角度:

- **GRPO/RLVR 的 rollout 天生前缀重度**。一个 prompt 要采 **N 条**样本(group),它们**共享完全相同的 prompt 前缀**。RadixAttention 把这段前缀**只算一次、N 条复用** → rollout 的显存和时延直接降。
- **树搜索 / best-of-N / 多轮 agent**:fork 共享前缀,探索不同分支不重算——正是 agentic RL(Dispatch 08)长 rollout 的省钱点。
- **agent 群共享 system/工具说明**:几十个 agent 同一套 system prompt → 零成本共享。

这也是为什么 **SGLang 成了很多 RL 栈的首选 rollout 引擎**(AReaL、Miles 等),以及 RadixArk 能同时做推理 + RL 两端(Dispatch 09)。

## 5 · 对 RL-on-NPU 的意义

- **rollout 引擎的前缀复用 = 直接缓解昇腾显存争用**。910B 单卡 64GB、无 sleep-mode,rollout 的 KV 是大头;GRPO 组内共享前缀复用,能实打实降低 rollout 峰值显存(见 NPU 架构页"RL 显存争用"视图)。
- **SGLang 有 Ascend 后端**(本看板 Ascend 标签已收录 docs.sglang.ai)——所以 RadixAttention 的收益**原则上可带到 NPU**;但 radix tree 的 KV 管理 + cache-aware 调度在 Ascend 上的成熟度、与 MindSpeed-RL 的整合度,是要验证的点。
- 叠加 **量化 rollout(FP8)+ 稀疏注意力**(Dispatch 02/04),前缀复用是"省 KV"的第三条正交杠杆。

## 6 · 下一步看什么

1. **SGLang-Ascend 上 RadixAttention 的真实命中率/吞吐**,对照 vLLM-Ascend。
2. **RL rollout 里 group 共享前缀的实测收益**(N 越大省得越多)。
3. **RadixAttention × 稀疏注意力**:前缀复用与块稀疏/滑窗能否叠加生效。

---

*来源:SGLang 论文(arXiv 2312.07104)、SGLang 仓库与文档、RadixAttention vs PagedAttention 解析(rajatpandit / inference.net / particula 等)。性能数字为第三方/媒体口径,provisional。相关卡片见本看板 Ascend / NPU 与 RL for LLMs 标签页。*
