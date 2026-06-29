# Dispatch 16 · DeepSeek-V4 的 Agent 是怎么开发的:两阶段专家培养 + on-policy 蒸馏

*2026-06-26 · NPU Frontier Dispatch · DeepSeek-V4 / agentic / 后训练 · 多 agent 工作流产出*

> **TL;DR** — DeepSeek-V4 的 agent 能力来自一套**两阶段后训练**:**Stage 1** 先**独立**培养 10+ 个领域专家(数学、竞赛代码、**agentic/工具使用**、指令跟随…),每个先 SFT、再用 **GRPO + 领域奖励模型/可验证奖励**精炼——**agent/工具专家是单独对着真实工具环境 RL 训出来的**,不和数学/代码抢容量;**Stage 2** 用 **on-policy 蒸馏(OPD)** 把这些专家合并成一个学生模型(学生自己产 rollout、学专家反馈),取代了 V3.2 的混合 RL 合并。配套**工具机制**:跨轮保留 reasoning、`|DSML|` 的 **XML tool-call 格式**(比 JSON-in-string 少转义错误)、strict-mode function calling(Anthropic/OpenAI 兼容)。Agent 跑分(均**厂商口径、未独立复现**):SWE-bench Verified **80.6**、Terminal-Bench 2.0 **67.9**、MCPAtlas **73.6**、Toolathlon **51.8**。**目前没有自家 agent harness 产品**(在招「Harness 团队」,口号 Model + Harness = Agent),现状是**插进 Claude Code / OpenCode** 用 Anthropic 兼容 API 跑。服务侧 **MTP → DSpark** 让长 agentic rollout 快(见 Dispatch 15)。

> 🤖 本期由 5 路并行子 agent 调研 + 综合产出。**溯源说明**:arxiv/HF/DeepSeek 文档在本环境被 403 拦,唯一能逐字读到的一手是 HF 发布博客的 GitHub raw 镜像;其余靠搜索片段 + 二手聚合。**所有跑分与内部机制细节均 provisional,以官方 [V4 技术报告 arXiv 2606.19348](https://arxiv.org/abs/2606.19348) 为准。**

你问的"DeepSeek-V4 关于 Agent 的部分怎么开发的"——这期专门回答。

---

## 1. 两阶段后训练(agent 能力的来源)

**Stage 1 · 独立培养领域专家**(confirmed-as-reported):
- 训 **10+ 个专才模型**,一个领域一个——数学、竞赛代码、**agentic 任务**、指令跟随等。
- 每个专家:先在高质量领域数据上 **SFT**,再用 **GRPO** 强化,奖励来自**领域专属奖励模型 / 可验证奖励**(代码用测试套件判对错)。
- **关键**:**agent / 工具使用能力是作为独立专才、对着真实工具环境单独 RL 训出来的**——不与数学/代码专家竞争模型容量,避免联合多任务 RL 常见的能力打架。

**Stage 2 · on-policy 蒸馏(OPD)合并**(confirmed-as-reported):
- 把 10+ 专家**蒸馏进一个学生模型**:学生**生成自己的 on-policy rollout**,从教师(专家)对这些输出的反馈中学。
- 这取代了 V3.2 的"混合 RL 合并"阶段。

**provisional(仅二手)**:OPD 用**全词表 reverse-KL** 稳住专家分歧时的合并;专家数"十几个";对工具使用/长程这类难验证任务用 **GRM(生成式奖励模型,actor 当自己的裁判)**。

> 一句话:**先把"会用工具的 agent"当成一个专才 RL 练好,再 on-policy 蒸馏灌进出货模型。**

## 2. 工具使用机制

(confirmed-as-reported,主要来自 HF 博客 + API 文档)
- **跨轮保留 reasoning**:对话含 tool call 时,V4 在用户消息边界间**保留推理内容**——对长多步循环很关键。
- **`|DSML|` XML tool-call 格式**:新的特殊 token + XML 工具调用格式,把字符串参数与结构化(JSON)参数分开,**比"JSON 塞字符串"少转义失败**。
- **strict-mode function calling**:严格遵循函数 JSON schema;Anthropic & OpenAI 兼容;thinking / 非 thinking 模式都支持。
- RL on **可验证奖励**(代码用测试判)在 DeepSeek 谱系是明确的,归到 V4 agentic 训练多为二手转述(把"RLVR-for-V4"当**推断**看)。

## 3. Agent 跑分(均 V4-Pro-Max,**provisional / 厂商口径,无独立复现**)

| 基准 | 分数 | 备注 |
|---|---|---|
| SWE-bench Verified | **80.6** resolved | ≈ Opus-4.6-Max(80.8)/ Gemini-3.1-Pro(80.6) |
| Terminal-Bench 2.0 | **67.9** | 落后 GPT-5.4-xHigh(75.1) |
| MCPAtlas(Public) | **73.6** | 仅次于 Opus-4.6-Max(73.8) |
| Toolathlon(工具使用) | **51.8** | 落后 GPT-5.4(54.6) |
| SWE-bench Pro | 55.4 | 二手、来源较弱 |

**没找到 V4 的**(可能官方没报):**BFCL**(榜上还是 V3/R1)、**经典 τ-bench / τ²-bench**(流传的 τ² 数字其实是 **V3.2**,别张冠李戴)、SWE-bench Multimodal。

## 4. 有没有"DeepSeek-V4 Agent"产品 / Claude-Code 式 harness?

**截至 2026-06,没有自家 agent CLI/harness 产品。**
- **在招**:DeepSeek 在组「**Harness 团队**」(北京,Harness PM + 研发),口号 **「Model + Harness = Agent」**——还没发布。(SCMP / Yicai)
- **现状用法**(confirmed-as-reported):V4 作为**模型直接插进第三方 harness**——Anthropic 兼容端点 `ANTHROPIC_BASE_URL=https://api.deepseek.com/anthropic`,主模型 `deepseek-v4-pro` / 子 agent `deepseek-v4-flash`;官方给了 **Claude Code** 和 **OpenCode** 的接入指南。
- ⚠️ 叫 "Deep Code" / "deepcode-cli" / "DeepSeek-TUI" 的是**第三方**,不是 DeepSeek 官方。

## 5. 服务侧:MTP → DSpark 让长 agentic rollout 快

- **MTP**:V4 保留 V3 的 **Multi-Token Prediction**(1 个 MTP 头),既提训练效率,又在推理时做投机解码草稿——这就是 **MTP-1** 基线。
- **DSpark**(已开源,见 [DeepSpec](https://github.com/deepseek-ai/DeepSpec) + Dispatch 15):半自回归 + 置信度调度,vs MTP-1 单用户 **+60–85%(Flash)/+57–78%(Pro)**。
- **为什么对 agent 重要**:agentic 负载(长工具循环、流式、RL rollout、批量评测)是**解码吞吐受限**的,1.5–5× 的吞吐摆动盖过边际跑分提升。⚠️ 但接受率"在 OOD prompt 上会塌",真实 agent 轨迹的加速可能低于头条数字。

## 6. 对 RL-on-NPU 的意义

- **EP 在 GPU + 昇腾双验证**(paper 级 confirmed-as-reported):V4 的专家并行(Expert-Parallel)方案**在 NVIDIA GPU 和华为 Ascend NPU 上都验证过**,fused MoE kernel 号称在"**RL rollout 和高速 agent 服务**"等延迟敏感场景 **1.96× 加速**——这是最硬的一手 NPU 关联,但属于 **rollout/推理侧**,不等于梯度 RL 训练跑在昇腾。
- **"V4 的 RL 训练跑在 Ascend 950PR"= 传闻**(有源称 NVIDIA 预训练 / Ascend 做 RL 的拆分),**未一手证实**。
- 生态有 **MindSpeed-RL、slime-ascend(slime 的 NPU 适配)**,但**没证据表明 V4 用了其中哪个**——是"生态在成熟",不是"V4 用了 X"。
- **开放**:开源权重(HF,MIT)、Day-0 的 SGLang/Miles 推理+RL 服务、Day-0 NPU 推理(vllm-ascend),DSpark/DeepSpec MIT 开源。

## 结论与最大不确定性

V4 的 agent 能力 = **① agent/工具专才 RL(GRPO + 可验证奖励 + 真实工具环境)→ on-policy 蒸馏进出货模型;② 工具机制(跨轮 reasoning + `|DSML|` XML 格式 + strict function calling);③ 强但厂商口径的 agent 分;④ DSpark 让长 rollout 快;⑤ 昇腾验证过的 rollout/服务路径 + MIT 开放生态**。**没有自家 harness,只有招聘**——今天要把 V4 当 agent 用,就插进 Claude Code / OpenCode。
**最大未确认项**:所有跑分、`|DSML|`/reverse-KL/GRM 内部细节都没核对过实际报告 PDF(403);V4 的 BFCL / τ-bench 似乎不存在;"在昇腾上做 RL 训练"仍是传闻。

---

*由 multi-agent workflow 产出。来源:DeepSeek-V4 HF 发布博客(via raw 镜像)与 HF 模型卡、DeepSeek API 文档(function calling / coding agents)、V4 技术报告 arXiv 2606.19348、DeepSpec 仓库,以及 SCMP/Yicai/Fireworks/Phemex/morphllm/explainx 等二手。跑分与内部机制 provisional。*
