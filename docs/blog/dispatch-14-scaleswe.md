# Dispatch 14 · 详解 ScaleSWE:把 GitHub 宇宙"挖"成 SWE 训练数据

*2026-06-26 · NPU Frontier Dispatch · SWE / 训练数据 / 蒸馏 · 多 agent 工作流产出*

> **TL;DR** — **ScaleSWE**(AweAI-Team,人大 RUC 关联,arXiv 2602.09892,论文《Immersion in the GitHub Universe: Scaling Coding Agents to Mastery》)是一个**大规模 SWE 训练数据 + 智能体**项目,**不是 benchmark,也和 Scale AI 无关(只是名字撞车)**。三件套:**真实 PR 大规模挖掘**(23k 仓库 / 6M PR →(LLM-as-judge 预过滤)→ 沙箱三智能体流水线 → **10 万已验证实例 / 5200 repo**)+ **合成 F2P**(unit-test-creator agent 在缺测时恢复可执行性,而非伪造任务)+ **轨迹蒸馏**(DeepSeek-V3.2 出 **71k 轨迹 / 3.5B token**)。把这 71k 轨迹 **SFT 到 Qwen3-30B-A3B-Instruct → SWE-bench Verified 64.0%**(基座 22%,+42 分,超当时开源 SOTA)。**本论文 SFT-only,RL 留给后续的 DeNovoSWE**(doc2repo 长程,明确 for SFT & RL)。它在生态里 = **SWE-Gym 路线的工业级放大**(规模 ×~3 个数量级),走**蒸馏-SFT**而非 DeepSWE 的从零 RL。

> ⚠️ 更正背景:本看板早前把「ScaleSWE」误当成 Scale AI 的 SWE-bench Pro,已在 Dispatch 12 订正。本篇是订正后的完整详解。数字为调研快照(全文 PDF / HF 在本环境被 egress 403 拦),标 **provisional**。

承接 Dispatch 12(SWE 上手)的更正。这期把 ScaleSWE 单独讲透,并和 SWE-smith / R2E-Gym 等横向比。

---

## 1. ScaleSWE 是什么

一句话:**用"最大规模真实 PR 挖掘 + 缺测时合成 F2P + 强模型轨迹蒸馏"造出迄今最大的开源可执行 SWE 训练数据,再蒸馏-SFT 出一个强编码 agent。**

- **团队**:AweAI-Team(人大 RUC 关联;作者含 Jiale Zhao、Guoxin Chen、Wayne Xin Zhao、Ruihua Song、Ji-Rong Wen、Kai Jia 等)。
- **产物**:数据集 **Scale-SWE**(2 万 Real-Executable 实例,自称最大开源可执行 SWE 数据集)+ **Scale-SWE-Distilled**(71k 蒸馏轨迹 / 3.5B token)+ 模型 **Scale-SWE-Agent**(Qwen3-30B-A3B SFT)+ 框架 **AweAgent**。
- **结果**:**SWE-bench Verified 64.0%**(基座 Qwen3-30B-A3B-Instruct 22.0% → +42 分),超当时开源 SOTA(如 KAT-Dev-32B 62.4%)。**provisional。**

## 2. 方法

**(a) 数据管线:6M PR → 10 万实例。** 漏斗式:**23k 仓库 / 6M PR →(LLM-as-judge 只看 PR 元数据[diff/描述/merge message]预过滤,在昂贵 build 前剔废)→ ~1M 候选 → 沙箱多智能体处理 → 10 万已验证实例(5200 repo)**。处理是**三智能体流水线**(一阶段一 agent):① 建环境 / build,② 造测试(unit-test creator),③ 写问题陈述。编排器 **MEGAFLOW**(分布式长程 agentic job 系统)把每个 build 派到独立**阿里云 ECS**,沙箱内跑完整 build 产出**已验证 Docker 镜像**、推到**阿里云 ACR**,靠 Docker 层缓存压存储成本。

**(b) 合成 F2P:在缺测时"恢复可执行性"。** 很多高质量 PR 没有作者写的 F2P 测试,于是 **unit-test creator agent** 从源仓库 + PR 合成可执行测试,产出经验证的 **F2P / P2P**(数据集有专门字段存这些 agent 生成脚本)。验证口径同 SWE-bench:**F2P = 打 golden patch 前失败、后通过;P2P = 前后都通过**;并**强制固定 F2P/P2P 执行顺序防 test pollution**。**关键:它只在缺测时合成测试,问题本身仍是真实 PR**——这正是它区别于 SWE-smith(造合成 bug)/ R2E-Gym(回译任务)的方法论核心。

**(c) 蒸馏 → SFT → 64%。** 教师 **DeepSeek-V3.2**;从 **25k 实例**采集,**每实例 5 次采样、temperature 0.95、每条轨迹 ≤100 交互轮**;**只保留最终提交通过全部测试的轨迹**(outcome-based 拒绝采样)→ **71,498 条 / ~3.5B token**。学生 **Qwen3-30B-A3B-Instruct** SFT 后即 Scale-SWE-Agent → **64% Verified**。**消融**:同一蒸馏+SFT 流程换 SWE-Gym / SWE-smith 数据做基线,Scale-SWE 数据显著胜出(支持"高保真真实数据 > 海量合成数据";逐基线数字未公开核实)。

**(d) 有没有 RL?** **没有——本论文是 SFT-only**(蒸馏 → 监督微调),64% 是纯 SFT 结果。RL 出现在后续的 **DeNovoSWE**(2026-06)里。
> ⚠️ 易混点:网上「6.2% → 57.8% over 0–60K、48K plateau」那条 scaling 曲线属于 **Skywork-SWE**,**不是** ScaleSWE。

## 3. 生态(AweAI-Team)

一个共享框架上叠两个旗舰数据项目 + 一个执行核:

- **AweAgent**(Apache-2.0,[repo](https://github.com/AweAI-Team/AweAgent)):统一 build/eval/train 的执行核,跑 search/code/terminal 三类任务。四层 —— **TaskRunner**(批量引擎)、**AgentContext**(共享状态,带 max_steps/上下文上限)、**AgentLoop**(rollout 引擎,按 action.type 分发,**训练模式下 tool 观测 loss_mask=0**,产 token 级轨迹 + logprobs —— 即 71k 蒸馏轨迹的产出机制)、**Scaffolds**(near-stateless 策略,如 `search_swe`)。Scale-SWE 跑在 **`search_swe`**(ACI 工具 ExecuteBash / StrReplaceEditor / Finish + **反泄漏 bash blocklist**,封 `git log`/`git fsck`、search 模式禁访问目标仓库)。
- **Scale-SWE-Agent**([HF 模型](https://huggingface.co/AweAI-Team/Scale-SWE-Agent)):推理 200 轮 / 256k 上下文 / temp 1。
- **DeNovoSWE**(2026-06,[repo](https://github.com/AweAI-Team/DeNovoSWE),arXiv 2606.10728):任务从"修已有 PR bug"转为 **doc2repo —— 仅给自然语言 spec 从零生成整个仓库**(长程)。明确设计为 **for SFT & RL** 的长程**环境**(支持 RL rollout);**4818 实例**,教师 DeepSeek-v4-Pro-High,基座 Qwen3.5-35B-A3B → **~50% BeyondSWE-Doc2Repo**;agent 配置 **500 轮**(比 ScaleSWE 200 轮更长程)。

## 4. 横向对比:ScaleSWE vs 同行

| 项目 | 组织 | 任务来源 | 合成/恢复 | 规模 | 训练法 | 可执行环境 | Verified 头条 |
|---|---|---|---|---|---|---|---|
| **ScaleSWE** | AweAI-Team(RUC) | **真实 PR 大规模** 6M→100k | unit-test-creator 缺测时合成 F2P | 20k Real-Exec + 71k 轨迹 | **蒸馏/SFT**(DeepSeek-V3.2) | 是 | **64%**(Qwen3-30B-A3B) |
| SWE-smith | Princeton | **合成造 bug** | 程序化造 bug | 50k / 128 repo | 蒸馏/SFT(5016 轨迹,Claude 3.7) | 是 | 40.2%(Qwen2.5-Coder-32B) |
| SWE-Gym | Berkeley/CMU | **真实 PR(小)** | 用作者测试 | 2438 / 11 repo | SFT + verifier | 是 | ~32%(32B/OpenHands) |
| R2E-Gym | Berkeley | 真实 commit | **SWEGEN 回译**造测试 | 8.1k | SFT + 混合 verifier(TTS) | 是 | 34.4%→51%(verifier) |
| DeepSWE | Agentica+Together | 真实 PR(R2E 4.5k) | 继承 R2E 环境 | 4500 | **纯 RL**(GRPO) | 是 | 42.2%→59%(TTS) |
| Meta SWE-RL | Meta FAIR | 软件演化数据 | 无 | 大规模语料 | **RL + 规则奖励**(不执行) | 否 | 41%(Llama3-SWE-RL-70B) |
| SWE-bench/Verified/Pro | Princeton(+Scale Pro) | 真实 PR,人工校验 | 仅评测 | 2294/500/1865 | — | 是 | 指标本身 |

*(数字均 provisional;基座不同,跨行非严格可比。)*

**几条要点:**
- **和 SWE-Gym 同路线、规模走极端**:都是真实 PR + 蒸馏 SFT,ScaleSWE 把挖掘推进 ~3 个数量级(6M PR/5.2k repo/10 万实例 vs SWE-Gym 2.4k/11)。本质是"**SWE-Gym 论点的工业级放大**";64% vs ~32% 与规模跃迁吻合。
- **合成只为"可执行性"、不伪造"任务"**:区别于 SWE-smith(造合成 bug)、R2E-Gym(回译任务)——ScaleSWE 保真实问题,只在缺测时补 F2P,是更保守的合成用法,也是其核心新意。
- **押在蒸馏侧、不在 RL 侧**:64% 来自 SFT;对照 DeepSWE(纯 RL 59% w/TTS)、Meta SWE-RL(非执行规则奖励 41%)。它扮演 R2E-Gym 那种"环境/数据供给"角色,但走 SFT。
- **可执行性 = 通向 RL 的桥**:Real-Executable 实例自带可跑 F2P,正好是 RL 要的可验证奖励;**DeNovoSWE 明确 for RL**,暗示生态路径 **SFT-now(ScaleSWE)→ RL-next(DeNovoSWE)**,镜像 R2E-Gym→DeepSWE 的交接。
- **新意在组合**:真实 PR 挖掘、超大规模、为可执行做测试合成、轨迹蒸馏——单看都不新;ScaleSWE 的新意是这几样在**最大挖掘规模**上的合取。

## 5. 对 RL-on-NPU 的意义

- **现成的 RL 冷启动语料**:ScaleSWE 的 71k 可执行轨迹(+ DeNovoSWE 的蒸馏轨迹)正好是 Dispatch 13 里"SFT 冷启动 → GRPO"那一步要的数据——在昇腾上做 SWE RL,可以**直接拿它做 SFT 起点**,省去自建数据管线。
- **DeNovoSWE 的 doc2repo 长程任务 = 昇腾 RL 的极限压测**:500 轮、256k 上下文的"从零造仓库",把长 rollout 显存/异步痛点推到顶,是检验昇腾 RL 系统(Dispatch 13)最严苛的负载。
- **蒸馏-SFT 路线对 NPU 团队友好**:相比从零 RL,蒸馏+SFT **算力轻、稳定、易复现**,适合先在昇腾上把"数据→SFT→评测"闭环跑通,再上 RL。
- **可执行 F2P 奖励直接可用**:不像 Meta SWE-RL 的非执行相似度奖励,ScaleSWE/DeNovoSWE 的 F2P 是真·可验证奖励,接到昇腾 RL 的 verifier 上即用。

---

*由 multi-agent workflow 深挖产出(paper / AweAgent / 生态 / 横向对比 四路并行 + 综合)。来源:arXiv 2602.09892、github.com/AweAI-Team(ScaleSWE / AweAgent / DeNovoSWE)、HF AweAI-Team,以及对比项 SWE-smith(2504.21798)/ SWE-Gym(2412.21139)/ R2E-Gym(2504.07164)/ DeepSWE / Meta SWE-RL(2502.18449)。数字 provisional(全文 PDF 在本环境被 403)。*
