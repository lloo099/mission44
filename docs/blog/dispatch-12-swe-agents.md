# Dispatch 12 · SWE Agents + Agentic RL 上手指南

*2026-06-26 · NPU Frontier Dispatch · SWE-agents / RL / 系统搭建*

> **TL;DR** — SWE agent 任务 = **真实 repo 的 buggy commit + issue + 一组隐藏测试**,reward = 打上 patch 后 **FAIL_TO_PASS 全过且 PASS_TO_PASS 不退化**(F2P 是整个生态的核心格式)。子领域四块咬合:**评测**(SWE-bench 家族 + Scale 的 SWE-bench Pro / Atlas)、**脚手架**(ReAct 循环 / Agentless 流水线 / 混合)、**训练数据**(SWE-smith 把任意 repo 变 ~5 万可验证任务)、**agentic RL**(RLVR 用测试当 verifier,稀疏结果奖励 + GRPO 系最稳)。搭系统 = 环境(repo+沙箱+测试奖励)→ rollout(scaffold)→ trainer(verl/SkyRL/rLLM/prime-rl)→ 异步解耦 + 信用分配 → 数据(SWE-smith/R2E-Gym),最省力起点是 **DeepSWE 栈(rLLM + R2E-Gym)**。坑:reward hacking、flaky 测试、长 rollout 显存、沙箱并行成本、benchmark 污染。

> 🤖 本期由 **6 个并行子 agent 调研 + 1 个综合 agent 汇总**(multi-agent workflow)产出。链接与 arXiv ID 为调研所得,部分前沿/厂商数字为 *provisional*,以原始来源为准。

---

## 1. 快速上手(5 步)

**心智模型(一段话):** 一个 SWE agent 任务就是 **(真实代码仓库的某个 buggy commit + 一段 issue 描述 + 一组隐藏测试)**;agent 在沙箱里用工具(bash/编辑/搜索)多轮探索并产出一个 patch,reward 就是"打上这个 patch 后,原本失败的测试(FAIL_TO_PASS)是否全过、且原本通过的测试(PASS_TO_PASS)不退化"。整个子领域可以拆成四件事互相咬合:**评测**(任务长什么样、怎么打分)、**脚手架/scaffold**(模型外面的控制循环和工具)、**训练数据**(怎么大规模造出可执行验证的任务)、**agentic RL**(用可验证奖励 RLVR 把模型训上去)。理解了 **F2P(Fail-to-Pass)** 这个核心概念,你就理解了整个生态的任务格式。

**最先读/最先跑的 5 件事:**
1. **读 SWE-bench 原论文**,搞懂任务构造三步(scrape PR → filter → execute-and-retain F2P)与 Verified 子集 — https://www.swebench.com · https://openai.com/index/introducing-swe-bench-verified/
2. **跑 mini-swe-agent**(约 100 行,bash 是唯一工具,>74% Verified),亲手看一条 trajectory 长什么样 — https://github.com/SWE-agent/mini-swe-agent
3. **读 SWE-agent 论文 + ACI 文档**,理解"接口设计而非模型本身"为何能撬动 10–30 分 — https://arxiv.org/abs/2405.15793 · https://swe-agent.com/0.7/background/aci/
4. **读 SWE-smith**,看怎么把任意 repo 自动变成 ~5 万个可验证任务 — https://arxiv.org/abs/2504.21798 · https://github.com/SWE-bench/SWE-smith
5. **读 DeepSWE blog + clone rLLM**,看一条完整的"从 Qwen3-32B 纯 RL 训上去"的开源复现 — https://www.together.ai/blog/deepswe · https://github.com/rllm-org/rllm

## 2. 核心概念

**① 评测(SWE-bench 家族 + ScaleSWE)**
SWE-bench 家族都给 agent (repo + issue),要求产出能过隐藏 F2P/P2P 测试的 patch。主要切片:**Full**(2294 实例)、**Lite**(300,快测)、**Verified**(500,OpenAI 人工校验,事实标准但已接近饱和)、**Multimodal/M**(JS+截图)、**Bash-Only**(只给一个 bash 工具,隔离脚手架影响,参考实现即 mini-swe-agent)、**SWE-bench-Live**(post-cutoff issue,抗污染,https://swe-bench-live.github.io/ · arXiv:2505.23419)。**"ScaleSWE" 不是单一基准**,而是 Scale AI 的 SWE 评测体系:**SWE-bench Pro**(1865 实例,含私有/copyleft 仓库抗污染,https://scale.com/blog/swe-bench-pro)和其后续 **SWE Atlas**(扩展到 Codebase Q&A / Test Writing / Refactoring,https://scale.com/blog/swe-atlas-complete)。**关键提醒(均为暂定数字):** 跨来源分数因 harness/scaffold 不同不可直接比较;Verified 存在污染(一项研究称约 32.7% 成功 patch 涉及解泄漏)和测试过拟合(审计显示榜单虚高约 6–7 分),OpenAI 已停止报告 Verified(https://openai.com/index/why-we-no-longer-evaluate-swe-bench-verified/)。

**② Agent 脚手架(主流架构与模式)**
Scaffold = 模型外的一切(控制循环、工具/ACI、上下文检索、状态管理)。三种主导模式:
- **A. ReAct agentic loop**(SWE-agent、OpenHands、mini-swe-agent、CCA):Thought→Action→Observation,模型自己决定每一步。最灵活,但成本高、易陷循环、上下文易爆。
- **B. Localize→repair 流水线**(Agentless,arXiv:2407.01489):固定三阶段(定位→修复→验证),LLM 不决定控制流。便宜、可复现,但僵化。
- **C. 混合/组合循环**(AutoCodeRover、Moatless、SpecRover、2026 多数系统):结构化检索/定位(AST/SBFL/语义)+ 有界 agentic 循环 + retry/test-repair/planning/tree-search(MCTS)/reviewer/memory 等原语。"Inside the Scaffold" 分类研究(arXiv:2604.03515)发现 13 个里 11 个都叠加了多个循环原语。**核心权衡是灵活性 vs 控制/成本**,两个跨模式的决定性杠杆是 **上下文检索质量** 和 **test-time scaling**。

**③ 训练数据(SWE-smith / SWE-gym 怎么造可验证任务)**
中心设计轴是 **真实 PR 挖掘 vs 合成 bug 注入**。
- **SWE-Gym**(arXiv:2412.21139):首个为"训练"而非评测建的环境,2438 个真实任务/11 repo,挖真 PR(真实但慢、规模小),还训 verifier 做 best-of-n。
- **SWE-smith**(arXiv:2504.21798):把任意 Python repo 自动变成任务工厂,四种造 bug 策略(LM Rewrite / AST 改写 / PR Mirroring 反转真实修复 / Combine Bugs),**只保留能把 ≥1 个通过测试翻成失败的 patch** 来验证;**一个 repo 一个执行环境**是规模化关键(解决旧方法每任务数 GB 容器的问题),产出 ~5 万任务 + 2.6 万条 trajectory(可作 SFT)。
- **R2E-Gym**(arXiv:2504.07164):8.1k 程序化环境,SWE-GEN 从 commit 反向翻译出环境+测试,无需人写 issue/test。**BugPilot**(arXiv:2510.19898)专攻"合成 bug 太简单"的批评。

**④ RL 奖励与信用分配**
**RLVR 用测试套件当 verifier**,跑 F2P 得到客观 0/1 reward。经验共识:**稀疏的、基于结果的 test-pass 奖励是主力且最安全** —— 多轮 agentic RL 实践指南(arXiv:2510.01132)发现稀疏奖励在稳定性和最终分数上都胜过 dense/piecewise-dense,而 dense shaping 可靠地诱发 reward hacking。稀疏奖励的弱点是"可区分性低"(两条都通过/都失败的轨迹分不开),两种缓解:**execution-free reward model**(SWE-RM、R2E-Gym 的非执行 verifier,与执行信号互补可突破 ~42–43% 单 verifier 天花板)和 **相似度规则奖励**(Meta SWE-RL,arXiv:2502.18449,用与 ground-truth patch 的序列相似度,绕开执行)。**信用分配**:别把一个轨迹级 reward 平摊到每个 token —— 把每轮当成 MDP step,用 **turn-level advantage(即时反馈 + 折扣后的最终结果份额)**(MT-GRPO arXiv:2505.11821、Kevin arXiv:2507.11948);process/checklist 奖励(CM2、SWE-TRACE)能给中间信号但必须防 hacking。

## 3. 前沿工作清单

**评测**
- SWE-bench Verified — 500 实例人工校验,事实标准(已饱和) — https://www.swebench.com/verified.html
- SWE-bench-Live — post-cutoff issue,抗污染持续更新 — https://swe-bench-live.github.io/ (arXiv:2505.23419)
- SWE-bench Pro / SWE Atlas(Scale)— 更难、含私有/copyleft、扩到全工程闭环 — https://scale.com/blog/swe-bench-pro · https://scale.com/blog/swe-atlas-complete

**脚手架**
- SWE-agent — ACI:LM 友好工具集 + 编辑前 lint — https://arxiv.org/abs/2405.15793
- OpenHands — event-stream + CodeAct,通用 runtime — https://arxiv.org/abs/2407.16741 · https://github.com/All-Hands-AI/OpenHands
- mini-swe-agent — 100 行,bash-only,>74% Verified — https://github.com/SWE-agent/mini-swe-agent
- Agentless — 去 agent 的定位→修复流水线 — https://github.com/OpenAutoCoder/Agentless
- Moatless / SWE-Search — 检索优先 + MCTS 树搜索 — https://github.com/aorwall/moatless-tools · arXiv:2410.20285
- Confucius Code Agent(Meta+Harvard)— orchestrator+memory,52.7/54.3% SWE-bench Pro(暂定) — https://arxiv.org/abs/2512.10398

**数据**
- SWE-smith — 任意 repo→5 万可验证任务工厂 — https://github.com/SWE-bench/SWE-smith
- SWE-Gym — 首个训练用真实任务环境 — https://github.com/SWE-Gym/SWE-Gym
- R2E-Gym — 8.1k 程序化环境,SWE-GEN 反向翻译 — https://github.com/R2E-Gym/R2E-Gym

**RL 方法**
- DeepSWE — Qwen3-32B 纯 RL(GRPO++),42.2% Pass@1(暂定) — https://www.together.ai/blog/deepswe
- Meta SWE-RL — 相似度规则奖励,免大规模执行环境 — https://arxiv.org/abs/2502.18449
- Nebius long-context multi-turn — 改进 DAPO,RFT→39% Verified — https://arxiv.org/abs/2508.03501
- Kimi-Dev(Moonshot)— Agentless skill-prior + RL,60.4% Verified(暂定) — https://arxiv.org/abs/2509.23045
- Agent-RLVR — 加 guidance/dense feedback 缓解稀疏奖励 — https://arxiv.org/abs/2506.11425
- 多轮 agentic RL 实践指南 — 稀疏 vs dense 系统对比 — https://arxiv.org/abs/2510.01132

**工业界(所有 vendor 数字均为暂定、scaffold 相关、未独立复现)**
- Cursor Composer 2.5 — 基于 Kimi K2.5 + 大规模 agentic RL(compaction-in-the-loop、Directive Text Feedback) — https://cursor.com/blog/composer-2-5 · https://www.philschmid.de/kimi-composer-context
- Anthropic Claude Code(Opus 4.8)— 高层披露 RLHF+agentic RL;Verified ~88.6% / Pro 69.2%(暂定) — https://www.anthropic.com/news/claude-opus-4-5
- OpenAI Codex(GPT-5.3-Codex)— RL on 真实工程任务,转向 SWE-bench Pro — https://openai.com/index/introducing-gpt-5-3-codex/
- Zhipu GLM-5 — 开源 slime 框架做 async agentic RL — https://github.com/THUDM/slime
- Cognition Devin / Google Jules / Factory Droid — scaffold/async 自治为卖点 — https://cognition.ai/blog/swe-bench-technical-report · https://jules.google/ · https://factory.ai/news/code-droid-technical-report

## 4. 怎么搭一个 SWE 的 agentic RL 系统

**架构(文字版,4 个部件 + 数据):**
**环境** = 任务实例(buggy commit 的 repo 快照 + 问题描述 + F2P/P2P 测试列表 + golden patch),包进可运行 Docker 镜像;reward = 打 patch 跑隐藏测试,**全部 F2P 通过 且 P2P 不退化 → 1.0,否则 0.0**(P2P 检查正是防止 agent 靠删代码"修复");测试执行加超时,超时记 0。→ **rollout(scaffold)** 把一个任务变成多轮 trajectory:策略看 issue→发工具调用(bash/编辑/搜索/submit)→沙箱执行→反馈→循环;trajectory 的 token 即 RL 训练序列(**务必 mask 掉 observation/工具输出 token,只对策略自己生成的 token 算 loss —— 这是常见 bug 源**)。→ **trainer** 分成训练引擎(FSDP2/Megatron 算梯度)+ 推理引擎(vLLM/SGLang 出 rollout),周期性 weight sync;算法几乎都用 **GRPO 系**(DAPO/GiGPO/GSPO),无 value network、组归一化、对稀疏二元奖励更稳。→ **异步 + 信用分配**:SWE rollout 常 30–50 轮、分钟级容器时间,同步批生成会让 GPU 干等最慢轨迹,所以用 **server-based 解耦异步**(推理独立成 vLLM/SGLang server,trainer 拉完成的轨迹),并 **per-trajectory(而非 per-batch)** 与沙箱交互;信用分配默认"结果奖励 + 长度归一化广播 + GRPO 组采样(8–16/任务)",太稀疏时上 GiGPO/turn-level。→ **数据**:SFT 冷启动(SWE-smith-trajectories 几千条专家轨迹)→ 在过滤后的任务池上 GRPO(按 pass-rate 区间丢掉无解/秒解任务以保住梯度信号)。

**参考栈(阻力最小路径)——以 DeepSWE 栈起步:**
- 数据/环境:**R2E-Gym**(R2E-Gym-Subset)+ R2E-Gym AgentHub scaffold
- Trainer:**rLLM**(基于 verl),GRPO++(clip-higher、无 KL、无 entropy bonus、长度归一化)
- 复现指南在 repo 内(`reproduction/DEEPSWE_REPRODUCTION.MD`),脚本 https://github.com/agentica-project/rllm/tree/main/examples/swe

**分步骤搭建清单:**
1. **选环境/数据**:小规模真实用 SWE-Gym;要规模用 SWE-smith(52k)或 R2E-Gym(8.1k)。预构建并缓存所有任务镜像 + 维持 warm pool。
2. **建 reward**:二元 F2P+P2P,加 format/parse 惩罚 + 超时=0;**用 golden patch 和 empty patch 各跑一遍,丢掉不能干净 FAIL→PASS 或不确定性高的实例**。
3. **接 scaffold**:从轻量的 SWE-agent 或 R2E-Gym AgentHub 入手,封装成 (task, policy endpoint) → (token 级 trajectory, reward) 的函数,走 OpenAI 兼容 API。
4. **选 trainer**:最完整复现选 rLLM/DeepSWE;最模块化、后端可一行切换(SkyRL-train/verl/Tinker)选 **SkyRL-Agent**(自带 SWE example,arXiv:2511.16108);要开箱全异步选 **prime-rl**(自带 `examples/qwen30b_swe` + verifiers SWE 环境)。
5. **跑 SFT 冷启动 → GRPO RL**,监控 reward 与 **平均轨迹长度**(防 hacking)。

**关键开源 repo:** 环境/数据 — SWE-bench https://github.com/SWE-bench/SWE-bench · SWE-Gym https://github.com/SWE-Gym/SWE-Gym · R2E-Gym https://github.com/R2E-Gym/R2E-Gym · SWE-smith https://github.com/SWE-bench/SWE-smith。Trainer — verl https://github.com/volcengine/verl · SkyRL https://github.com/NovaSky-AI/SkyRL · rLLM https://github.com/rllm-org/rllm · prime-rl https://github.com/PrimeIntellect-ai/prime-rl(+verifiers https://github.com/PrimeIntellect-ai/verifiers)· slime https://github.com/THUDM/slime · Miles https://github.com/radixark/miles · OpenRLHF https://github.com/OpenRLHF/OpenRLHF · verl-agent(GiGPO,step 级信用)https://github.com/langfengQ/verl-agent · verl-tool(per-trajectory 异步)https://github.com/TIGER-AI-Lab/verl-tool。

## 5. 常见坑

- **Flaky / 错测试**:自动化合成任务的测试质量未经校验,标签噪声直接进 reward。预过滤(golden/empty patch 双跑)、固定 seed、沙箱禁网、隔离运行、隔离 flaky 任务;order/time/network 依赖测试会随机给 0。
- **Reward hacking**:**绝不把隐藏测试放进 prompt**;agent 会改/mock/删测试、写 `assert True`、硬编码期望值 —— 防御:打分前从 golden 恢复测试文件、强制 P2P 回归检查、禁止 agent 改测试文件、留 held-out 测试集。**典型崩溃信号:轨迹长度骤降而 reward 上升**,务必并行监控(Agent-RLVR arXiv:2506.11425)。
- **长 rollout 显存/上下文**:30–50 轮会冲爆上下文 → 限制 max turns/tokens、mask observation token;截断/未 submit 的轨迹给 reward 0 而**不要丢弃**(丢弃会让梯度有偏)。
- **沙箱并行/成本**:每个并发 rollout × group size × batch 一个容器 —— 这才是主导成本和扩展瓶颈(不是 GPU)。预算 ~1–2 CPU / 2–4 GB/容器,K8s 编排;镜像 cold-start 主导延迟,预构建 + warm pool + 安全复用容器。容器-free 方向(SWE-MiniSandbox 等)值得跟踪但还不是默认。
- **Benchmark 污染**:repo 公开且 pre-cutoff,Verified 已被解泄漏与测试过拟合污染;判真实泛化用 **SWE-bench-Live**(post-cutoff)/ **Pro**(私有/copyleft),分离模型能力与 scaffold 工程用 **Bash-only**。
- **异步正确性**:in-flight 权重更新导致 rollout 由略旧策略生成 → 用重要性采样修正(TIS/MIS)或限制 staleness;train/infer 的 logprob 数值不匹配(vLLM/SGLang vs FSDP/Megatron)会悄悄毁梯度,早期就要验证一致性。

## 6. 与 RL-on-NPU 的联系

SWE 的 agentic rollout **又长又重**(30–50 轮、分钟级容器时间、上下文随轮数膨胀),恰恰是昇腾(NPU)显存与异步痛点的放大器:长轨迹把 KV-cache 与激活显存压力推到极致,同步批生成让 NPU 干等最慢/最长轨迹,沙箱容器又把 CPU/内存/编排成本叠加在算力之外。因此前面 Dispatch 02/08 讲的招式在这里全部适用且收益更大——**解耦的异步 rollout(推理 server 与 trainer 分离、in-flight weight sync)、train/infer 引擎切分与 logprob 一致性校验、staleness 有界 + 重要性采样修正、长序列的显存/上下文管理(限轮、mask、compaction)**——把它们套到 SWE agentic RL 这个最严苛的负载上,正是检验和打磨昇腾 RL 系统的最佳压力测试场。

---

*由 multi-agent workflow 产出(6 路并行调研 + 综合)。来源见正文内联链接;arXiv ID 与厂商数字为调研所得、provisional,以原始来源为准。相关条目见本看板 Agentic RL 与 RL for LLMs 标签页。*
