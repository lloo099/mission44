# Dispatch 08 · Agentic RL 入门:从"会推理"到"会行动"

*2026-06-25 · NPU Frontier Dispatch · agentic-RL / tool-use / credit-assignment / RL-on-NPU*

> **TL;DR** — Agentic RL 是 2026 的 RL 主线:把 LLM 从"被动生成序列"训成"自主决策的智能体"。它和单轮 RLVR 的根本区别是**时间尺度**——单轮是一步 MDP,agentic 是**跨 10–100+ 回合的 POMDP**,每回合一次 LLM 调用 + 一次环境交互,整条轨迹动辄 **10万–50万 token**。由此带来三个核心难题:**① 信用分配**(episode 级奖励在上百回合里几乎没信息)、**② 工具集成推理 TIR**(学会"何时/如何/用哪个"工具,而非模仿 ReAct 脚本)、**③ 长 rollout 的系统开销**。一批框架(VerlTool、AgentRL、SkyRL-Agent、ProRL Agent、ARLArena)和综述正在把它工程化。对 RL-on-NPU:**超长 agentic rollout 把昇腾"无 sleep-mode + 64GB HBM"的显存痛点放大到极致**——这恰恰让 async/解耦 rollout、量化 rollout、稀疏注意力(Dispatch 02/04/05)在这里收益最大。

这期不讲单个模型,讲一个**范式**。应要求,把 agentic RL 的地图、难题、框架和它对昇腾的含义梳理一遍。论文均 arXiv 一手来源。

---

## 1 · 什么是 Agentic RL

一句话:**从"奖励一段推理"变成"奖励一串行动"。**

| | 单轮 RLVR(如 R1) | **Agentic RL** |
|---|---|---|
| 形式 | 一步 MDP:prompt → 回答 → 奖励 | 多回合 **POMDP**:观察→动作→环境→… |
| 轨迹 | 1 次生成 | **10–100+ 回合**,每回合 LLM + 环境 |
| token | 几千 | **10万–50万+** |
| 信号 | 一个可验证奖励 | episode 级奖励 + 大量中间步 |
| 能力 | 推理 | 用工具、查资料、写代码、跑命令、纠错重试 |

它把 LLM 从"passive sequence generator"重塑为"autonomous decision-maker":**自己决定何时调工具、调哪个、看了结果怎么继续**——靠结果驱动(outcome-driven)而非模仿示范(imitation)。

## 2 · 三个核心难题

**① 信用分配(Credit Assignment)—— 最硬的前沿**
轨迹跨上百回合,只有最后一个 episode 级奖励,**中间哪一步功过难分**。2026 的一篇综述梳理了 **47 种信用分配方法**,按两维分类:**粒度**(token / segment / step / turn / multi-agent)× **方法**(蒙特卡洛 / 时序差分 TD / 基于模型 / 博弈论 / 信息论)。这是 agentic RL 能否扩到长程任务的瓶颈。

**② 工具集成推理(TIR)/ ARLT**
从 ReAct 式"先想后做"的事后拼接,走向**深度交织的多轮工具使用**。RL 把范式从"模仿工具调用脚本"换成"按结果优化"——智能体自己学会 **何时、如何、用哪个工具**。这条线叫 **ARLT(Agentic RL with Tool use)**。

**③ 长 rollout 的系统开销**
一条 agentic 轨迹要跑几十次"生成 + 环境往返",**rollout 比单轮重得多、长尾更严重、显存占用爆炸**。所以 agentic RL 框架普遍押注**异步 rollout**(让生成和训练解耦、不被最慢的轨迹拖住)。

## 3 · 框架地图(2026)

| 框架 | 定位 | 关键点 |
|---|---|---|
| **VerlTool**(ARLT) | verl 上的工具使用 agentic RL | 全异步 rollout,统一工具接口 |
| **AgentRL** | 多轮 × 多任务 RL 系统 | 规模化 agentic 训练 |
| **Agent-R1** | 工具环境下的多轮推理 RL | 多轮 + 工具调用 |
| **SkyRL-Agent** | 高效多轮 agent 训练 | 针对多轮的训练效率 |
| **ProRL Agent** | Rollout-as-a-Service | 把 agent rollout 生命周期做成 API(见 RL 标签卡) |
| **ARLArena** | 统一、稳定的 agentic RL | 稳定性 + 统一环境 |
| **AgentV-RL** | 用 agentic verifier 扩奖励建模 | 奖励侧 |

综述入口:**《The Landscape of Agentic RL for LLMs》**(arXiv 2509.02547)、**《From Reasoning to Agentic: Credit Assignment》**(2604.09459)、**《Rethinking Agentic RL in LLMs》**(2604.27859,已在 RL 标签)。

## 4 · 被低估的一环:环境与奖励设计

研究火力大多压在**算法**上,而 **数据 / 环境 / 奖励设计**关注度低得多——但它常常是上限所在:

- **可验证奖励(RLVR)**:用单元测试、答案校验等自动信号,而非学一个奖励模型——agentic 场景下尤其稳。
- **rubric-based + 模拟环境 + 合成任务**:像《Mock Worlds, Real Skills》那样,用合成任务 + 模拟环境 + 评分量表训出小型 agent,绕开真实环境的昂贵与不可复现。
- **agentic verifier**:用智能体本身去做更细的奖励判定(AgentV-RL)。

## 5 · 对 RL-on-NPU 的意义

Agentic RL 几乎是为"放大昇腾痛点"量身定制的,但也因此**最能体现前几期那些技术的价值**:

- **超长 rollout = 显存痛点放大到极致**。10万–50万 token 的多轮轨迹,KV cache 巨大;而昇腾 910B 单卡 64GB、**无 vLLM sleep-mode**,rollout 与 train 抢显存(见 NPU 架构页"RL 显存争用"视图)。agentic 场景下这个矛盾最尖锐。
- **所以前几期的招式在这里收益最大**:**异步/解耦 rollout**(Dispatch 02 的 VerlTool/AgentRL 同源思路)、**量化 rollout**(FP8/INT8,Dispatch 02)、**稀疏注意力 / KV 压缩**(MSA/DSA/CSA-HCA/SWA,Dispatch 04/05/07)——每一样都直接砍 agentic rollout 的显存与时延。
- **环境/奖励设计是 NPU 团队的低门槛切入点**:可验证奖励 + 模拟环境**算力轻、价值高**,不必先解决大规模训练就能做出贡献。
- **信用分配在 NPU 上还要多防一层**:跨回合的 advantage 计算 + NPU 上重写的算子,会叠加 train-inference 数值漂移——又是 **align-probe** 的用武之地。

## 6 · 怎么入门 / 下一步看什么

1. **读两篇综述**(2509.02547 + 2604.09459)建立地图,再挑一个框架(VerlTool / AgentRL)跑通一个多轮工具任务。
2. **在昇腾上复现一个多轮 agentic RL**:量化 rollout + 异步,量化它对 64GB HBM 的缓解——这正是看板 Project Ideas 里"异步 off-policy RL"那条的 agentic 版。
3. **盯信用分配的"够用粒度"**:turn 级 vs step 级 vs token 级,在长程任务上谁性价比最高。
4. **环境设计**:把 MiMo Code / Terminal-Bench 这类 200+ 步任务做成可训练的 RL 环境。

---

*来源:arXiv —— Landscape of Agentic RL(2509.02547)、Credit Assignment(2604.09459)、Rethinking Agentic RL(2604.27859)、VerlTool(2509.01055)、AgentRL(2510.04206)、SkyRL-Agent(2511.16108)、ProRL Agent(2603.18815)、ARLArena(2602.21534)、AgentV-RL(2604.16004)、Mock Worlds Real Skills(2601.22511)。相关卡片见本看板 RL for LLMs 标签页。*
