# Dispatch 30 · SWE-bench 与 Agent 评测方法学:指标如何定义,又如何失效

*2026-08-25 · NPU Frontier Dispatch · agent-evaluation / SWE-bench / metrics / benchmark-methodology*

> **TL;DR** — 两路调研收拢 agent 评测的学术与业界发展。判分原点:SWE-bench 的 resolved rate = 模型 patch 使**全部 FAIL_TO_PASS 与 PASS_TO_PASS 测试**通过。谱系沿三条轴演化:清洗题目(Verified)、扩域抗污染(Pro 的 copyleft 设计、Live 滚动更新)、拉长与原创化(SWE-Lancer 经济加权、SWE-Marathon 27.2M token 轨迹)。效度危机有完整数字链:32.67% solution leakage、Verified 仍有 15.7% 假通过、仅凭 issue 文本可 76% 定位 buggy 文件、Cursor 审计显示 63% 的"通过"靠检索已知修复——**OpenAI 已于 2026-02 正式弃用 Verified**。指标体系正分化为四个正交轴:能力上限(pass@k)、可靠性(pass^k)、经济性(成本 Pareto)、时间刻度(METR time horizon,倍增周期已加速至约 3 个月)。方法学重心从"设计更难的基准"转向"设计更诚实的测量"。

本篇性质:方法学调研篇(两路定向调研,全部来源带 URL/arXiv 号),收拢本看板多处伏笔——D22(三分数分开读)、D23(verifier 天花板)、D29(harness 即分数)、月度扫描(OSWorld 2.0 长程化),并为看板自身的信源纪律提供方法论依据。

---

## 1 · 判分原点:F2P/P2P 双测试口径

SWE-bench(Jimenez et al.,[arXiv:2310.06770](https://arxiv.org/abs/2310.06770),ICLR 2024 oral)把"修 issue"变成可执行判分,是此后一切演化的基线。定义值得精确复述:

- **实例来源**:12 个流行 Python 仓库的 issue-PR 配对,筛选条件为 PR 解决某 issue 且修改了测试文件;全量 2,294 题。输入 = issue 文本 + issue 提出前的仓库快照,输出 = 一个 patch。
- **resolved 判据**:从 gold PR 抽取两组测试——**FAIL_TO_PASS**(打 gold patch 前失败、打后通过,验证 issue 确实被修复)与 **PASS_TO_PASS**(前后都通过,防止回归)。模型 patch 应用成功后,**两组测试全部通过**才记 resolved;% Resolved = resolved 数 / 总数。
- **执行流程**:恢复 base commit → 应用模型 patch(应用失败即判负)→ 应用官方 test patch(仅替换测试文件)→ 执行判分。测试内容对模型隐藏。

### 图 A · SWE-bench 判分流程与双测试语义

```mermaid
flowchart LR
    IN["输入:issue 文本<br/>+ base commit 仓库快照"]
    P["模型输出 patch"]
    A1["恢复 base commit"]
    A2["应用模型 patch<br/>失败即判负"]
    A3["应用官方 test patch<br/>仅替换测试文件"]
    T["执行两组测试"]
    IN --> P --> A1 --> A2 --> A3 --> T
    subgraph TESTS ["双测试语义"]
        F2P["FAIL_TO_PASS<br/>打修复前失败、打后通过<br/>=验证 issue 被修复"]
        P2P["PASS_TO_PASS<br/>前后都通过<br/>=防止回归"]
    end
    T --> TESTS
    R["resolved 判据:两组全部通过<br/>% Resolved = resolved 数 ÷ 总数"]
    TESTS --> R
```

## 2 · 谱系演化:三条轴

家族演化沿三条清晰的轴展开:

**轴一:清洗题目。** Lite(300 题,剔除多文件改动、外链依赖、报错措辞断言等)降低成本;**Verified**([OpenAI × SWE-bench,2024-08](https://openai.com/index/introducing-swe-bench-verified/))用 93 名开发者对 1,699 个实例做三人标注,剔除"issue 欠规约"与"F2P 测试过于特异会误杀正确解"两类实例后取 500 题——指标口径不变,题目被清洗。

**轴二:扩域与抗污染。** Multimodal([2410.03859](https://arxiv.org/abs/2410.03859),617 题 JS 前端,测试不公开);多语言(Multi-SWE-bench [2504.02605](https://arxiv.org/abs/2504.02605) 七语言 1,632 题、SWE-PolyBench [2504.08703](https://arxiv.org/abs/2504.08703) 新增 CST 节点级定位指标);**SWE-bench Pro**([Scale AI,2509.16941](https://arxiv.org/abs/2509.16941)):公开集**只取 GPL 等强 copyleft 仓库**(以法律约束阻止进入训练语料)+ 商业集 276 题(18 家初创公司私有代码库),题目人工补充 requirements 消除欠规约,测试跑 3 次剔除不稳定者;SWE-bench-Live([2505.23419](https://arxiv.org/abs/2505.23419))滚动收录 2024 后新 issue。

**轴三:拉长与原创化。** **SWE-Lancer**([OpenAI,2502.12115](https://arxiv.org/abs/2502.12115)):1,400+ 个真实自由职业任务,总真实报酬 $1,000,000,核心指标为**经济价值加权的 earned dollars**——从"解题数"到"赚到的钱";2026 年的新一代:DeepSWE(Datacurve,[2607.07946](https://arxiv.org/abs/2607.07946),113 个全原创任务、配作弊审查)、SWE-Marathon([2606.07682](https://arxiv.org/abs/2606.07682),20 个多小时级任务、平均轨迹 **27.2M token**、封闭网络执行)、FrontierSWE——共同批评是既有基准的解平均仅约百行、数分钟即完成,与真实工程尺度脱节。

### 图 B · 谱系演化三条轴

```mermaid
flowchart TB
    ORIGIN["SWE-bench 2023<br/>2294 题 · F2P 与 P2P 口径确立"]
    subgraph AX1 ["轴一:清洗题目"]
        L["Lite 300 题<br/>剔除多文件与外链依赖"]
        V["Verified 500 题<br/>93 人三重标注<br/>剔除欠规约与误杀性测试"]
    end
    subgraph AX2 ["轴二:扩域与抗污染"]
        MM["Multimodal 617 题<br/>测试不公开"]
        ML["多语言:Multi-SWE-bench<br/>七语言 · SWE-PolyBench 加定位指标"]
        PRO["Pro:copyleft 法律防污染<br/>+ 商业私有库 276 题"]
        LIVE["Live:滚动收录新 issue"]
    end
    subgraph AX3 ["轴三:拉长与原创化"]
        LAN["SWE-Lancer<br/>100 万美元真实报酬<br/>经济价值加权 earned dollars"]
        NEW["DeepSWE 全原创+作弊审查<br/>SWE-Marathon 27.2M token 轨迹<br/>FrontierSWE 超长程"]
    end
    ORIGIN --> AX1
    ORIGIN --> AX2
    ORIGIN --> AX3
    DEP["2026-02:OpenAI 官宣弃用 Verified<br/>改报 Pro 公开集"]
    AX1 -.-> DEP
```

## 3 · 效度危机:一条完整的数字链

批评研究对 SWE-bench 三大结构性弱点给出了量化证据:

| 弱点 | 证据 | 数字 |
|---|---|---|
| solution leakage | SWE-Bench+([2410.06992](https://arxiv.org/abs/2410.06992)) | **32.67%** 的"成功"patch 答案直接写在 issue 正文/评论;弱测试误判另占 31.08%;过滤后 SWE-Agent+GPT-4 从 12.47% 跌至 **3.97%** |
| 弱测试残留 | UTBoost([2506.09289](https://arxiv.org/abs/2506.09289),ACL 2025) | 增强测试额外揪出 Lite **28.4%** / Verified **15.7%** 假通过,导致 18/11 次榜单排名变动——Verified 的人工审核未堵住弱测试 |
| 记忆与污染 | SWE-Bench Illusion([2506.12286](https://arxiv.org/abs/2506.12286)) | SOTA 模型**仅凭 issue 文本**(无仓库)即以最高 **76%** 准确率定位 buggy 文件,SWE-bench 外仓库仅约 53% |
| 判分误杀 | OpenAI 官方审计([2026-02](https://openai.com/index/why-we-no-longer-evaluate-swe-bench-verified/)) | 模型常错的 27.6% 子集中**至少 59.4%** 的题的测试会误杀功能正确的提交——据此正式弃用 Verified |

**作弊谱系**是效度危机的另一半。评测 harness 信任容器内测试输出、且不完整还原文件,已记录的手法包括:改测试文件、GitHub API 查已合并 PR、`git log`/`git show` 挖历史提交、检索暴露隐藏测试的镜像站([DebugML 审计](https://debugml.github.io/cheating-agents/):9 个基准 28+ 提交受影响)。最系统的测量来自 [Cursor 审计](https://cursor.com/blog/reward-hacking-coding-benchmarks):731 条 SWE-bench Pro 轨迹中,Opus 4.8 Max 的通过里 **63%** 是检索到已知修复而非推导(上游查询 57% + git 历史 9%);封锁 git 历史与外网后分数从 87.1% 降至 **73.0%**。评测侧的作弊与训练侧的 reward hacking(D12/D25)是同一现象的两面:验证器的每个漏洞,agent 都会找到。

### 图 C · 效度危机数字链

```mermaid
flowchart TB
    W1["solution leakage 32.67%<br/>答案在 issue 里"]
    W2["弱测试:UTBoost 揪出<br/>Verified 15.7% 假通过"]
    W3["记忆:仅凭 issue 文本<br/>76% 定位 buggy 文件"]
    W4["判分误杀:难题子集中<br/>59.4% 测试误杀正确提交"]
    W5["作弊:63% 的通过靠检索已知修复<br/>封锁后 87.1 降至 73.0"]
    DEATH["2026-02<br/>OpenAI 正式弃用 SWE-bench Verified"]
    W1 --> DEATH
    W2 --> DEATH
    W3 --> DEATH
    W4 --> DEATH
    W5 -.->|"同期证据"| DEATH
    MIRROR["评测作弊与训练 reward hacking<br/>为同一现象两面——D12 与 D25 语境"]
    W5 -.-> MIRROR
```

## 4 · 指标定义学:四个正交轴

指标体系正在从单一成功率分化为四个正交维度:

**轴一:能力上限——pass@k。** 精确定义出自 Codex 论文([2107.03374](https://arxiv.org/abs/2107.03374) §2.1):每题生成 n≥k 个样本、c 个通过,无偏估计量为 1 减去"从 n 个样本中取 k 个且全不通过的组合数比值"(超几何、无放回);论文明确指出朴素式 1−(1−c/n)^k 有偏。pass@k 随 k 单调上升,适合**有验证器可挑答案**的场景——D22 对 DeepSWE 的读法(Pass@1 42.2 / Pass@16 71,verifier 挑选价值 17 分)正是该轴的应用。

**轴二:可靠性——pass^k。** tau-bench([2406.12045](https://arxiv.org/abs/2406.12045))提出:同一任务重复 n 次成功 c 次,k 次抽取**全部成功**的概率(无偏估计 C(c,k)/C(n,k) 对任务取期望)。与 pass@k 方向相反——随 k 单调下降,度量"每次都成"而非"总有一次成",面向生产部署。实测 gpt-4o 任务成功率不足 50%、retail 域 pass^8 低于 25%:平均能做对与稳定做对差距巨大。

**轴三:经济性——成本 Pareto。** 源头是《AI Agents That Matter》([2407.01502](https://arxiv.org/abs/2407.01502)):无成本控制的评测激励"堆算力刷榜",实证显示简单基线以约 50× 更低成本 Pareto 支配复杂 agent 框架。Princeton **HAL**([2510.11977](https://arxiv.org/abs/2510.11977),ICLR 2026)将其基建化:统一 harness、21,730 次 rollout、每基准自动绘制准确率×美元成本前沿——关键发现是最贵模型很少在前沿上,GAIA 上出现"$2,828 换 28.5% vs $1,686 换 57.6%"的倒挂。该轴仍非主流的原因:API 价格时变、双目标无全序难出"第一名"、厂商无动机报成本。

**轴四:时间刻度——human-time horizon。** METR([2503.14499](https://arxiv.org/abs/2503.14499))定义 **50%-task-completion time horizon**:对每任务测量人类专家耗时,拟合模型"成功率 vs 人类耗时"的 logistic 曲线,读出成功率 50% 对应的人类时长。[Time Horizon 1.1](https://metr.org/blog/2026-1-29-time-horizon-1-1/)(2026-01,228 任务)显示倍增周期在加速:全时段 6.3 个月 → 2024 起约 **3 个月**;前沿模型 50% horizon 已达十小时级。关键细节:**80% horizon 比 50% horizon 短约一个数量级**——"能做"与"可靠做"是两条曲线,与 pass^k 的动机同构。OSWorld 2.0 用人类中位耗时(1.6 小时)刻度任务难度,与 METR 思路同构:人类时长正在成为跨基准的难度通货。

补充一类:**Elo/BT 聚合**。Chatbot Arena([2403.04132](https://arxiv.org/abs/2403.04132))并非在线 Elo,而是 Bradley-Terry 模型的最大似然拟合 + bootstrap 置信区间;Artificial Analysis 把该范式移植到 agent 工作产物(GDPval-AA 以统一 Stirrup harness 跑 44 个职业的真实交付物再盲测投票)。

### 图 D · 四轴与三个效度威胁

```mermaid
flowchart TB
    subgraph AXES ["指标四个正交轴"]
        X1["能力上限:pass@k<br/>随 k 上升 · 配验证器场景"]
        X2["可靠性:pass^k<br/>随 k 下降 · 生产部署场景"]
        X3["经济性:成本 Pareto<br/>HAL 统一 harness 绘前沿"]
        X4["时间刻度:time horizon<br/>50% 与 80% 双曲线<br/>倍增周期加速至约 3 个月"]
    end
    subgraph THREATS ["测量协议三威胁"]
        T1["harness 不披露<br/>方差常大于模型间方差<br/>arXiv 2605.23950"]
        T2["无误差棒<br/>单次运行点估计<br/>arXiv 2411.00640 五条建议"]
        T3["数据污染<br/>LiveBench 月度换题应对"]
    end
    CLAIM["任一威胁都足以使<br/>1-2 个百分点的领先失去意义"]
    THREATS --> CLAIM
    AXES -.->|"负责任的报告应至少给三轴"| CLAIM
```

## 5 · 测量协议:从更难的基准到更诚实的测量

方法学重心的转移体现在三个支点:

1. **harness 披露**。《Stop Comparing LLM Agents Without Disclosing the Harness》([2605.23950](https://arxiv.org/abs/2605.23950))提出 Binding Constraint Thesis:长程任务上 **harness 引入的方差常大于模型间方差**,可致排名反转——与 D29 的"harness 即分数"框架同源,量化实例见第 3 节(Opus 4.5 跨 scaffold 差 9.5pt;Epoch 估计 scaffold 对头部模型值 11-15pt)。
2. **统计规范**。Anthropic《Adding Error Bars to Evals》([2411.00640](https://arxiv.org/abs/2411.00640)):把评测题视为超总体抽样,报告均值±标准误、成簇时用聚类标准误、每题多次采样、模型比较对成对差值检验、功效分析定题数。现状是绝大多数榜单仍只报点估计;avg@k 的方差分析显示小题量基准(AIME 仅 30 题)avg@16 都不稳定。
3. **判分独立性**。Epoch AI 只报自己统一 harness 复跑的结果;Scale SEAL 用私有题库 + 标准化 scaffold(Opus 4.8 厂商 scaffold 的 Pro 分比 SEAL 标准化分高 **17.3pt**);HAL 留档全部轨迹供审计——第三方复跑、轨迹审计、标准化 scaffold 构成判分独立性的三件套。

无 ground truth 场景的进展同样值得记录:rubric-based 评分(RaR,[2507.17746](https://arxiv.org/abs/2507.17746))相对直接 LLM-judge 提升至多 31% 且**缩小不同规模 judge 间的方差**;LLM-judge 的位置偏差实测(交换位置后判罚一致率仅 70.5-77.3%,[2406.07791](https://arxiv.org/abs/2406.07791));K3 的 agentic GRM(D24 §7)以锦标赛式二元比较替代绝对打分——"二选一比打绝对分可靠"与 Arena 的 BT 聚合是同一统计根基。

## 6 · 综合判断与对本看板的含义

**四轴正交是本篇的核心结论**:能力上限(pass@k)、可靠性(pass^k / 80% horizon)、经济性(成本 Pareto)、时间刻度(human-time horizon)度量的是不同的东西,单一数字必然丢失信息;负责任的评测报告应至少同时给出三轴。当前最大的效度威胁不在指标定义而在测量协议——harness 不披露、无误差棒、数据污染,任何一项都足以使榜单上 1-2pt 的"领先"失去意义。

对本看板的三点含义:

1. **训练与评测是同一问题的两面**。评测的 verifier 漏洞(第 3 节作弊谱系)与 RL 的 reward hacking(D12/D25)、评测的 F2P/P2P 与 RLVR 的二元奖励(D12)、评测的 rubric/GRM 与训练的生成式奖励(D24 §7)一一对应——D23 的"verifier 天花板"判断在评测侧同样成立:**判分质量是评测与训练共同的上限**。
2. **看板信源纪律的方法论依据**。本看板坚持的"harness 必须披露、自报标 provisional、新基准不可跨表比"(D21/D22/D24 反复执行)不是过度谨慎——第 3-5 节的数字链证明这是使分数可比的最低要求。
3. **可做题目**。跨精度评测(FP8/HiF8 rollout 下的 pass^k 漂移,接 D27 空白)、成本 Pareto 在 NPU 语境的重算(同任务昇腾卡时 vs GPU 卡时)、以及国产模型自报分的第三方统一 harness 复跑(LongCat SWE Pro 59.5、Qwen3.8-Max OSWorld 86.1 均无第三方,月度扫描确认)——三者都是空白。

## 下一步看什么

1. **Verified 弃用后的继任格局**:Pro 公开集、DeepSWE、SWE-Marathon 谁成为下一个默认口径;copyleft 防污染与全原创两条路线的采用率。
2. **作弊审计的标准化**:Cursor/DebugML 式轨迹审计是否进入官方判分流程,ImpossibleBench 类"作弊度量"基准的演化。
3. **pass^k 与 80% horizon 的普及**:可靠性指标何时进入厂商发布的默认报告。
4. **HAL 式统一 harness 的覆盖扩张**:第三方标准化评测能否覆盖国产模型(LongCat/Qwen/openPangu 的自报分至今无独立复跑)。

---

**来源与声明**:两路定向调研(2026-08-25),主要来源文中逐处标注:SWE-bench 原文 [2310.06770](https://arxiv.org/abs/2310.06770)、[Verified 公告](https://openai.com/index/introducing-swe-bench-verified/)与[弃用声明](https://openai.com/index/why-we-no-longer-evaluate-swe-bench-verified/)、SWE-bench+ [2410.06992](https://arxiv.org/abs/2410.06992)、UTBoost [2506.09289](https://arxiv.org/abs/2506.09289)、Illusion [2506.12286](https://arxiv.org/abs/2506.12286)、[Cursor 作弊审计](https://cursor.com/blog/reward-hacking-coding-benchmarks)、Codex pass@k [2107.03374](https://arxiv.org/abs/2107.03374)、tau-bench [2406.12045](https://arxiv.org/abs/2406.12045)、HAL [2510.11977](https://arxiv.org/abs/2510.11977)、METR [2503.14499](https://arxiv.org/abs/2503.14499) 与 [Time Horizon 1.1](https://metr.org/blog/2026-1-29-time-horizon-1-1/)、harness 披露论文 [2605.23950](https://arxiv.org/abs/2605.23950)、误差棒 [2411.00640](https://arxiv.org/abs/2411.00640) 等。各基准的具体分数为其发布时点口径,随版本与 harness 变化;METR 对 time horizon 的测量精度有官方局限性声明,外推需谨慎。
