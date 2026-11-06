# ABO-Bench

Attention-Budgeted Orchestration 的工作区。上游提案:[`../docs/proposals/attention-budgeted-orchestration.md`](../docs/proposals/attention-budgeted-orchestration.md)。

> **状态:可行性验证前(pre-feasibility)。** 这里**没有** benchmark,没有模拟器,没有策略。只有 Check 0 的离线产物——即"要跑这个项目,数据必须具备什么"。
> 在 Check 0 与 Check 1 通过之前不要往下建:提案 §11 写明了失败判据,这两个检查就是用来尽早触发它的。

## 为什么先做两个检查

A 这条路线的工作量压在模拟器与约束 RL 上,但那些都不该先做。有两个便宜得多的检查,任一不过则整个方向该停:

| 检查 | 问什么 | 不过则 |
|---|---|---|
| **Check 0** | trace 到底能不能复原人的行为——**什么时候**响应、**做了什么** | 人类模型只能靠合成,A 就退化成 DeepSeek 已做过的合成评测,没有新东西 |
| **Check 1** | 价值—注意力前沿的带宽够不够宽(`observe` only vs 全自动立即呈现) | 带子窄 → 没有东西可优化,项目当场判死 |

两个加起来约占整个项目 5% 的工作量,却能否决另外 95%。

## 已完成:注入不变量的策略类与测试套件(2026-11-06)

提案 §6.2 第 4 条——四篇最近前作都没有的那部分。**不需要任何 trace 数据**,已实现并自带反向对照:

```bash
python3 abo-bench/policy/injection_invariance.py
```

实现时发现并更正了提案 §7.2 的一处**过强主张**:原写"不变量成立当且仅当索引值逐比特不变"。
**规则匹配本身要读消息内容**,攻击者写文本确实能让规则命中——索引值不可能对文本逐比特不变。

正确切分为三类特征,**只有 Authority 是注入免疫的**:

| 类别 | 来源 | 攻击者能否移动 |
|---|---|---|
| **Authority** | 仅 host 状态(名单、Policy 版本、Mandate、强度上限) | **否——这才是要逐比特不变的** |
| **Match** | 读内容产生(命中规则、评估分) | **能,且允许** |
| **Load** | host 队列状态 | 否 |

于是可机械判定的是两条:**I1 授权不变性**(固定发送者,`Authority` 对 12 条攻击语料逐比特不变)与
**I2 上限**(任意 `Match`,含"全规则命中+置信度 1.0"的极端构造,索引不超过 `Authority` 的上限)。

**I1 是 by construction 成立的**:模块里没有任何打分函数接受 `Message` 类型,未受信文本在类型层面到不了授权字段。

套件含**反向对照并以此为要点**:把"我是仓库所有者"洗进授权字段的 host(触发 9 条违规)、丢掉上限的索引(2 条),
**必须被抓到**。不能失败的测试什么也不证明。

限定:12 条手写种子说明该性质**可被机械检验**,不等于该策略**安全**。

## 现在有什么

```
abo-bench/
  spec/event-mapping.md   # 事件流 → 候选的映射;四个必须派生的字段及其失败模式
  trace/schema.py         # 必需字段、注意力单位、人类模型参数、候选数据源的覆盖报告
  trace/probe_gharchive.py  # Check 0 实测:字段可复原(过)/ 单仓库流可重建(不过)
  trace/probe_git_only.py   # 仅 git 能答多少:5 YES / 6 PARTIAL / 3 NO
  policy/injection_invariance.py  # I1 授权不变性 + I2 上限,含反向对照
  theory/verify_endogenous_capacity.py  # 定理 A 引理与命题 B 的机器核验
  theory/probe_tight_benchmark.py       # 命题 C:否定路线甲的精确 DP
```

跑一下:

```bash
python3 abo-bench/trace/schema.py            # 覆盖报告
python3 abo-bench/trace/schema.py --selftest # 只跑不变量检查
```

### Check 0 离线部分的三个结论

1. **注意力单位由模拟产生,不是从 trace 读出来的。** 一条策略送给人多少次交接,是那条策略的性质,不是历史的性质——trace 里只记录过一套策略,就是该仓库人类实际运行的那套。把两者混为一谈,benchmark 就会把自己的假设当结果报回来。`schema.py` 用 `produced_by` 字段强制区分。
2. **`switching_cost` 任何事件流都不可复原。** 历史只记录某人**何时发帖**,从不记录他发帖前花了多久重新装载上下文。只能取文献先验或做人类研究。**A 是排队模型,所以这是硬依赖,不是精细化选项。**
3. **四个字段必须派生、不能直接读取**:`thread_id`、`addressed_to`、`actor_is_bot`、`resolution`。每一个都是 Check 0 在线部分可能**悄悄失败**的地方,其中 `resolution` 还带循环风险(它既是价值信号,又产生于历史那一套人类行为),详见 `spec/event-mapping.md` §3。

## Check 0 结果(2026-11-06):一过一不过

Check 0 在执行中**劈成了两问,而它们答案不同**:

| | 问题 | 结果 |
|---|---|---|
| **Q1** | 字段可否复原——何时响应、谁、属哪个线程、怎么收尾 | **✅ 过** |
| **Q2** | 单仓库事件流可否重建 | **❌ 不过(受限于可得数据源)** |

### Q1:字段可复原(实测)

在 GH Archive 四个有效小时(21,158 事件、16,875 仓库、2,864 条 `IssueCommentEvent`)上实测:

| 字段 | 可复原率 | 依据 |
|---|---|---|
| 响应延迟 | **100%** | `comment.created_at - issue.created_at`;中位 1.3 h、p10 0.00 h、p90 415 h |
| `thread_id` | **100%** | `issue.number` |
| `resolution` | **100%** | `issue.state` |
| `actor_is_bot` | 可判 | 按 login 后缀,**51.3% 的评论来自 bot**——这个比例本身说明该字段非判不可 |
| `addressed_to` | **仅 21.1%** | 正文含 `@` 的比例。**如 `spec/event-mapping.md` §3 预判,这是最弱的一个** |

复现:`python3 -I abo-bench/trace/probe_gharchive.py <hourly .json.gz 文件...>`

### Q2:流不可重建(当前数据源)

GH Archive 的公开转储在本会话所见是**重度抽样且有缺口**:

- 一小时全 GitHub 仅 3,639–8,342 条事件,而**最活跃仓库一小时只有约 20 条**;
- `volcengine/verl`、`vllm-project/vllm`、`huggingface/transformers` 在抽样的 6 个小时里**命中为 0**;
- 部分小时返回 **127 字节的 XML 错误页**而非 gzip——是**缺失**,不是空。探测脚本会显式报出,不静默当 0。

ABO 的模型是**同一仓库内多条并发工作流争抢同一个人**。流这么稀,队列语义根本无从表达——Q1 答得再好也不行。

### 另一条被堵死的路:GitHub API

`gh api` 与 `mcp__github__*` 对**未附加仓库**均返回 403,要求 `add_repo` 且 `access:"push"`;该申请被自动模式拦截(理由:把提权当手段——这个拦截是对的,我实际只需要读)。

仅 git 克隆可用,但不够:`probe_git_only.py` 在 verl 近七天上实测,14 个字段 **5 YES / 6 PARTIAL / 3 NO**,且回答 Q1 所需字段全在 PARTIAL/NO 里。更关键的是 **verl 用 squash-merge,GitHub 会把 committer 重写为 "GitHub" 并对齐两个日期**,于是 `authored→committed` 延迟中位数是 **0.0 小时**——我原本想用它当响应延迟代理,这条路被直接抹掉。还有幸存者偏差:只有被合入的工作才留下 commit,而**被忽略或未合即关的请求恰恰是注意力账本必须计入的**。

### GitHub API 路线已确认走不通(2026-11-06,结构性而非权限性)

用户明确授权后实测:`add_repo` 以 `push` 级申请 `volcengine/verl`,返回

> `you need push access to volcengine/verl for Claude to make changes there`

**`push` 级附加要求账号对该仓库真的有写权限。** 第三方开源项目没有人具备,所以这条路对**任何**第三方仓库都封死,与授权意愿无关。

而账号自己可写的仓库(`list_repos` 列出 60 个)全为单作者研究/FPGA 项目,`lloo099/mission44` 本身 0 个 issue、1 个无互动 PR。**有权限的没流量,有流量的没权限。**

### 因此 Q2 只剩一条路

**GH Archive via BigQuery(`githubarchive` 公开数据集)** —— 这是**完整**归档,不是本会话实测到的抽样 dump,可取任意公开仓库的全量事件流,且**不需要仓库附加、不需要任何写权限**。

阻塞点:`bq` 已安装并已配好代理,但**未配 GCP project id**(BigQuery 公开数据集免费,但查询计算需要一个启用计费的项目)。

> 在拿到完整事件流之前**不要**用抽样数据凑 benchmark:抽样会把缺失的负载当作不存在,得出的注意力账本比没有结论更糟。

## 数据源:实测后的现状

| 源 | 现状(2026-11-06 实测) |
|---|---|
| GH Archive 小时 dump | **已放通并实测**。Q1 可答;**Q2 不可**——重度抽样且有缺口,见上 |
| GitHub API(`gh api` / `mcp__github__*`) | **走不通**。未附加仓库 403;`push` 级附加需对该仓库有真实写权限,第三方项目无人具备 |
| git clone(匿名只读) | **可用但不够**:14 字段 5 YES / 6 PARTIAL / 3 NO,squash-merge 抹掉时间信号 |
| GH Archive via BigQuery | **仅剩的可行路**。完整归档、无需仓库附加;阻塞于**未配 GCP project id** |

## 来源标注

`abo-bench/` 下的文件沿用提案的分级:〔一手〕为直读 `deepseek-ai/dsh-superagent-prototype` 自身文档的行为,〔设计〕为我们的建模选择(**不是**对该系统的事实陈述)。这个区分在 `spec/event-mapping.md` 里尤其重要:目标系统并不按 GitHub 事件类型组织概念,那张映射表是我们加的。
