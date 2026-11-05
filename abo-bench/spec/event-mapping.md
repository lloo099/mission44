# Check 0 规格:事件流 → 候选的映射

> Check 0 离线部分的产物之一 · 2026-11-05 · 配套 `../trace/schema.py`
> 上游提案:`../../docs/proposals/attention-budgeted-orchestration.md`

本文件回答一个问题:**要重放一条 GitHub 事件流,哪些事件会变成需要 router 做准入判断的候选,哪些不会?**

弄错这件事,后面的注意力账本全部失真——把 CI 回调当成需要人决策的候选,会凭空造出大量不存在的注意力消耗。

## 0. 来源标注

| 标注 | 含义 |
|---|---|
| 〔一手〕 | 直读 `deepseek-ai/dsh-superagent-prototype` 的 `docs/002-authorization.md`、`003-execution.md`、`007-architecture.md` |
| 〔设计〕 | 我们自己的建模选择。**不是**对该系统的事实陈述,它并不按 GitHub 事件类型组织这些概念 |

## 1. 目标系统的真实分类(一手)

`002-authorization.md` 描述 `off` 档时,顺带给出了候选的分类依据:该档下"没有任何推断性的东西成为候选,这覆盖**未被指名的消息、bot 消息、以及不在名单上的人的后续**;显式请求仍然有效"。

由此可以读出四类,以及它们各自的准入条件〔一手〕:

| 类别 | `off` | `observe`(默认) | `auto` |
|---|---|---|---|
| **显式请求**(被允许的人指名提出) | 有效 | 有效 | 有效 |
| **未被指名的消息** | 不成为候选 | 评估,正向结果记为 `would accept`,**不接受** | 正向结果真正接受 |
| **bot 消息** | 不成为候选 | 同上,且 bot 须在 bot 名单上 | 同上 |
| **不在名单上的人的后续** | 不成为候选 | 同上 | 同上 |

另外三条机制约束,直接决定重放时哪些事件**不触发**准入判断〔一手〕:

1. **CI 结果是 Input,不是新 Eval。** `003-execution.md`:CI 结果这类 Input 要等一个普通 worker 槽;`007-architecture.md`:Engine 把 Trigger **扇出给已订阅的 Ticket**,或在其主 Thread 上排一个 Eval。已订阅即走前者,不经准入。
2. **订阅线程上的消息绕过 gate 但不授予任何新权限。** `002`:它作为上下文送达。所以它消耗注意力为零(不需人决策),但可能改变 Ticket 行为。
3. **`blocked` / `failed` 之后,CI 与 bot 消息被记录但不送达。** `003`。重放必须复现这一条,否则会高估失败 Ticket 的注意力消耗。

还有一条影响 trace 完整性的〔一手〕:被通知的条目若持续读取失败,会在首次失败一天后被丢弃;GitHub 的 404 会重试,因为它也可能意味着失去访问权限。即**真实系统会丢事件**,一个无损重放其实比真实系统乐观。

## 2. GitHub 事件类型 → 类别(设计)

下表是我们的映射〔设计〕。目标系统并不按 GitHub 事件类型组织,它只看"消息是否指名、发送者是否在名单上、是否 bot"。

| GitHub 事件 | 映射为 | 理由 |
|---|---|---|
| `IssuesEvent` (opened) | 显式请求 或 未被指名的消息 | 取决于正文是否指名 agent;**这是最关键的派生字段**,见 §3 |
| `IssueCommentEvent` | 同上;若在已订阅 Thread 上则为上下文送达 | 需先判断该 Thread 是否已有 Ticket |
| `PullRequestEvent` (opened) | 显式请求 或 未被指名 | 同 Issues |
| `PullRequestReviewEvent` / `...ReviewCommentEvent` | 上下文送达(若该 PR 由 Ticket 产出)否则未被指名 | 对自己产出的 PR 的 review 是 Input |
| `CheckRunEvent` / `StatusEvent` / `WorkflowRunEvent` | **Input,不经准入** | §1 第 1 条〔一手〕 |
| 任何 `actor.type == "Bot"` 的上述事件 | bot 消息 | 需 bot 名单〔一手〕 |
| `PushEvent` | **不映射** | 不是请求,也不是对 Ticket 的输入 |
| `ForkEvent` / `WatchEvent` / `StarEvent` | **不映射** | 无人需要行动 |
| `DeleteEvent` / `CreateEvent` (branch/tag) | **不映射** | 同上 |

**未映射的事件仍须保留在 trace 里**,因为它们是估计可用窗口的证据:一个人在推代码,说明他在线。

## 3. 四个必须派生、不能直接读取的字段

`schema.py` 的 `DERIVED_NOT_GIVEN` 列的就是这四个。它们是 Check 0 在线部分**最可能悄悄失败**的地方:

| 字段 | 为什么不能直接读 | 失败模式 |
|---|---|---|
| `addressed_to` | 事件里没有"指名"这个字段。要从正文判断(@提及、固定前缀、或仓库约定) | 判错会整体移动显式/推断的比例,而这两类的准入规则完全不同 |
| `thread_id` | Eval 按 Thread 批处理〔一手〕,但 GitHub 的 issue / PR / review 线程结构不统一 | 分组错了,批处理行为与上下文切换计数同时失真 |
| `actor_is_bot` | `actor.type` 不可靠:很多自动化以普通用户身份发帖 | 低估 bot 流量 → 高估真实人类负载 |
| `resolution` | 需要跟到 Thread 终态(合入 / 关闭 / 无人理) | 这是**价值信号本身**,判错则前沿的纵轴无意义 |

**注意最后一条的循环风险**:我们用 `resolution` 当价值,但 `resolution` 是在历史那一套人类行为下产生的。一个历史上被关掉的 issue,不等于它在别的策略下没有价值。这是提案 §12.2 列的"价值归因是代理指标"在数据层的具体样子,不是能靠更好的派生规则消掉的。

## 4. 本规格不覆盖的

- **Feishu/飞书 Source**〔一手:README 称可选〕。公开 trace 里没有对应物,所以 ABO-Bench 只做 GitHub 单 Source。这会低估真实系统的注意力消耗(少了一个入口)。
- **Mandate 的 revalidation 逻辑**〔一手:002 按准入方式分六种情形〕。Check 1 的两条极端策略不需要它;到 P3 建模拟器时必须补。
- **panel 的呈现规则**。它是被优化的对象,不是重放的输入。
