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

## 现在有什么

```
abo-bench/
  spec/event-mapping.md   # 事件流 → 候选的映射;四个必须派生的字段及其失败模式
  trace/schema.py         # 必需字段、注意力单位、人类模型参数、候选数据源的覆盖报告
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

## 数据源:三条路,都未验证

`schema.py` 里的 `SOURCE_PROFILES` 全部标 `verified: False`,这是故意的——它们是待证假设,不是事实。

| 源 | 现状 |
|---|---|
| GH Archive via BigQuery | `bq` 已装但**未配 project id** |
| GH Archive 小时 dump | `data.gharchive.org` 当前不在允许域名内 |
| GitHub REST API | 现在就能用,但本会话的仓库授权只覆盖一个仓库 |

## 来源标注

`abo-bench/` 下的文件沿用提案的分级:〔一手〕为直读 `deepseek-ai/dsh-superagent-prototype` 自身文档的行为,〔设计〕为我们的建模选择(**不是**对该系统的事实陈述)。这个区分在 `spec/event-mapping.md` 里尤其重要:目标系统并不按 GitHub 事件类型组织概念,那张映射表是我们加的。
