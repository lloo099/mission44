# Dispatch 01 · Seed 2.1 与开源 1M 上下文浪潮

*2026-06-23 · NPU Frontier Dispatch · frontier / Seed 2.1 / RL-on-NPU*

> **TL;DR** — ByteDance 的 **Seed 2.1 Pro 抢先版**(6 月 19 日)在 LMArena 的 Code-Arena 前端榜以 **1539 Elo** 排到全球第 8,水平约等于 Claude Opus 4.6。但它是**闭源 API 模型、没有 Ascend 移植**——它定义的是"要追的标杆",而不是"能跑在 NPU 上的模型"。对 RL-on-NPU 真正有用的,仍是同期那批**开源权重**的 1M 上下文模型(DeepSeek-V4 / MiniMax-M3 / GLM-5.2)。

这是本看板的第一期"前沿观察"(Dispatch)。计划每隔一段时间做一次这样的横向总结:把新出的模型/论文放进统一对比,再回答同一个问题——**它对"在昇腾 NPU 上做高效 RL 训练"这件事意味着什么?**

---

## 1 · 本期发生了什么

- **Seed 2.1 Pro(抢先版)· 6 月 19 日**:ByteDance Seed 团队把 Seed-2.1-Pro-Preview 放上了 LMArena 的 Code Arena 试用。前端编码子榜 **1539 Elo,全球第 8**,与 Anthropic 旗舰 **Claude Opus 4.6** 同档;React、品牌营销、数据分析工具等 7 个子类里有 5 个进全球前十。完整基准(AIME / SWE-bench / GPQA 等)官方说"未来几周"随正式版放出。
- **开源权重浪潮继续**:6 月里 **GLM-5.2(MIT)**、**MiniMax-M3 + MSA** 相继开放权重,都主打 1M 上下文;叠加 4 月的 **DeepSeek-V4**,开源前沿在 80% 的 SWE-bench Verified 一带已经咬住了上一档闭源模型。

## 2 · 把 Seed 放进对比

Seed 2.1 的完整跑分还没公布,所以对比里我用**两行**来诚实呈现:Seed **2.0 Pro** 的 2 月确证数字作为能力基线,Seed **2.1 Pro 抢先版**只填它唯一公开的 Code-Arena 一项。

| 模型 | 类型 | SWE-bench Verified | AIME 2025 | GPQA Diamond | 输出价 $/1M | Code-Arena 前端 |
|---|---|---|---|---|---|---|
| **Seed 2.1 Pro**(抢先版) | 闭源 API | 待公布 | 待公布 | 待公布 | — | **1539 Elo(第 8)** |
| Seed 2.0 Pro | 闭源 API | 76.5 | 98.3 | 88.9 | $2.37 | — |
| DeepSeek-V4-Pro | 开源权重 | 80.6 | — | 90.1 | $3.48 | — |
| MiniMax-M3 | 开源权重 | 80.5 | — | 92.7 | ~$1.20 | — |
| GLM-5.2 | 开源 (MIT) | (5 系 ~77.8) | (5 系 98.0) | (5 系 94.0) | — | — |

> 数字均为厂商/媒体/第三方口径,**provisional**;各项覆盖度不同,以官方报告为准。完整可交互版见 **Overview** 顶部的对比组件(已新增 Seed 行与一个 *Code Arena · Frontend* 指标)。

读法:Seed 2.0 Pro 当年是实打实的前沿闭源模型(AIME 98.3、SWE 76.5);Seed 2.1 抢先版在**前端编码**这一窄口径上已经摸到 Opus 4.6 的高度。它很强,但强在一个**闭源、托管、调 API** 的形态里。

## 3 · 这对 RL-on-NPU 意味着什么

一句话:**Seed 2.1 是标杆,不是载荷。**

- **不可移植**:闭源、仅 API,没有权重、也没有 vLLM-Ascend 适配。你没法把它当作在 910B 上做 RL rollout / 训练的对象。它的价值是"目标分数"——告诉你开源侧还差多少。
- **昇腾的现实载荷仍在开源侧**:真正能放到 NPU 上做 RL 后训练或评测的,是 **DeepSeek-V4**(已 vLLM-Ascend 支持)、**MiniMax-M3**(纯 GQA + MSA,解码侧最省、最好移植)、**GLM-5.2**(MIT 许可,适合做 RL 基座)。本看板的 *Ascend 就绪度* 徽标也只对这批亮灯。
- **"编码/Agent 强"恰好压在 RL 的痛点上**:Seed 2.1 的强项是多轮、长链路的前端/Agent 编码——这正是 agentic RL 的训练对象。它在闭源侧把天花板抬高,反向说明:**开源 + 昇腾**这条线如果想接近,绕不开 *长 rollout 的显存争用*(见 NPU 架构页的 "RL 显存争用" 视图)和 *异步/解耦 RL* 这两块硬骨头。

## 4 · 下一步看什么

1. **Seed 2.1 正式版的完整跑分**:尤其 SWE-bench Verified / Pro 与 AIME,看它相对 2.0 Pro 提了多少,以及和开源 80% 一带的真实差距。
2. **开源侧能否在 agentic/前端编码追上**:Code-Arena 这种多轮交互榜,是观察 agentic RL 是否真见效的好镜子。
3. **解码侧稀疏注意力的落地**:MSA / DSA / CSA 这类机制在 NPU 上重写后的数值漂移,是 *align-probe* 该盯的——也是开源模型能否在昇腾上稳定做 RL 的前提。

---

*下一期 Dispatch 会在又攒够一批新模型/论文后更新。来源:ByteDance Seed 官方博客与 LMArena 排名(Seed-2.1-Preview)、各模型技术报告/媒体报道(provisional)。*
