# Dispatch 03 · 昇腾 950 硬件深读:自研 HBM、原生 FP8,与 PR/DT 分工

*2026-06-23 · NPU Frontier Dispatch · hardware / Ascend 950 / FP8 / SuperPoD*

> **TL;DR** — 昇腾 950 是第一代**用华为自研 HBM、原生支持 FP8/MXFP4** 的 Ascend,并拆成两颗:**950PR**(prefill/推荐,~Q1 2026,128GB HiBL @ ~1.6TB/s)和 **950DT**(decode/训练,~Q4 2026,144GB HiZQ HBM @ ~4TB/s + 2TB/s 互联)。对"在昇腾上做 RL"这件事,三个改变最关键:**(1) 原生 FP8** 把上期 Dispatch 02 讲的"FP8 量化 rollout"从设想变可行;**(2) 自研 HBM + 更大容量/带宽**直接松绑 910B 上 rollout/train 抢 64GB 显存的死结;**(3) PR/DT 的硬件分工**天然对应 RL 里 rollout(生成)与 train(更新)的解耦。代价是:软件栈得跟上,且 950DT 要等到年底。

接 Dispatch 02(rollout 瓶颈)。上期说的两条软件主线——FP8 量化 rollout + 异步解耦——这期看它们的**硬件对应物**:昇腾 950 几乎是照着这两条线设计的。

---

## 1 · 规格速览

| | **Ascend 950PR** | **Ascend 950DT** | 910B(参照) |
|---|---|---|---|
| 定位 | prefill / 推荐 / 推理 | decode / 训练 | 训练 + 推理 |
| 上市 | ~2026 Q1 | ~2026 Q4 | 在售 |
| 显存 | 128 GB **HiBL 1.0** | 144 GB **HiZQ 2.0 HBM** | 64 GB HBM |
| 带宽 | ~1.6 TB/s | **~4 TB/s** | ~0.4 TB/s* |
| 互联 | — | ~2 TB/s | HCCS |
| 低精度 | FP8 / MXFP8 / HiF8 / **MXFP4** | 同左 | 无原生 FP8 |
| 算力 | ~1 PFLOPS FP8 / ~2 PFLOPS MXFP4 | 同量级 | — |

> 数字来自华为 Connect 2025 与分析师/媒体口径,**provisional**,以实际出货为准。`*` 910B 带宽口径各源不一。完整卡片见 **Ascend / NPU** 标签页(已补一张 *Atlas 950 SuperPoD* 卡)。

## 2 · 三个真正重要的改变

**① 自研 HBM——绕开卡脖子,顺带把显存做大。**
910B 时代,RL 最硬的约束是单卡 **64GB HBM**:rollout 的 KV cache + 权重和 train 的优化器状态+梯度挤在一起(看 NPU 架构页的"RL 显存争用"视图)。950DT 用华为自己的 **HiZQ 2.0 HBM,144GB @ ~4TB/s**——容量翻倍多、带宽近 10×。这不只是性能,更是**供应链自主**:不再受 HBM 出口管制掣肘。

**② 原生 FP8/MXFP4——让 FP8 RL 从 PPT 变成可跑。**
910B 没有原生 FP8,做低精度要绕。950 直接支持 **FP8/MXFP8/HiF8/MXFP4**。把上期 Dispatch 02 的链条接上:Jet-RL / FP8-RL / Quantized Rollout 这些**量化 rollout** 配方,在 950 上有了硬件落点。"FP8 RL on Ascend"是一个**几乎没人做过**的选题——硬件这块拼图 2026 到位。

**③ PR/DT 分工——硬件层面的 prefill/decode(也是 rollout/train)解耦。**
华为把芯片拆成 **PR(prefill/推荐,算力密集、带宽次要)** 和 **DT(decode/训练,带宽/显存敏感)**:用便宜的 HiBL 喂 prefill,用高带宽 HBM 喂 decode/训练。这正好对上 RL 的结构——rollout 偏生成(decode)、train 偏更新——以及 AsyncFlow/RollMux 那类**解耦 RL** 的系统设计。硬件开始按"分离式"思路出货。

## 3 · 放进 SuperPoD:还是"规模补效率"

单芯片华为仍落后(节点工艺、HBM 代际、能效)。它的答案一直是**堆规模**:**Atlas 950 SuperPoD 最多 8,192 颗 950DT**(约 CM384 的 20×),UnifiedBus 互联 ~16 PB/s,系统级 ~**8 EFLOPS FP8 / 16 EFLOPS FP4**。路线图 **950(2026)→960(2027)→970(2028)**,每代算力大致翻倍,2028 目标 4 ZettaFLOPS FP4。

思路和 CM384 对标 NVL72 一脉相承(见 Ascend 标签的"scale vs efficiency"卡):**用超节点聚合补单芯片效率**。对大规模 RL 意味着:可以把 rollout 池和 train 池分到不同芯片/机柜,用大互联带宽兜住跨池通信。

## 4 · 对 RL-on-NPU 意味着什么

- **显存争用被硬件缓了一截**。144GB + 4TB/s 让 rollout/train 不必那么早就时间分片;叠加 FP8 量化 rollout(占用更小),910B 上"无 sleep-mode"的痛会明显减轻——但注意,**sleep-mode 是软件能力,950 并不自动带来它**。
- **FP8 RL 成为高新颖度选题**。硬件就位后,谁先在 950 上跑通**端到端 FP8 RL** 并公布 train-inference 一致性,谁就占住这个空白。
- **分离式 RL 有了对口硬件**。PR/DT + 大互联 = 把 rollout/train 解耦到不同资源池的天然载体,正好接 Dispatch 02 的异步线。
- **数值漂移要盯紧**。FP8/MXFP4 + NPU 上重写的算子,会放大 train-inference mismatch——这正是看板里 **align-probe** 想法的用武之地。

## 5 · 别高兴太早

- **软件是真瓶颈,不是算力**。CANN / vLLM-Ascend / MindSpeed-RL 要把 FP8 RL、sleep-mode 等价物、新算子全跟上;硬件领先不等于栈成熟(见"scale vs efficiency"卡的结论)。
- **950DT 要等到 ~Q4 2026**。训练/解码这颗最关键的,年底才出;上半年能摸到的主要是 950PR(偏推理/prefill)。
- **规格都还是厂商口径**。EFLOPS、PFLOPS、带宽都待实测复现;本看板一律按 *provisional* 标注。

## 6 · 下一步看什么

1. **CANN/vLLM-Ascend 何时把 FP8 RL 路径打通**,以及有没有 sleep-mode 的等价方案。
2. **950DT 出货后的真实 HBM 带宽/能效实测**,对照 910B 看 RL 吞吐提升。
3. **第一篇"950 + FP8 RL"的端到端工作**——这会是 RL-on-NPU 最有分量的里程碑。

---

*来源:华为 Connect 2025 主题演讲与 Ascend 路线图、TrendForce / Tom's Hardware / 分析师报道(950PR/950DT 规格、Atlas 950 SuperPoD、950→970 路线图);均为厂商/媒体口径,provisional。相关卡片见本看板 Ascend / NPU 标签页。*
