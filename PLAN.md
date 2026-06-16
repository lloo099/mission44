# 项目方案:GRPO-on-Ascend 基准 + 精度对齐套件
### (working title) **AscendRL-Bench: A Reproducible Benchmark & Numerical-Alignment Toolkit for RL Post-Training on Ascend NPU**

> 版本 v1 · 2026-06-15 · 配套报告见 `REPORT.md`

---

## 1. 一句话定位
做一套**可复现的、跨平台(NVIDIA↔Ascend)对照**的 GRPO/RLVR 训练基准 + **数值精度对齐诊断工具**,把"在 Ascend 上做 RL 到底行不行、慢多少、准不准"这件目前零散、口口相传的事,变成有数据、有脚本、可一键复现的公共资产。

## 2. 为什么是这个项目(动机 & 时机)
- **痛点真实**:社区里 Ascend 上的 RL 吞吐/精度数据是零散的(verl discussion #900、各家博客),没有统一基准。研究者无法回答"我换到 NPU 会损失多少、要改什么"。
- **门槛低**:单台 Atlas 800T A2(8×910B)即可做 0.5B–7B GRPO,**不需要超节点**。
- **天花板可延伸**:基准做扎实后,天然延伸到更硬的课题(sleep-mode、FP8 RL、异步 RL)——是后续工作的"地基"。
- **可发表 + 可落地**:既能写成 benchmark/systems 论文(MLSys/NeurIPS D&B track 风格),也能作为开源工具被 verl/TRL 社区直接用。

## 3. 目标与范围
**In scope(做)**
1. 在 Ascend(910B)与 NVIDIA(A100/H800,作对照)上,用**同一套配置**跑 GRPO/RLVR,产出标准化指标。
2. **精度对齐诊断**:逐层/逐算子对比 logits、advantage、KL、reward、loss 曲线,定位 NPU 上的数值漂移来源。
3. 一键复现脚本 + 公开结果表(吞吐、显存、MFU、收敛曲线、最终精度)。

**Out of scope(暂不做)**:超大规模(>32B)、超节点、改 CANN 底层算子、写新 RL 算法。

## 4. 技术方案
### 4.1 训练栈选型
- **框架**:**verl**(一方 Ascend 支持 + 主流)为主;**TRL**(910B GRPO 已验证)做交叉验证。
- **推理/rollout**:**vLLM-Ascend**(NPU 侧)对 vLLM(GPU 侧)。
- **算法**:GRPO 为基线;加 **Dr.GRPO**(去长度偏置)、**DAPO** 作为变体(报告显示 DAPO 在 Ascend 上漂移更大,正好是研究点)。

### 4.2 模型与数据
- **模型**(由小到大):Qwen2.5-0.5B / 1.5B / 7B-Instruct(Ascend 生态对 Qwen 支持最好)。
- **任务/奖励**(用可验证奖励 RLVR,排除 RM 噪声便于对齐):
  - 数学:**GSM8K**、**MATH**(答案精确匹配)
  - 代码:**MBPP / HumanEval** 子集(单元测试通过率)
- **评测**:AIME-2024(小样本)、GSM8K/MATH test、pass@1。

### 4.3 精度对齐诊断(核心创新点)
建一个 `align-probe` 工具,在两平台用**相同 seed + 相同 prompt batch**:
1. 固定 rollout(同样的生成序列),只对比 **training forward**:logits / log-probs / KL / advantage / loss 的逐 step 偏差(MAE、相对误差)。
2. 二分定位:关掉 fused attention、关掉 FP16 累加、换优化器实现,看哪一步让曲线分叉。
3. 输出"漂移来源归因报告"(attention 融合算子 / 混合精度累加 / RNG / 优化器 哪个贡献最大)。

### 4.4 度量指标(标准化)
| 维度 | 指标 |
|---|---|
| 吞吐 | tokens/s、samples/s、每步 wall-clock |
| 效率 | MFU、显存峰值、rollout vs train 时间占比 |
| 收敛 | reward 曲线、KL、entropy、step-to-target |
| 精度 | 最终 pass@1 / 准确率,跨平台差值 |
| 稳定性 | 是否熵坍缩、是否发散、可复跑性(多 seed 方差) |

## 5. 里程碑(建议 8–10 周,单人/小组)
| 阶段 | 周 | 产出 |
|---|---|---|
| M0 环境 | 1 | Atlas 800T A2 + CANN + torch_npu + vLLM-Ascend + verl 跑通 smoke test |
| M1 单平台基线 | 2–3 | Ascend 上 Qwen2.5-0.5B/1.5B GRPO 在 GSM8K 收敛,出第一条 reward 曲线 |
| M2 跨平台对照 | 4–5 | 同配置在 GPU 上跑,产出吞吐/显存/MFU/精度对照表(0.5B→7B) |
| M3 精度对齐工具 | 6–7 | `align-probe` 完成,出漂移归因报告(定位主要来源) |
| M4 变体与扩展 | 8 | 加 Dr.GRPO/DAPO,验证报告里"DAPO 漂移更大"的假设 |
| M5 整理发布 | 9–10 | 一键复现脚本、结果网站(复用本 dashboard)、技术报告草稿 |

## 6. Baselines & 对照
- **平台对照**:Ascend 910B vs NVIDIA A100/H800(同模型同配置)。
- **框架对照**:verl vs TRL(同平台,验证结论不是框架偏置)。
- **算法对照**:GRPO vs Dr.GRPO vs DAPO。
- **外部参照**:verl discussion #900 的吞吐比(0.38–0.59)、MindSpeed-RL 报告——用来 sanity-check 我们的数。

## 7. 算力估算(粗算)
- 主力:**1 台 Atlas 800T A2(8×910B,512GB HBM 合计)** 足够 0.5B–7B GRPO。
- 对照:**1 台 8×A100/H800** 节点。
- 时长:7B GRPO 在 GSM8K 单次实验约数百~上千 step;按 8×910B,**一次 7B 实验 ~0.5–2 天**。整个项目实验量约 20–40 次跑(含多 seed/多模型/多算法)→ **NPU 机时约 2–4 周、GPU 对照约 1–2 周**。
- 0.5B/1.5B 模型可在更小切片上快速迭代,降低成本。

## 8. 风险与缓解
| 风险 | 缓解 |
|---|---|
| 缺 vLLM sleep mode → 显存吃紧、需 offload | 先用小模型;rollout/train 分时;记录这本身就是有价值的发现 |
| 算子覆盖/FlashAttention 缺失导致跑不通 | 用 `npu_fusion_attention`;遇缺算子记录到"移植清单"(也是产出之一) |
| 精度漂移定位困难 | 固定 seed + 固定 rollout,只比 forward,缩小搜索空间 |
| 拿不到 Ascend 机器 | 华为云 ModelArts / 昇腾社区算力券;或先在 GPU 把工具链做好,Ascend 上只跑验证 |
| 结论"NPU 就是慢" 不够 novel | 价值在**归因+工具+可复现**,不在结论本身;诊断工具是核心贡献 |

## 9. 可发表 / 影响点
- **Benchmark & Dataset / Systems 论文**:首个跨平台、可复现的 RL-on-NPU 基准 + 精度归因方法论。
- **开源工具**:`align-probe` 可直接贡献回 verl/TRL,社区采用率 = 影响力。
- **后续衍生**(本项目作地基):
  1. Ascend sleep-mode / colocated hybrid engine(系统课题)
  2. FP8 RL training on Ascend
  3. 异步 off-policy RL(AReaL ascend 分支 + ExGRPO 回放)

## 10. 第一周 checklist(可立即动手)
- [ ] 申请/确认 Ascend 机器(910B,CANN 8.x)
- [ ] 装 torch_npu + vLLM-Ascend + verl(Ascend 教程)
- [ ] 跑通 Qwen2.5-0.5B 的 vLLM-Ascend 推理 smoke test
- [ ] 用 verl 在 GSM8K 上启动一个 0.5B GRPO 的 100-step 试跑
- [ ] GPU 侧装同版本 verl/vLLM,准备对照
- [ ] 把每次跑的指标写进本 dashboard 的 `data/feed.json` 风格 JSON,持续积累

---

### 如果想换方向
- 想要**更硬核系统课题** → 「Ascend sleep-mode / hybrid engine」(天花板最高)
- 想要**最快出成果** → 「单节点 R1-Zero 复现 cookbook」(M1 基本就够)
- 想要**前沿研究味** → 「Ascend 上的 FP8 RL 训练」(几乎无人做)
