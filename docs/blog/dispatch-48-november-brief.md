# Dispatch 48 · 11 月前沿速报：Agentic RL 与大模型

*截至 2026-11-12*

## Agentic RL

- **TriGRPO（11 月 2 日）** 在匹配的决策状态上复用已发现的成功动作，生成分支并比较结果，为长程任务中的中间决策分配信用。[论文](https://arxiv.org/abs/2611.02706)
- **Targeted Guidance（11 月 3 日）** 在多轮软件工程环境中，对失败回合提供文本反馈，并将自蒸馏与按标准计算的 RL 奖励结合。[论文](https://arxiv.org/abs/2611.05266)
- **Agentic-OPD（11 月 9 日）** 从同一次训练的历史 checkpoint 中选教师，把蒸馏集中在学生失败而教师表现更好的任务上。[论文](https://arxiv.org/abs/2611.12799)

三篇均为预印本。跟踪长程训练时，可以分别记录分支结果、失败回合与各 checkpoint 的任务覆盖。

## 大模型

- **Qwen4-Max-Preview（11 月 11 日）** 支持图像和视频理解及工具调用。阿里云通过 Token Plan 提供限时预览，至北京时间 11 月 25 日 24:00。[阿里云文档](https://help.aliyun.com/en/model-studio/qwen4-max-preview)
- **LongCat-2.5** 已发布 MIT 权重。官方模型卡给出 1.6T 总参数、约 48B 激活，面向长程多模态任务。[官方模型卡](https://huggingface.co/meituan-longcat/LongCat-2.5)
- **Claude Fable 5.2 / Mythos 5.2（11 月 9 日）** 共享底层模型。Fable 已向用户开放，Mythos 的直接访问需申请验证项目。[官方公告](https://www.anthropic.com/claude-fable-and-mythos-5-2)
- **Kimi K3.1 Pro** 权重已可下载，采用专用许可。官方模型卡确认将 K3 的全注意力 Gated MLA 层替换为 Gated DSA。[官方模型卡](https://huggingface.co/moonshotai/Kimi-K3.1-Pro)

昇腾上的训练配置和实验指标可继续查阅 [910B 异步 RL 工程笔记](#blog/dispatch-47-910b-async-rl)。
