# 简历项目描述

## 推荐项目名称

**Grounded Visual Assistant：可追溯视觉多模态问答系统**

技术栈：Qwen3-VL-8B-Instruct、Grounding DINO、SAM 2.1、PyTorch、
Transformers、Gradio、COCO

## 一句话介绍

面向通用图像问答，构建“语义回答 + 开放词汇目标定位 + 像素级分割”
端到端系统，为视觉大模型回答提供可视化、可量化的证据链。

## 中文简历版

- 独立搭建 Qwen3-VL-8B、Grounding DINO 与 SAM 2.1 端到端推理链路，
  将图像问答结果解析为结构化 `answer/evidence_targets`，生成对应目标框、
  分割掩码及运行诊断信息。
- 设计任务感知提示词和结构化输出解析策略；在冻结 Dev60 上，相比 v1
  策略将物体枚举 Macro F1 提升 11.71 个百分点、目标抽取 Micro F1
  提升 12.22 个百分点，并将 JSON Schema 有效率从 96.67% 提升至 100%。
- 构建 COCO held-out Test240 评测协议，覆盖物体枚举、存在性判断和空间关系
  三类任务；最终取得存在性准确率 95.00%、关系 Balanced Accuracy
  71.99%、目标抽取 Micro F1 84.06%，端到端完整证据成功率 59.17%。
- 实现断点续评、逐样本错误分析、指标独立回放、输入与运行配置哈希锁定，
  并开发 Gradio 可视化页面展示实时问答、Grounded-SAM-2 证据、冻结样本
  和 Dev-to-Test 泛化结果；单张 RTX 3090 平均端到端延迟 1.30 秒，
  峰值显存 19.31 GB。
- 构建隔离的 Verifier Dev110 协议，对 Grounding 置信度、语义复核和对比类别
  复核三类答案纠错方案进行 21 组受控消融；使用预注册准确率/F1/净纠错门槛，
  在所有候选均劣于基线时冻结并保留 Qwen 基线，避免将负增益模块部署上线。

## 一页简历压缩版

- 基于 Qwen3-VL-8B、Grounding DINO、SAM 2.1 构建通用视觉问答与像素级
  证据定位系统，支持结构化目标抽取、目标框/掩码生成和 Gradio 可视化。
- 在自建、冻结的 COCO held-out Test240 上实现存在性准确率 95.0%、
  目标抽取 F1 84.1%、端到端完整证据成功率 59.2%；实现断点续评、指标回放、
  哈希锁定及逐样本失败分析，单卡 RTX 3090 平均延迟 1.30 秒。
- 在隔离 Dev110 上完成 21 组 evidence verifier 消融和 V3 对比类别复核；所有
  纠错策略未通过预注册门槛，因此保留 96.36% 准确率的冻结 Qwen 基线，并将
  Grounded-SAM-2 限定为证据定位与审计模块。

## 英文简历版

**Grounded Visual Assistant | Qwen3-VL, Grounding DINO, SAM 2.1, PyTorch**

- Built an end-to-end visual QA pipeline that converts Qwen3-VL responses into
  structured answers and evidence targets, then produces open-vocabulary
  bounding boxes and pixel-level masks with Grounding DINO and SAM 2.1.
- Designed task-aware prompting and structured-output parsing; improved
  object-listing macro F1 by 11.71 points and target extraction micro F1 by
  12.22 points over the v1 policy on a frozen Dev60 split.
- Established a locked COCO held-out Test240 protocol across object listing,
  existence, and spatial relations, achieving 95.0% existence accuracy, 84.1%
  target micro F1, and 59.2% complete end-to-end evidence success.
- Implemented resumable evaluation, metric replay, artifact/config hash
  verification, per-sample failure analysis, and a Gradio dashboard; reached
  1.30 s mean latency with 19.31 GB peak memory on one RTX 3090.
- Evaluated 21 evidence-verifier policies plus a contrastive category-review
  cascade on an isolated Dev110 protocol; enforced pre-registered
  accuracy/F1/net-correction gates and retained the 96.36% Qwen baseline when
  every answer-rewrite policy regressed.

## 面试时必须补充的实验边界

- Test240 是基于 COCO val2017 构建并冻结的项目评测集，不是官方公开排行榜。
- `59.17%` 表示“答案正确且所需证据全部定位成功”的严格联合成功率。
- 已在隔离 Dev110 上受控评估三类证据纠错方案，但均未通过预注册门槛；最终
  系统只将证据用于定位和审计，因此不要声称已经证明“显著降低幻觉”。
- 与最新方法的正式公平比较仍需要 POPE、RefCOCOg 或 GroundingME 等公开
  协议；这些属于下一阶段，不应写成已经完成。
