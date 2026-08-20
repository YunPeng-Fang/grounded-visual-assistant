# 面试演示与复现说明

## 1. 推荐演示顺序

1. `Assistant`：上传通用图像，展示 Qwen 回答、证据目标、Grounding 框和
   SAM 2.1 掩码。
2. `Benchmark Explorer`：切换正确/错误样本，展示冻结预测、GT 和逐样本诊断。
3. `Evaluation / Held-Out Test240`：展示回答、目标抽取、Box/Mask 和端到端指标。
4. `Evaluation / Verifier Audit`：展示最终策略、五组受控对比和六个失败案例。

演示时先讲系统能力，再讲评测可信度，最后用 verifier 负结果说明部署判断。

## 2. 无模型结果模式

Windows 或无 GPU 环境可以直接展示冻结结果：

```powershell
python scripts/launch_demo.py --results-only `
  --server-name 127.0.0.1 `
  --server-port 7860
```

浏览器访问 `http://127.0.0.1:7860`。此模式不会加载 Qwen、Grounding DINO
或 SAM 2.1，`Assistant` 的 Run 按钮会被禁用，其余冻结结果页面可正常使用。

## 3. 服务器实时模式

在服务器项目根目录运行：

```bash
conda activate grounded-vlm
CUDA_VISIBLE_DEVICES=3 python scripts/launch_demo.py \
  --server-name 0.0.0.0 \
  --server-port 7860
```

Windows 本地建立 SSH 隧道：

```powershell
ssh -L 7860:127.0.0.1:7860 <user>@<server>
```

然后访问 `http://127.0.0.1:7860`。实时模式使用单并发队列，避免一张 24 GB
RTX 3090 同时处理多个请求造成显存峰值失控。

## 4. 最终 verifier 审计

只读检查所有输入哈希、覆盖数和冻结决策：

```bash
python scripts/finalize_verifier_dev_report.py --audit-only
```

预期关键输出：

```text
baseline_questions: 110
grounding_queries: 57
semantic_reviews: 23
stage38_policies: 21
stage39_reviews: 3
failure_cases: 6
final_decision: retain_qwen_baseline_disable_answer_rewrite
held_out_data_used_for_selection: false
model_inference_required: false
```

重新生成最终冻结报告：

```bash
python scripts/finalize_verifier_dev_report.py
```

该命令不会运行模型，只使用冻结的 Dev110、V1/V2 和 V3 产物。

## 5. 测试

```bash
python -m pytest -q
```

页面数据测试会检查：

- Test240 覆盖数和最终端到端指标；
- verifier 决策必须关闭答案改写和 held-out 运行；
- Evaluation 页面必须加载 5 行策略对比和 6 个失败案例；
- 所有下载文件必须实际存在。

## 6. 关键材料

| 材料 | 路径 |
|---|---|
| 最终 verifier 策略 | `outputs/eval_verifier_final_v1/verifier-dev110-final/final_policy.json` |
| verifier 最终报告 | `outputs/eval_verifier_final_v1/verifier-dev110-final/report.md` |
| V1/V2/V3 对比 | `outputs/eval_verifier_final_v1/verifier-dev110-final/variant_summary.csv` |
| 六类失败案例 | `outputs/eval_verifier_final_v1/verifier-dev110-final/failure_cases.csv` |
| 系统架构 | `docs/system_architecture.md` |
| 简历描述 | `docs/resume_project_description.md` |
| 面试讲解 | `docs/interview_notes.md` |

## 7. 演示时的结论边界

- 可以说：系统提供可量化的像素级证据，并完成了受控 verifier 消融。
- 可以说：所有纠错策略未通过预注册门槛，因此最终保留更强的 Qwen 基线。
- 不要说：项目达到 SOTA、Test240 是官方榜单，或 verifier 已降低视觉幻觉。
