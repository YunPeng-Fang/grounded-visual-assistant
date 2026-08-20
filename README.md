# Grounded Visual Assistant

An evidence-grounded visual question answering system built with Qwen3-VL,
Grounding DINO, and SAM 2.1. It converts a visual answer into structured
evidence targets, localizes them with open-vocabulary boxes, and produces
pixel-level masks for inspection and evaluation.

**Project status:** interview-ready mainline completed and frozen. The final
system keeps Qwen's answer unchanged and uses Grounded-SAM-2 for localization
and audit. All tested answer-rewrite policies were rejected on the isolated
development protocol because they did not improve the frozen baseline.

## Highlights

- End-to-end local inference with structured `answer/evidence_targets`, boxes,
  masks, diagnostics, and a Gradio dashboard.
- Locked Dev/Test evaluation with resumable execution, metric replay, source
  hashes, failure attribution, and no post-test tuning.
- Public POPE Full9000 evaluation plus COCO held-out and cross-dataset hard-set
  evaluation.
- Controlled V1/V2/V3 verifier ablations with pre-registered acceptance gates;
  negative results are retained instead of being hidden behind selected demos.

## Final Results

| Protocol | Result |
| --- | --- |
| COCO held-out Test240 | 95.00% existence accuracy; 84.06% target micro F1 |
| Complete answer + evidence success | 59.17% |
| POPE Full9000 | 88.51% accuracy; 87.65% F1; 100% strict parse rate |
| Cross-dataset Hard Test400 | 57.50% exact accuracy; 72.59% mean task score |
| Verifier Dev110 baseline | 96.36% accuracy; answer rewriting disabled |
| RTX 3090 Test240 runtime | 1.30 s mean latency; 19.31 GB peak memory |

Test240 is a project-specific frozen protocol, not an official leaderboard.
POPE is reported as a Qwen baseline, not as a claimed improvement over other
methods. Evidence improves inspectability, but the current experiments do not
show that post-hoc answer rewriting improves accuracy.

Project-facing documentation:

- [Architecture](docs/system_architecture.md)
- [Resume bullets](docs/resume_project_description.md)
- [Interview notes](docs/interview_notes.md)
- [Demo reproduction](docs/interview_demo_reproduction.md)
- [GitHub upload guide](docs/github_upload_guide.md)

## Goal

Build a general visual multimodal system that can:

- answer questions about arbitrary images,
- ground mentioned objects to visual evidence,
- segment grounded regions,
- evaluate unsupported visual claims without silently rewriting answers,
- provide a clean demo and evaluation pipeline for resume/interview use.

## Recommended Model Plan

Start with a stable VLM baseline:

- Default starter: `Qwen/Qwen3-VL-8B-Instruct`
- Stable fallback: `Qwen/Qwen2.5-VL-7B-Instruct`
- Lightweight comparison later: MiniCPM-V or another small VLM

On your server, use the mostly idle GPU:

```bash
CUDA_VISIBLE_DEVICES=3 python scripts/run_vlm_baseline.py \
  --image data/demo_images/example.jpg \
  --question "What objects are visible in this image?"
```

## Project Structure

```text
grounded-visual-assistant/
  configs/
    default.yaml
    demo.yaml
    grounding_answer_verifier_v1.yaml
    grounding_answer_verifier_v2.yaml
    grounded_sam2.yaml
    live_prompt_policy_v1.yaml
    live_prompt_policy_v2.yaml
    verifier_dev_grounding_v1.yaml
    verifier_dev_semantic_review_v1.yaml
  data/
    demo_images/
    verifier_dev_v1/
  docs/
    implementation_steps.md
    interview_notes.md
    project_plan.md
    resume_project_description.md
    system_architecture.md
  outputs/
    answers/
    logs/
  scripts/
    analyze_pope_errors.py
    batch_eval_grounded_sam2.py
    batch_eval_live_pipeline.py
    batch_eval_pope.py
    batch_eval_pope_verifier.py
    batch_eval_pope_verifier_v2.py
    batch_eval_verifier_dev.py
    batch_ground_verifier_dev.py
    batch_review_verifier_dev.py
    batch_eval_vlm.py
    build_verifier_dev_v1.py
    build_coco_grounding_gt.py
    build_eval_splits.py
    build_eval_v0.py
    check_grounded_sam2_env.py
    compare_grounding_score_sweeps.py
    compare_live_prompt_policies.py
    eval_grounding_score_sweep.py
    install_grounded_sam2.sh
    eval_grounded_sam2_coco.py
    export_portable_live_predictions.py
    finalize_locked_live_test_report.py
    launch_demo.py
    lock_live_prompt_policy.py
    prepare_pope_data.py
    run_grounding_answer_verifier.py
    run_grounded_sam2.py
    run_vlm_baseline.py
  src/
    grounded_visual_assistant/
      demo.py
      evaluation.py
      coco_grounding_evaluation.py
      grounded_sam2.py
      grounding_evaluation.py
      grounding_answer_verifier.py
      live_pipeline_evaluation.py
      live_pipeline_prompting.py
      live_prompt_policy_comparison.py
      pope_dataset.py
      pope_error_analysis.py
      pope_evaluation.py
      pope_semantic_verifier_evaluation.py
      pope_verifier_evaluation.py
      semantic_answer_verifier.py
      verifier_dev_dataset.py
      verifier_dev_evaluation.py
      verifier_dev_grounding.py
      verifier_dev_semantic_review.py
      live_prompt_policy_lock.py
      live_test_reporting.py
      io_utils.py
      prompts.py
      vlm_baseline.py
```

## Setup

The server uses NVIDIA driver 550.135 and supports CUDA runtimes up to 12.4.
Install the locked CUDA 12.4 PyTorch build before the remaining dependencies:

```bash
conda create -n grounded-vlm python=3.10 -y
conda activate grounded-vlm
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements-torch-cu124.txt
python -m pip install -r requirements.txt
```

Do not use an unqualified `pip install torch` on this server. A newer CUDA wheel
can fail with `The NVIDIA driver on your system is too old (found version
12040)`. Verify `torch.version.cuda == 12.4` and
`torch.cuda.is_available() == True` before loading the model.

For the full online/offline installation, repair steps, and checks, see
[docs/environment_setup.md](docs/environment_setup.md). For local model
deployment, see [docs/model_deployment.md](docs/model_deployment.md).

## Stage 1: VLM Baseline

Put several images under:

```text
data/demo_images/
```

Run:

```bash
python scripts/run_vlm_baseline.py --image data/demo_images/example.jpg --question "Describe this image."
```

The result will be saved to:

```text
outputs/answers/
```

Each JSON record contains:

- image path
- question
- answer
- model id
- latency
- timestamp
- device information

## Next Stages

1. Build the first evaluation set:

   ```bash
   python scripts/build_eval_v0.py --num-images 100
   ```

   If a campus proxy or VPN causes a COCO HTTPS certificate mismatch, use the
   official HTTP endpoint with checksum validation:

   ```bash
   python scripts/build_eval_v0.py --num-images 100 --transport http
   ```

2. Run a five-sample smoke test on GPU 3:

   ```bash
   CUDA_VISIBLE_DEVICES=3 python scripts/batch_eval_vlm.py \
     --max-samples 5 \
     --max-new-tokens 64 \
     --local-files-only
   ```

3. Run the full resumable evaluation:

   ```bash
   CUDA_VISIBLE_DEVICES=3 python scripts/batch_eval_vlm.py \
     --max-new-tokens 64 \
     --local-files-only
   ```

   Results are written under `outputs/eval_v0/<dataset__model>/` as
   `predictions.jsonl`, `errors.jsonl`, `metrics.json`, and `run_config.json`.
   Rerunning the same command skips completed sample IDs.

4. Install the official Grounded-SAM-2 repository and run text-guided
   detection plus segmentation. See
   [docs/grounded_sam2_deployment.md](docs/grounded_sam2_deployment.md).

   The official source is staged under `third_party/Grounded-SAM-2`. After its
   two model assets have been uploaded to the offline server, install and test:

   ```bash
   bash scripts/install_grounded_sam2.sh
   CUDA_VISIBLE_DEVICES=3 python scripts/run_grounded_sam2.py \
     --image data/eval_v0/images/000000230993.jpg \
     --prompt "person. umbrella. backpack." \
     --local-files-only
   ```
5. Run the five-image oracle grounding smoke test:

   ```bash
   CUDA_VISIBLE_DEVICES=3 python scripts/batch_eval_grounded_sam2.py \
     --max-images 5 \
     --local-files-only
   ```

6. Resume the same run over all 100 unique images:

   ```bash
   CUDA_VISIBLE_DEVICES=3 python scripts/batch_eval_grounded_sam2.py \
     --local-files-only
   ```

   The evaluator loads both models once, skips completed image IDs, and reports
   class-aware Precision/Recall/F1, mean matched box IoU, mAP50, per-category
   metrics, stage latency, throughput, errors, and CUDA peak memory. Predicted
   masks are retained as COCO RLE; mask quality is not scored until COCO
   segmentation ground truth is added.

7. Restore full COCO instances for every prompted image/category pair. This
   includes small objects that `eval_v0` intentionally omitted when constructing
   its VLM questions:

   ```bash
   python scripts/build_coco_grounding_gt.py
   ```

8. Reuse the completed `predictions.jsonl` and run standard COCO bbox and
   segmentation evaluation; no model inference is repeated:

   ```bash
   python scripts/eval_grounded_sam2_coco.py --require-complete
   ```

   The evaluator exports COCO result JSON files and reports AP@[0.50:0.95],
   AP50, AP75, small/medium/large AP, AR, and per-category AP for both boxes and
   masks. Results are oracle-conditioned and must not be presented as full
   80-class detector COCO AP.

9. Build the immutable 20-image development and 80-image test splits:

   ```bash
   python scripts/build_eval_splits.py
   ```

   The split is deterministic with seed 2026, approximately stratifies
   category and COCO object-size presence, and keeps singleton categories in
   the test set whenever possible.

10. Run class-aware NMS ablations on the development split only:

   ```bash
   CUDA_VISIBLE_DEVICES=3 python scripts/batch_eval_grounded_sam2.py \
     --image-ids data/eval_v0/splits/dev_image_ids.json \
     --box-threshold 0.4 \
     --text-threshold 0.3 \
     --nms-iou-threshold 0.6 \
     --run-name dev__box-0.40__text-0.30__nms-0.60 \
     --local-files-only
   ```

   Then evaluate that run with the same split:

   ```bash
   python scripts/eval_grounded_sam2_coco.py \
     --predictions outputs/eval_grounding_v0/dev__box-0.40__text-0.30__nms-0.60/predictions.jsonl \
     --image-ids data/eval_v0/splits/dev_image_ids.json \
     --require-complete
   ```

   The completed dev experiment suppressed 0/118 boxes at NMS 0.60 and changed
   neither bbox nor mask AP. Same-class box IoU never exceeded 0.295, so NMS
   0.50 and 0.70 are skipped and NMS remains disabled for threshold tuning.

11. Run one low-threshold dev inference and evaluate several box thresholds
    offline from the same candidates:

   ```bash
   CUDA_VISIBLE_DEVICES=3 python scripts/batch_eval_grounded_sam2.py \
     --image-ids data/eval_v0/splits/dev_image_ids.json \
     --box-threshold 0.25 \
     --text-threshold 0.30 \
     --run-name dev__box-0.25__text-0.30__nms-none \
     --local-files-only

   python scripts/eval_grounding_score_sweep.py \
     --predictions outputs/eval_grounding_v0/dev__box-0.25__text-0.30__nms-none/predictions.jsonl
   ```

   This evaluates box score cutoffs 0.25/0.30/0.35/0.40/0.45 without repeating
   inference. Validate the offline 0.40 result against the existing 0.40
   baseline. Box 0.30 retains the same AP as 0.25 while removing all 46
   unmapped `object` labels, so it is the current provisional choice.

12. Repeat the low-box run for text thresholds 0.20 and 0.40, sweep box scores,
    and combine all 15 configurations:

   ```bash
   for text in 0.20 0.40; do
     CUDA_VISIBLE_DEVICES=3 python scripts/batch_eval_grounded_sam2.py \
       --image-ids data/eval_v0/splits/dev_image_ids.json \
       --box-threshold 0.25 \
       --text-threshold "${text}" \
       --run-name "dev__box-0.25__text-${text}__nms-none" \
       --local-files-only

     python scripts/eval_grounding_score_sweep.py \
       --predictions "outputs/eval_grounding_v0/dev__box-0.25__text-${text}__nms-none/predictions.jsonl"
   done

   python scripts/compare_grounding_score_sweeps.py
   ```

   Lock one box/text pair on dev, then evaluate it once on the 80-image test
   split.
13. Replace oracle categories with categories parsed from the saved Qwen3-VL
    `object_listing` answers. Start with Dev20 and keep the selected detector
    thresholds fixed:

    ```bash
    CUDA_VISIBLE_DEVICES=3 python scripts/batch_eval_grounded_sam2.py \
      --prompt-source vlm \
      --vlm-predictions outputs/eval_v0/eval_v0__qwen3-vl-8b-instruct/predictions.jsonl \
      --image-ids data/eval_v0/splits/dev_image_ids.json \
      --box-threshold 0.30 \
      --text-threshold 0.30 \
      --run-name dev__vlm-prompt__box-0.30__text-0.30__nms-none \
      --local-files-only
    ```

    The run records VLM prompt precision/recall/F1, empty prompts, hallucinated
    and missed categories, Grounded-SAM-2 metrics, and additive stage latency.
    Empty VLM prompts remain in the benchmark as zero-detection samples. Evaluate
    boxes and masks on the same split after inference:

    ```bash
    python scripts/eval_grounded_sam2_coco.py \
      --predictions outputs/eval_grounding_v0/dev__vlm-prompt__box-0.30__text-0.30__nms-none/predictions.jsonl \
      --image-ids data/eval_v0/splits/dev_image_ids.json \
      --output-dir outputs/eval_grounding_v0/dev__vlm-prompt__box-0.30__text-0.30__nms-none/coco_eval/dev \
      --require-complete
    ```

14. Generate the stage-attributed Dev20 failure report without model inference:

    ```bash
    python scripts/analyze_vlm_grounding_failures.py \
      --predictions outputs/eval_grounding_v0/dev__vlm-prompt__box-0.30__text-0.30__nms-none/predictions.jsonl
    ```

    The `failure_analysis` directory contains `summary.json`, `per_image.jsonl`,
    `per_image.csv`, and `report.md`. It separates misses caused by absent VLM
    prompt categories from misses that remain after a category was prompted.
15. Generate ontology-constrained JSON categories on Dev20. Run five images with
    the final run name, then resume the same directory over all 20 images:

    ```bash
    CUDA_VISIBLE_DEVICES=3 python scripts/batch_generate_grounding_prompts.py \
      --image-ids data/eval_v0/splits/dev_image_ids.json \
      --max-new-tokens 128 \
      --max-images 5 \
      --run-name dev__qwen3-vl__coco80-json-v1 \
      --local-files-only

    CUDA_VISIBLE_DEVICES=3 python scripts/batch_generate_grounding_prompts.py \
      --image-ids data/eval_v0/splits/dev_image_ids.json \
      --max-new-tokens 128 \
      --run-name dev__qwen3-vl__coco80-json-v1 \
      --local-files-only
    ```

    Inspect category F1, strict JSON rate, schema-valid rate, and the
    `hit_max_new_tokens` rate before running Grounded-SAM-2. The structured
    predictions file is directly compatible with `--vlm-predictions`.
16. If structured Dev20 prompt quality improves, run Grounded-SAM-2 with the
    fixed box/text thresholds and compare it with the free-text VLM-prompt run.
    Freeze the better prompt method before running Test80.
17. Evaluate grounding-verified answers on all three Dev20 tasks. The listing
    task uses saved structured Qwen categories, while existence and relation
    tasks parse their queried entities directly from the question. Start with
    two images (six questions), then resume the same run over Dev20:

    ```bash
    CUDA_VISIBLE_DEVICES=3 python scripts/batch_eval_grounded_answering.py \
      --structured-predictions outputs/eval_v0/dev__qwen3-vl__coco80-json-v1/predictions.jsonl \
      --image-ids data/eval_v0/splits/dev_image_ids.json \
      --box-threshold 0.30 \
      --text-threshold 0.30 \
      --evidence-score-threshold 0.30 \
      --run-name dev__evidence-answering__coco80-json-v1__box-0.30__text-0.30 \
      --max-images 2 \
      --local-files-only

    CUDA_VISIBLE_DEVICES=3 python scripts/batch_eval_grounded_answering.py \
      --structured-predictions outputs/eval_v0/dev__qwen3-vl__coco80-json-v1/predictions.jsonl \
      --image-ids data/eval_v0/splits/dev_image_ids.json \
      --box-threshold 0.30 \
      --text-threshold 0.30 \
      --evidence-score-threshold 0.30 \
      --run-name dev__evidence-answering__coco80-json-v1__box-0.30__text-0.30 \
      --local-files-only
    ```

    Each record keeps a `forced_answer` for direct comparison with the original
    VLM benchmark and a `selective_answer` that abstains when evidence is
    missing or spatial geometry is ambiguous. `metrics.json` reports closed-set
    accuracy/F1, selective accuracy and coverage, unsupported-claim rate,
    question-conditioned evidence IoU50, stage latency, and CUDA memory. Tune
    only the evidence score/mask gates and relation margin on Dev20; freeze that
    policy before one Test80 run.
18. Calibrate the answer policy offline from the completed Dev20 run. This
    reuses saved raw detections and does not load any model:

    ```bash
    python scripts/calibrate_evidence_answering.py
    ```

    The script compares structured-only and evidence-filtered listing policies,
    Qwen/Grounding consensus for existence, and evidence/geometry gates for
    spatial relation. It enforces at least `0.80` selective coverage for the
    latter two tasks and writes the immutable selection to:

    ```text
    outputs/eval_answering_v0/
      dev__evidence-answering__coco80-json-v1__box-0.30__text-0.30/
        policy_calibration/
          candidates.json
          candidates.csv
          selected_policy.json
          selected_predictions.jsonl
          summary.json
          report.md
    ```

    The selected Dev20 policy uses score/area `0.40/0.005` for listing,
    Qwen/Grounding consensus at score `0.30` for existence, and score `0.45`
    with relation margin `0.08` for spatial relation. Do not rerun calibration
    on Test80.
19. Validate all frozen Test80 inputs without loading a model or producing a
    prediction:

    ```bash
    CUDA_VISIBLE_DEVICES=3 python scripts/batch_eval_grounded_answering.py \
      --structured-predictions outputs/eval_v0/test__qwen3-vl__coco80-json-v1/predictions.jsonl \
      --policy-file outputs/eval_answering_v0/dev__evidence-answering__coco80-json-v1__box-0.30__text-0.30/policy_calibration/selected_policy.json \
      --answer-vlm-predictions outputs/eval_v0/eval_v0__qwen3-vl-8b-instruct/predictions.jsonl \
      --image-ids data/eval_v0/splits/test_image_ids.json \
      --box-threshold 0.30 \
      --text-threshold 0.30 \
      --evidence-score-threshold 0.30 \
      --run-name test__locked-task-aware__box-0.30__text-0.30 \
      --preflight-only \
      --local-files-only
    ```

    A valid preflight reports `240 questions / 80 images`, `Pending: 240`, and
    explicitly states that no model was loaded. Remove only
    `--preflight-only` and rerun the otherwise identical command for the single
    held-out evaluation. The run is resumable after interruption. Locked Test
    evaluation rejects `--max-images`, and its run config hashes the Dev policy,
    structured prompts, original Qwen answers, dataset, and split.
20. Freeze and audit the single Test80 result without changing the policy:

    ```bash
    python scripts/finalize_locked_test_report.py
    ```

    The command verifies coverage, saved metrics, and all input hashes; replays
    the pre-defined shared-threshold policy as an ablation; and writes the final
    Markdown report, comparison CSV files, per-sample failure attribution, and
    three figures under the locked run's `final_report/` directory. The frozen
    result improves exact accuracy from `0.566667` to `0.733333`; selective
    accuracy is `0.801980` at `0.841667` coverage. The Dev-selected spatial
    threshold is conservative on Test80, so it must be reported as-is rather
    than retuned after inspecting Test results.
21. Build the cross-dataset hard-case candidate index. Download and extract the
    official Open Images validation box metadata and Visual Genome relationship
    metadata into `data/raw/`, then run:

    ```bash
    python scripts/build_cross_dataset_hard_v1.py
    ```

    The default configuration selects 200 Open Images and 200 Visual Genome
    candidates, excludes Visual Genome records whose COCO IDs occur in the
    existing 100-image benchmark, and creates source-balanced Hard-Dev/Hard-Test
    candidate splits. It writes `candidates.jsonl`, `download_manifest.jsonl`,
    split IDs, source hashes, and difficulty statistics under
    `data/cross_dataset_hard_v1/`. This stage does not download pixels or create
    final questions. Exact/perceptual image deduplication must run after pixel
    download and before the new test split is frozen.
22. Download and audit only the selected cross-dataset images:

    ```bash
    python scripts/download_and_dedup_hard_images.py --workers 8
    ```

    The command is resumable. It validates image decoding and dimensions,
    computes SHA-256 and dHash fingerprints, compares the selected images with
    each other and with the existing COCO100 pixels, and writes its decision
    files under `data/cross_dataset_hard_v1/image_audit/`. Corrupt downloads and
    exact duplicates are excluded automatically. Near duplicates and metadata
    dimension mismatches require review and are never auto-deleted.
23. After any candidate-index change, rerun the image download/audit. Then
    freeze only a complete, source-balanced 400-image set:

    ```bash
    python scripts/download_and_dedup_hard_images.py --workers 8
    python scripts/freeze_cross_dataset_hard_v1.py
    ```

    The freeze command rejects stale candidate IDs, unresolved reviews,
    modified pixels, incomplete splits, and non-balanced sources. A repeated
    run verifies the immutable snapshot instead of overwriting it.
24. Download the official Open Images V7 human-verified labels and complete
    class-name table into `data/raw/open_images/`:

    ```text
    https://storage.googleapis.com/openimages/v7/oidv7-val-annotations-human-imagelabels.csv
    https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions.csv
    ```

    Only human `verification` or `crowdsource-verification` rows with
    `Confidence=0` may support negative existence claims.
25. Generate the immutable source-aware question set:

    ```bash
    python scripts/build_cross_dataset_hard_questions.py
    ```

    Open Images contributes restricted-vocabulary listing, balanced existence,
    and derived largest-instance relation questions. Visual Genome contributes
    only explicit relations whose endpoints are uniquely referable. The default
    result is 800 questions: 600 from Open Images and 200 from Visual Genome,
    split evenly between Hard-Dev and untouched Hard-Test. In the frozen v1
    result, 193 listing questions contain verified negative distractors and 7
    use an explicitly marked positive-only restricted vocabulary.
26. Establish the Qwen3-VL baseline on Hard-Dev only. First validate all frozen
    inputs without importing PyTorch or creating a run:

    ```bash
    python scripts/batch_eval_vlm.py \
      --dataset data/cross_dataset_hard_v1/questions_v1/dev_questions.jsonl \
      --output-dir outputs/cross_dataset_hard_v1/vlm \
      --run-name hard-dev400__qwen3-vl-8b-instruct \
      --required-split dev \
      --preflight-only \
      --local-files-only
    ```

    Then run an eight-question smoke test and resume the same run over all 400
    Dev questions:

    ```bash
    CUDA_VISIBLE_DEVICES=3 python scripts/batch_eval_vlm.py \
      --dataset data/cross_dataset_hard_v1/questions_v1/dev_questions.jsonl \
      --output-dir outputs/cross_dataset_hard_v1/vlm \
      --run-name hard-dev400__qwen3-vl-8b-instruct \
      --required-split dev \
      --max-new-tokens 64 \
      --max-samples 8 \
      --local-files-only

    CUDA_VISIBLE_DEVICES=3 python scripts/batch_eval_vlm.py \
      --dataset data/cross_dataset_hard_v1/questions_v1/dev_questions.jsonl \
      --output-dir outputs/cross_dataset_hard_v1/vlm \
      --run-name hard-dev400__qwen3-vl-8b-instruct \
      --required-split dev \
      --max-new-tokens 64 \
      --local-files-only
    ```

    The evaluator verifies the question-manifest hash and all image paths before
    model loading. Restricted Open Images listings are scored against each
    question's frozen vocabulary, while COCO evaluation retains its COCO80
    parser. Metrics include task and source-by-task breakdowns. Do not run the
    `test_questions.jsonl` file during model or policy development.
27. Recompute balanced relation metrics and generate the Dev-only failure
    report without loading a model:

    ```bash
    python scripts/analyze_hard_vlm_failures.py
    ```

    The report verifies all 400 IDs and replays every saved score. For the
    baseline run, Open Images relation balanced accuracy is `0.419666` with 26
    refusal/absence claims; Visual Genome relation balanced accuracy is
    `0.490193`, below its `0.61` majority-class baseline. Negative existence
    accuracy is `0.64`, compared with `0.98` for positive existence. Use these
    Dev findings for prompt and policy design; keep Hard-Test untouched.
28. Evaluate the Dev-designed relation prompt v2 as a paired, single-variable
    experiment. Build and preflight artifacts locally, then run only the 200
    relation questions on the server:

    ```bash
    python scripts/build_hard_dev_relation_prompt_v2.py

    CUDA_VISIBLE_DEVICES=3 python scripts/batch_eval_vlm.py \
      --dataset data/cross_dataset_hard_v1/questions_v2_dev/dev_questions.jsonl \
      --output-dir outputs/cross_dataset_hard_v1/vlm \
      --run-name hard-dev200-relation__qwen3-vl-8b-instruct__prompt-v2 \
      --task-type spatial_relation \
      --required-split dev \
      --max-new-tokens 64 \
      --max-samples 8 \
      --local-files-only

    CUDA_VISIBLE_DEVICES=3 python scripts/batch_eval_vlm.py \
      --dataset data/cross_dataset_hard_v1/questions_v2_dev/dev_questions.jsonl \
      --output-dir outputs/cross_dataset_hard_v1/vlm \
      --run-name hard-dev200-relation__qwen3-vl-8b-instruct__prompt-v2 \
      --task-type spatial_relation \
      --required-split dev \
      --max-new-tokens 64 \
      --local-files-only
    ```

    Prompt v2 tells the model to treat both named instances as present, compare
    visual centers, and emit exactly one of four labels. Compare parse-valid
    rate, balanced accuracy, token-limit hits, and paired answer transitions
    against v1 before adopting it:

    ```bash
    python scripts/compare_hard_relation_prompts.py
    ```

    The paired Dev result improves overall relation accuracy from `0.505` to
    `0.56`, balanced accuracy from `0.503657` to `0.590457`, parse-valid rate
    from `0.815` to `1.0`, and token-limit hits from 25 to 0. The improvement is
    source-dependent: Open Images gains 16 accuracy points (`p=0.000855`), while
    Visual Genome loses 5 points (`p=0.332306`). Accept v2 only as the Open
    Images candidate; do not globally lock it or create a Test variant yet.
29. Evaluate a Visual Genome-specific semantic relation prompt v3 on Dev only:

    ```bash
    python scripts/build_hard_dev_vg_relation_prompt_v3.py

    CUDA_VISIBLE_DEVICES=3 python scripts/batch_eval_vlm.py \
      --dataset data/cross_dataset_hard_v1/questions_v3_dev_vg/dev_questions.jsonl \
      --output-dir outputs/cross_dataset_hard_v1/vlm \
      --run-name hard-dev100-vg-relation__qwen3-vl-8b-instruct__prompt-v3 \
      --task-type spatial_relation \
      --required-split dev \
      --max-new-tokens 64 \
      --max-samples 8 \
      --local-files-only

    CUDA_VISIBLE_DEVICES=3 python scripts/batch_eval_vlm.py \
      --dataset data/cross_dataset_hard_v1/questions_v3_dev_vg/dev_questions.jsonl \
      --output-dir outputs/cross_dataset_hard_v1/vlm \
      --run-name hard-dev100-vg-relation__qwen3-vl-8b-instruct__prompt-v3 \
      --task-type spatial_relation \
      --required-split dev \
      --max-new-tokens 64 \
      --local-files-only
    ```

    Visual Genome labels are explicit human relationships rather than relations
    derived from box centers. Prompt v3 therefore preserves semantic spatial
    judgment while retaining the premise and four-choice constraints that
    eliminated v2 parse failures. Its manifest freezes the acceptance gate
    before inference: parse-valid rate at least `0.98`, zero token-limit hits,
    balanced accuracy at least `0.508782`, and exact accuracy at least `0.58`.
    Do not create a Test variant until all four Dev conditions pass.
30. Recompute all paired prompt comparisons and lock the source-aware Dev
    policy:

    ```bash
    python scripts/select_hard_relation_prompt_policy.py
    ```

    The command verifies complete, error-free Dev runs and writes an immutable
    report, selected policy, and 100 Visual Genome per-sample transitions under
    `outputs/cross_dataset_hard_v1/relation_prompt_policy_dev_v1/`. The locked
    relation policy uses prompt v2 for Open Images and prompt v3 for Visual
    Genome. The report preserves both positive and null statistical findings:
    v2 significantly improves Open Images over v1 (`p=0.000855`), v3
    significantly improves Visual Genome over v2 (`p=0.035156`), while the
    smaller v3 gain over v1 is not significant (`p=0.454498`). Hard-Test remains
    ungenerated and unevaluated.
31. Build and preflight the complete locked Hard-Test dataset:

    ```bash
    python scripts/build_locked_hard_test_questions.py

    python scripts/batch_eval_vlm.py \
      --dataset data/cross_dataset_hard_v1/questions_locked_test_v1/test_questions.jsonl \
      --output-dir outputs/cross_dataset_hard_v1/vlm \
      --run-name hard-test400__qwen3-vl-8b-instruct__source-aware-policy-v1 \
      --required-split test \
      --max-new-tokens 64 \
      --preflight-only \
      --local-files-only
    ```

    The immutable Test400 artifact applies v2 to Open Images relations and v3
    to Visual Genome relations; listing and existence questions remain exactly
    unchanged. Held-out Test evaluation prohibits `--max-samples`, so do not run
    a smoke subset. Launch the single complete server run:

    ```bash
    CUDA_VISIBLE_DEVICES=3 python scripts/batch_eval_vlm.py \
      --dataset data/cross_dataset_hard_v1/questions_locked_test_v1/test_questions.jsonl \
      --output-dir outputs/cross_dataset_hard_v1/vlm \
      --run-name hard-test400__qwen3-vl-8b-instruct__source-aware-policy-v1 \
      --required-split test \
      --max-new-tokens 64 \
      --local-files-only
    ```

    If interrupted, resume with the identical command. Do not inspect partial
    metrics or change prompts, thresholds, decoding settings, or the run name.
32. Freeze the completed held-out run and generate read-only diagnostics:

    ```bash
    python scripts/finalize_locked_hard_test_report.py
    ```

    The finalizer verifies dataset, manifest, selected-policy, model-config, and
    prediction hashes; independently replays all 400 scores; and refuses
    incomplete, truncated, or changed runs. It writes the final Markdown report,
    deterministic summary, Dev-to-Test comparison, relation confusion table,
    and 400 per-sample failure records under the Test run's `final_report/`
    directory. The held-out result is existence accuracy `0.85`, listing macro
    F1 `0.873664`, and relation accuracy/balanced accuracy
    `0.59 / 0.610559`. Visual Genome relation generalization is the primary
    limitation (`0.62 -> 0.56` accuracy and `0.591774 -> 0.491330` balanced
    accuracy from Dev to Test). Treat this as a final reported limitation, not
    a signal for Test-driven retuning.
33. Launch the Gradio demo around the frozen policy and auditable evidence:

    ```bash
    python -m pip install -r requirements-demo.txt

    CUDA_VISIBLE_DEVICES=3 python scripts/launch_demo.py \
      --server-name 0.0.0.0 \
      --server-port 7860
    ```

    The `Assistant` tab lazily loads Qwen3-VL and Grounded-SAM-2 on the first
    request, returns a natural-language answer, resolves evidence targets, and
    displays grounded boxes and masks. Inference is serialized with a queue so
    the two model stacks share one 24 GB GPU predictably.

    The `Benchmark Explorer` tab is read-only. It exposes all 400 frozen
    Hard-Test samples with source, task, and outcome filters, plus ground truth,
    prediction, evidence boxes, and per-sample diagnostics. The `Evaluation`
    tab leads with the immutable live-pipeline Test240 result, including
    Dev-to-Test generalization, relation confusion, success/failure evidence,
    runtime and integrity checks. Earlier Hard-Test400, COCO Test80, and
    grounded-answering results remain as supporting benchmarks.

    To inspect the complete result interface without loading either model:

    ```bash
    python scripts/launch_demo.py \
      --results-only \
      --server-name 0.0.0.0 \
      --server-port 7860
    ```

    Open `http://<server-ip>:7860`. If the server port is not exposed, forward
    it from the local machine:

    ```bash
    ssh -L 7860:127.0.0.1:7860 <user>@<server-ip>
    ```

    Then open `http://127.0.0.1:7860`. For a shared server, add
    `--auth-user <name> --auth-password <password>`. Do not use `--share` on an
    offline server because the external Gradio tunnel requires network access.
34. Batch-evaluate the exact live Demo chain on the COCO Dev20 split:

    ```bash
    python scripts/batch_eval_live_pipeline.py --preflight-only

    CUDA_VISIBLE_DEVICES=3 python scripts/batch_eval_live_pipeline.py \
      --max-images 2

    CUDA_VISIBLE_DEVICES=3 python scripts/batch_eval_live_pipeline.py
    ```

    The first command verifies 20 Dev images, 60 task-balanced questions, all
    referenced images, and all COCO segmentation IDs without loading a model.
    The smoke run uses a separate `smoke-2` output directory. The complete run
    processes 20 listing, 20 existence, and 20 spatial questions through:

    ```text
    Qwen answer + evidence targets -> Grounding DINO boxes -> SAM 2 masks
    ```

    Results are resumable and are written under
    `outputs/eval_live_pipeline_v0/`. The metrics include answer quality,
    structured-output validity, target precision/recall, question-conditioned
    Box IoU50, Mask IoU50, negative-evidence behavior, end-to-end success,
    stage latency, throughput, and peak CUDA allocation.

    Rerun the identical full command after an interruption. Test is locked by
    default; do not use `--allow-test` until the Dev protocol and any future
    evidence-verification policy have been frozen.
35. Evaluate the pre-registered task-aware prompt policy on the same Dev60
    questions:

    ```bash
    python scripts/batch_eval_live_pipeline.py \
      --preflight-only \
      --prompt-policy task-aware-coco-v1

    CUDA_VISIBLE_DEVICES=3 python scripts/batch_eval_live_pipeline.py \
      --prompt-policy task-aware-coco-v1 \
      --max-images 2

    CUDA_VISIBLE_DEVICES=3 python scripts/batch_eval_live_pipeline.py \
      --prompt-policy task-aware-coco-v1
    ```

    The candidate only uses the task type and question text. It constrains
    listing outputs to COCO-80, existence answers to `yes/no`, and relations to
    the four benchmark labels. It does not read answers or evidence ground
    truth while constructing prompts. The policy name, manifest hash, and
    prompt-template hash are stored in `run_config.json`.

    After the complete candidate output is synchronized back, run:

    ```bash
    python scripts/compare_live_prompt_policies.py
    ```

    This verifies the immutable generic baseline hashes, independently replays
    both metrics files, computes paired transitions and exact McNemar tests,
    and evaluates every gate frozen in
    `configs/live_prompt_policy_v1.yaml`. Keep Test locked until the resulting
    Dev decision has been recorded.

    To make old server-generated gallery paths portable without modifying the
    frozen baseline:

    ```bash
    python scripts/export_portable_live_predictions.py
    ```

    The command writes `predictions_portable.jsonl` next to each original
    prediction file and verifies every referenced artifact locally.
36. Repair the two truncated v1 listing responses with the pre-registered
    `task-aware-coco-v2` policy:

    ```bash
    python scripts/batch_eval_live_pipeline.py \
      --preflight-only \
      --prompt-policy task-aware-coco-v2

    CUDA_VISIBLE_DEVICES=3 python scripts/batch_eval_live_pipeline.py \
      --prompt-policy task-aware-coco-v2 \
      --max-images 2

    CUDA_VISIBLE_DEVICES=3 python scripts/batch_eval_live_pipeline.py \
      --prompt-policy task-aware-coco-v2
    ```

    Only the listing system prompt changes. It limits output to eight
    high-confidence visible COCO categories and explicitly prohibits copying
    the vocabulary. Dev listing ground truth contains at most five categories,
    so this cap does not truncate a target answer. Existence and relation
    prompts remain byte-identical to v1.

    Compare the complete v2 run against frozen v1:

    ```bash
    python scripts/compare_live_prompt_policies.py \
      --manifest configs/live_prompt_policy_v2.yaml
    ```

    The v2 manifest locks the v1 hashes, prompt-template hash, and
    non-regression gates before inference. The report is written under
    `outputs/eval_live_pipeline_v0/prompt_policy_v1_vs_v2_dev/`.
37. Freeze accepted v2 and prepare the one-shot held-out Test protocol:

    ```bash
    python scripts/lock_live_prompt_policy.py
    ```

    The command independently replays the Dev metrics and paired comparison,
    verifies every gate and source hash, and creates immutable
    `selected_policy.json`, `test_protocol.json`, and `report.md` under
    `outputs/eval_live_pipeline_v0/locked_policy_v1/`. Repeating the command
    verifies byte-identical artifacts.

    Synchronize the lock directory to the server and run the no-model Test
    preflight:

    ```bash
    python scripts/batch_eval_live_pipeline.py \
      --split-image-ids data/eval_v0/splits/test_image_ids.json \
      --prompt-policy task-aware-coco-v2 \
      --allow-test \
      --preflight-only
    ```

    Expected coverage is 80 images and 240 questions, split evenly across the
    three tasks. The protocol prohibits partial Test runs and freezes the run
    name plus policy, template, dataset, Test IDs, sample IDs, and COCO hashes.

    Run the single complete held-out evaluation:

    ```bash
    CUDA_VISIBLE_DEVICES=3 python scripts/batch_eval_live_pipeline.py \
      --split-image-ids data/eval_v0/splits/test_image_ids.json \
      --prompt-policy task-aware-coco-v2 \
      --allow-test
    ```

    Resume only with the identical command after an interruption. Do not read
    partial metrics or change any prompt, threshold, model, decoding option,
    evaluator, or run name.
38. Finalize the completed held-out Test240 result:

    ```bash
    python scripts/finalize_locked_live_test_report.py
    ```

    The finalizer verifies every frozen input and runtime hash, checks 240
    unique prediction IDs and all referenced artifacts, independently replays
    the complete metrics file, and writes an immutable report under the locked
    Test run's `final_report/` directory. Rerunning the command must return
    `verified`.

    Final artifacts include the Markdown and JSON summaries, input/artifact
    hash manifest, Dev-to-Test CSV, relation confusion CSV, and compact
    per-sample failure analysis in JSONL and CSV. The final held-out result is
    listing macro F1 `0.749713`, existence accuracy `0.95`, relation
    accuracy/balanced accuracy `0.7375 / 0.719907`, target F1 `0.840580`,
    Box/Mask F1 `0.502114 / 0.502114`, and complete end-to-end success
    `0.591667`. One listing reached the 192-token limit and is retained as a
    reported limitation rather than a tuning signal.

The `eval_v0` builder downloads official COCO val2017 annotations and only the
selected images. It creates 300 questions across object listing, balanced object
existence, and spatial relation tasks. COCO boxes are retained as evidence for
later grounding evaluation.

39. Present the locked result and prepare the career-facing project package:

    - The Gradio `Evaluation` tab reads the final live-pipeline `summary.json`,
      `generalization.csv`, `relation_confusion.csv`, evidence artifacts, and
      integrity manifest directly from the locked Test240 run.
    - Resume-ready Chinese and English descriptions are maintained in
      `docs/resume_project_description.md`.
    - The online inference and offline evaluation diagrams are maintained in
      `docs/system_architecture.md`.
    - The 30-second pitch, two-minute explanation, metric notes, limitations,
      and common technical follow-ups are maintained in
      `docs/interview_notes.md`.
40. Prepare the official COCO POPE benchmark with selective image downloads:

    ```bash
    # Inspect the official question files first.
    python scripts/prepare_pope_data.py \
      --metadata-only \
      --backend urllib

    # Use HTTP only when a proxy breaks the COCO HTTPS certificate.
    python scripts/prepare_pope_data.py \
      --backend urllib \
      --image-transport http \
      --max-images 3

    # Resume and complete all 500 referenced images.
    python scripts/prepare_pope_data.py \
      --backend urllib \
      --image-transport http

    # Verify the complete dataset without network access.
    python scripts/prepare_pope_data.py \
      --audit-only \
      --backend urllib \
      --image-transport http
    ```

    The script retains the three official question files, creates a normalized
    `questions.jsonl`, downloads only the 500 referenced COCO val2014 images,
    validates image decoding, and writes SHA-256 manifests. A complete result
    contains 9000 questions: 3000 each for random, popular, and adversarial
    negatives, with 1500 yes and 1500 no labels per strategy.

    Copy the complete `data/pope/` directory to the same project path on an
    offline server. Run the final `--audit-only` command there before starting
    model evaluation.
41. Run the official-compatible POPE Qwen3-VL baseline:

    ```bash
    # Validate a balanced smoke selection without loading the model.
    python scripts/batch_eval_pope.py \
      --samples-per-strategy 30 \
      --preflight-only

    # Run 30 random, 30 popular, and 30 adversarial questions.
    CUDA_VISIBLE_DEVICES=3 python scripts/batch_eval_pope.py \
      --samples-per-strategy 30 \
      --run-name pope-smoke90__qwen3-vl-8b-instruct \
      --local-files-only

    # Validate the complete official selection before the long run.
    python scripts/batch_eval_pope.py \
      --require-complete \
      --preflight-only

    # Run all 9000 questions. Repeat the identical command to resume.
    CUDA_VISIBLE_DEVICES=3 python scripts/batch_eval_pope.py \
      --require-complete \
      --run-name pope-full9000__qwen3-vl-8b-instruct \
      --local-files-only
    ```

    The runner fixes deterministic decoding and a versioned system prompt that
    requests exactly one `Yes` or `No` token. It reports the official POPE
    Accuracy, Precision, Recall, F1, and Yes Ratio overall and separately for
    random, popular, and adversarial negatives. It also records strict
    Yes/No parse validity, token-limit hits, latency, memory, selected-ID hash,
    and runtime versions. Smoke and complete runs must use separate output
    directories.
42. Attribute the completed POPE errors and render qualitative cases:

    ```bash
    python scripts/analyze_pope_errors.py
    ```

    The analyzer independently reproduces every saved Yes/No evaluation and
    verifies the completed metrics and run config before writing derived
    artifacts under the full run's `error_analysis/` directory. Outputs
    include `summary.json`, `errors.jsonl`, per-object and per-image CSV files,
    representative-case JSONL, two JPEG contact sheets, and `report.md`.

    POPE repeats each positive query across all three negative-sampling
    strategies. The report therefore retains the official raw confusion
    counts while also deduplicating semantic queries for qualitative review.
    False positives are treated as benchmark disagreements rather than proven
    hallucinations because COCO-derived negative labels can reflect annotation
    omissions or category-boundary ambiguity.
43. Run the Grounding-aware binary answer verifier V1:

    ```bash
    # Verify one known POPE false negative without loading a VLM again.
    CUDA_VISIBLE_DEVICES=3 python scripts/run_grounding_answer_verifier.py \
      --sample-id pope_coco_random_1139 \
      --local-files-only

    # Validate paths, thresholds, and the selected baseline record first.
    python scripts/run_grounding_answer_verifier.py \
      --sample-id pope_coco_random_1139 \
      --preflight-only \
      --local-files-only
    ```

    The versioned `grounding_positive_rescue_v1` policy reuses the saved Qwen
    answer, queries Grounding DINO with the requested object, segments accepted
    boxes with SAM 2.1, and changes `No` to `Yes` only when localized evidence
    reaches the frozen `0.45` promotion threshold. A missing detection never
    changes `Yes` to `No`, because detector silence is not proof of absence.

    Each run writes Grounding boxes, SAM masks, the original Grounded-SAM-2
    result, and `verification.json` under
    `outputs/grounding_answer_verifier_v1/<sample-id>/`. This stage validates
    the correction mechanism only; reportable gains require the separate
    batch comparison and ablation protocol in the next stage.
44. Validate and run the paired POPE verifier `Smoke90` comparison:

    ```bash
    # Validate the balanced selection and frozen inputs without loading models.
    python scripts/batch_eval_pope_verifier.py \
      --samples-per-strategy 30 \
      --preflight-only \
      --local-files-only

    # Run 90 questions, deduplicated to 53 Grounded-SAM-2 queries.
    CUDA_VISIBLE_DEVICES=3 python scripts/batch_eval_pope_verifier.py \
      --samples-per-strategy 30 \
      --run-name pope-verifier-smoke90__positive-rescue-v1 \
      --local-files-only
    ```

    The batch runner reuses the frozen Qwen predictions and performs paired
    baseline-versus-verified evaluation on exactly the same POPE questions.
    Repeated semantic queries across POPE strategies share one cached grounding
    result. The run records beneficial and harmful corrections, per-strategy
    metric deltas, correction status counts, cached and uncached latency
    projections, peak memory, source hashes, and the selected query hash.

    `Smoke90` is an engineering acceptance run only. Do not select thresholds
    from its accuracy. Threshold ablations must be pre-registered and selected
    on a separate development split before one locked full-9000 evaluation.
45. Run the V2 semantic crop verifier on the completed `Smoke90` evidence:

    ```bash
    # Confirm all frozen inputs and report the exact crop-review workload.
    python scripts/batch_eval_pope_verifier_v2.py \
      --samples-per-strategy 30 \
      --preflight-only \
      --local-files-only

    # Reuse V1 Grounded-SAM-2 evidence and load only Qwen for crop review.
    CUDA_VISIBLE_DEVICES=3 python scripts/batch_eval_pope_verifier_v2.py \
      --samples-per-strategy 30 \
      --run-name pope-verifier-v2-smoke90__semantic-rescue \
      --local-files-only
    ```

    V2 does not trust detector confidence as final semantic evidence. Negative
    baseline answers first pass the frozen evidence gate; near-full-frame masks
    are rejected, at most two context crops are retained, and deterministic
    Qwen review must answer exactly `Yes` before a `No` answer is promoted.
    Positive baseline answers are never demoted by detector silence.

    The current Smoke90 preflight contains 40 unique negative baseline queries,
    but only 17 queries and 20 candidate crops require semantic review. Outputs
    are written under
    `outputs/eval_pope_verifier_v2/pope-verifier-v2-smoke90__semantic-rescue/`.
    The run records crop hashes, raw semantic answers, paired V2 metrics,
    correction outcomes, exact McNemar significance, latency, memory, and all
    immutable source/config hashes.

    This run validates the V2 mechanism on the engineering smoke selection.
    Final thresholds and module choices must still be selected on the separate
    development protocol before a locked held-out evaluation.
46. Build and run the POPE-isolated Verifier Dev110 baseline:

    ```bash
    # Deterministically build 55 positive/negative question pairs.
    python scripts/build_verifier_dev_v1.py

    # Recheck source, artifact, balance, GT, and image-isolation hashes.
    python scripts/build_verifier_dev_v1.py --audit-only

    # Validate the complete model run without loading Qwen.
    python scripts/batch_eval_verifier_dev.py \
      --preflight-only \
      --local-files-only

    # Freeze the complete Dev110 Qwen baseline.
    CUDA_VISIBLE_DEVICES=3 python scripts/batch_eval_verifier_dev.py \
      --run-name qwen-baseline-dev110__qwen3-vl-8b-instruct \
      --local-files-only
    ```

    Dev110 starts from the existing 20-image COCO val2017 development split,
    excludes every image referenced by POPE Full500, and therefore retains 19
    images with zero POPE overlap. Every annotated image/category pair becomes
    a positive query. Each is paired with an absent category from the same
    official COCO supercategory when possible; singleton/fallback cases use a
    deterministic globally balanced absent category.

    The frozen protocol contains 110 questions, 55 balanced pairs, 64 queried
    categories, 42 same-supercategory hard negatives, and 13 balanced fallback
    negatives. `data/verifier_dev_v1/manifest.json` locks all source hashes,
    selection rules, record order, overlap exclusions, and artifact hashes.
    Rebuilding the dataset is byte deterministic.

    The Qwen runner uses the same exact `Yes`/`No` prompt and deterministic
    decoding as POPE, but reports this result under the distinct
    `verifier_dev_qwen_baseline_v1` protocol. It records overall, pair-role,
    supercategory, paired-question, parsing, latency, and memory metrics. This
    baseline must complete before Dev grounding, because the asymmetric method
    only needs to process frozen negative Qwen answers.
47. Cache Grounded-SAM-2 evidence for the frozen Dev110 negative answers:

    ```bash
    # Verify the baseline hashes and exact GT-free query selection.
    python scripts/batch_ground_verifier_dev.py \
      --preflight-only \
      --local-files-only

    # Run 57 strict-No queries and retain 12 qualitative artifacts.
    CUDA_VISIBLE_DEVICES=3 python scripts/batch_ground_verifier_dev.py \
      --run-name grounding-dev57__grounding-dino-base__sam2.1-base-plus \
      --visualize-limit 12 \
      --local-files-only
    ```

    Selection uses only the frozen Qwen answer: every strict `No` is processed
    and every `Yes` is skipped. GT labels and pair roles are absent from the
    inference jobs. The frozen baseline produces 57 queries over all 19 Dev
    images and 51 targets, with ordered query-key hash
    `117e46c4596700e3db55129305a8a1adaa1a82775e105fd9e9e18d36bb32f265`.

    The detector runs once at the lowest pre-registered candidate threshold
    `box=0.30, text=0.30`; higher-score ablations filter this evidence offline.
    `evidence.jsonl` contains no GT fields. `metrics.json` joins labels only
    after inference to compare candidate presence for the three false
    negatives and 54 true negatives, along with score bins, latency, and
    memory.
48. Cache deterministic Qwen reviews for the complete Dev candidate union:

    ```bash
    # Verify baseline/evidence hashes and the GT-free crop selection.
    python scripts/batch_review_verifier_dev.py \
      --preflight-only \
      --local-files-only

    # Generate 23 context crops and review each with exact Yes/No decoding.
    CUDA_VISIBLE_DEVICES=3 python scripts/batch_review_verifier_dev.py \
      --run-name semantic-review-dev23__qwen3-vl-8b-instruct \
      --local-files-only
    ```

    The candidate union contains 23 crops from 19 queries, 12 images, and 18
    target categories. It retains every cached candidate with grounding score
    at least `0.30` and mask score at least `0.50`, before applying the `0.90`
    area ablation, and keeps at most two candidates per query. Its ordered key
    hash is
    `186c3a3e15bc9c901cdfe528518e2a422ffe5e334709ef53d68cc83001ab4f51`.

    Jobs and crops contain no GT. Every crop is hashed, raw Qwen answers are
    retained, and only exact deterministic `Yes`/`No` outputs are accepted by
    the later V2 policy. Once this run completes, V1 score-only, geometry,
    candidate-count, and full semantic-gate ablations reuse the same baseline,
    evidence, and reviews without additional model inference.
49. Run and lock the offline Verifier Dev110 ablation:

    ```bash
    # Verify the complete 110/57/23 source chain and policy-grid hash.
    python scripts/compare_verifier_dev_ablations.py --audit-only

    # Evaluate all 21 frozen policies without loading a model.
    python scripts/compare_verifier_dev_ablations.py
    ```

    `configs/verifier_dev_ablation_v1.yaml` expands four score thresholds
    (`0.30`, `0.40`, `0.45`, and `0.50`) across five controlled verifier
    families: V1 score-only, V1 plus geometry, V2 without geometry using
    Top-2, and V2 with the `0.90` area gate using Top-1 or Top-2. The baseline
    is the twenty-first policy. Policy order is frozen by hash
    `c5ed07118702ef2b9bb402b9168c9f93af8663de8c0284f893ae520c3718bf5d`.

    A verifier may be locked only if it strictly improves Dev accuracy,
    preserves or improves F1, and has positive net corrections. No candidate
    passed all three gates. Baseline accuracy remains `0.963636`; the
    score-`0.30` V2 variants rescue one false negative but introduce two false
    positives (`0.954545` accuracy), while score `0.50` makes no changes and
    only ties the baseline. The frozen decision is therefore
    `retain_baseline_no_eligible_verifier`.

    Outputs under
    `outputs/eval_verifier_dev_v1/offline-ablation-dev110__v1-v2/` include
    JSONL/CSV policy tables, every changed sample, selected predictions,
    immutable source hashes, `selected_policy.json`, and a Markdown report.
    Held-out POPE data is not accessed during selection. Because the verifier
    was rejected on Dev, do not run it on full POPE solely to search for a
    favorable test result; redesign must remain Dev-only.
50. Prepare and run the Dev-only contrastive Verifier V3:

    ```bash
    # Verify the complete source chain and three GT-free V3 jobs.
    python scripts/batch_review_verifier_dev_v3.py \
      --preflight-only \
      --local-files-only

    # Optional: write the red-box crops without loading Qwen.
    python scripts/batch_review_verifier_dev_v3.py \
      --prepare-only \
      --local-files-only

    # Run only three contrastive category reviews on the server GPU.
    CUDA_VISIBLE_DEVICES=3 python scripts/batch_review_verifier_dev_v3.py \
      --local-files-only
    ```

    V3 keeps the Stage 38 low-cost path (`score>=0.30`, mask score `>=0.50`,
    area ratio `<=0.90`, Top-1). Only the three candidates that received an
    exact V2 `Yes` continue to V3. Their crops contain a red rectangle around
    the detector candidate, and Qwen must choose exactly one label from the
    target's complete COCO supercategory plus `none`. For example, the
    `truck` candidate is compared against every official vehicle category,
    including `bus`, instead of being asked another target-biased Yes/No
    question.

    `configs/coco_80_supercategories_v1.yaml` is a standalone public ontology
    with no image IDs or annotations. V3 jobs contain no GT, pair role, or
    expected output. The ordered Dev3 job hash is
    `c50a00cf39fdca04ac3a46053817569bcb983dabb4cec14919da8bb6ce485172`.
    When all three reviews complete, the same script writes full paired
    Dev110 predictions, corrections, runtime projections, and
    `v3_decision.json`. V3 may advance to held-out evaluation only if it
    strictly improves accuracy, does not reduce F1, and has positive net
    corrections.
51. Freeze the final verifier decision and generate the failure audit:

    ```bash
    # Validate every source hash and recompute the frozen metrics.
    python scripts/finalize_verifier_dev_report.py --audit-only

    # Generate the final policy, comparison tables, and case analysis.
    python scripts/finalize_verifier_dev_report.py
    ```

    The completed V3 reviews return `chair`, `none`, and `car` for the three
    `chair`, `book`, and `truck` candidates. This blocks the V2 truck
    confusion, but it also removes the only beneficial book rescue and still
    promotes the absent chair. V3 therefore records zero beneficial and one
    harmful correction, reducing accuracy from `0.963636` to `0.954545` and
    F1 from `0.962963` to `0.954128`. Its locked decision is
    `reject_v3_on_dev`.

    Stage 40 reads the existing Dev artifacts only; it loads no model and
    accesses no held-out record. It rechecks the 110 baseline predictions, 57
    grounding queries, 23 V2 reviews, 21 Stage 38 policies, three V3 reviews,
    all source hashes, and both rejection decisions. The final policy is
    `retain_qwen_baseline_disable_answer_rewrite`: answer rewriting is
    disabled, while Grounding DINO and SAM 2.1 remain available for visual
    evidence, localization, and failure auditing.

    Outputs are written under
    `outputs/eval_verifier_final_v1/verifier-dev110-final/`. They include the
    immutable `final_policy.json`, a five-row V1/V2/V3 comparison, six traced
    failure cases, CSV/JSONL exports, an interview-facing Markdown report,
    and `artifact_manifest.json` with hashes for every generated artifact.
    Since every verifier failed the pre-registered Dev gates, do not run one
    on held-out POPE to search for a favorable result and do not claim that
    the verifier reduced hallucination.
52. Launch the final interview dashboard and review the presentation package:

    ```bash
    # Frozen artifacts only; no model or GPU is required.
    python scripts/launch_demo.py \
      --results-only \
      --server-name 0.0.0.0 \
      --server-port 7860
    ```

    The `Evaluation` page now contains two focused views. `Held-Out Test240`
    preserves the final answer/evidence metrics, generalization table,
    relation confusion, qualitative evidence, and downloadable report.
    `Verifier Audit` loads the Stage 40 policy, five-row V1/V2/V3 comparison,
    six-case failure audit, and immutable artifacts. It explicitly shows that
    answer rewriting is disabled and that grounded evidence is used only for
    localization and audit.

    The final interview package is maintained in:

    - `docs/resume_project_description.md` for Chinese/English resume bullets;
    - `docs/system_architecture.md` for online, evaluation, and rejected
      verifier architecture diagrams;
    - `docs/interview_notes.md` for the 30-second, two-minute, STAR, metric,
      and verifier-decision explanations;
    - `docs/interview_demo_reproduction.md` for local/server launch,
      SSH-forwarding, audit, test, and artifact commands.

    This completes the required interview-project path. Additional public
    benchmark comparisons or a learned verifier are optional research
    extensions and must use a new development protocol.
