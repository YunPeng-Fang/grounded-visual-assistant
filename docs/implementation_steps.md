# Implementation Steps

## Stage 1: Single-Image VLM Baseline

### Step 1: Prepare Environment

```bash
conda create -n grounded-vlm python=3.10 -y
conda activate grounded-vlm
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements-torch-cu124.txt
python -m pip install -r requirements.txt
```

This server uses NVIDIA driver 550.135. Do not install an arbitrary newer
PyTorch CUDA wheel. Run the checks in `docs/environment_setup.md` before loading
Qwen3-VL.

### Step 2: Prepare Demo Images

Put 20 images into:

```text
data/demo_images/
```

Recommended image types:

- street scenes
- indoor scenes
- products
- people
- multiple objects
- small objects
- occlusions
- low-light images

### Step 3: Run One Image

```bash
CUDA_VISIBLE_DEVICES=3 python scripts/run_vlm_baseline.py \
  --image data/demo_images/example.jpg \
  --question "What objects are visible in this image?"
```

### Step 4: Ask Three Question Types

For every image, ask:

1. Image description: What is in the image?
2. Object existence: Is there a specific object?
3. Relation reasoning: Where is A relative to B?

### Step 5: Record Failures

For each failure, label it as:

- hallucinated object
- missed small object
- wrong spatial relation
- OCR error
- ambiguous visual evidence
- insufficient evidence

## Stage 2 Preview: Grounding

Before adding the grounding model, build a reproducible baseline evaluation set:

```bash
python scripts/build_eval_v0.py --num-images 100 --seed 2026
```

Generated files:

```text
data/eval_v0/
  images/
  manifest.json
  questions.jsonl
```

The dataset contains 100 object-listing questions, 100 balanced yes/no object
existence questions, and 100 coarse spatial-relation questions. Every record
also includes COCO evidence boxes that can later supervise or evaluate the
grounding stage.

Run a short batch smoke test before committing to all 300 questions:

```bash
CUDA_VISIBLE_DEVICES=3 python scripts/batch_eval_vlm.py \
  --max-samples 5 \
  --max-new-tokens 64 \
  --local-files-only
```

Then resume into the full evaluation. The same output directory and model
configuration are reused, so the five completed samples are skipped:

```bash
CUDA_VISIBLE_DEVICES=3 python scripts/batch_eval_vlm.py \
  --max-new-tokens 64 \
  --local-files-only
```

`metrics.json` reports exact accuracy for object existence and spatial
relations, macro precision/recall/F1 for object listing, parse-valid rates,
latency percentiles, throughput, errors, and completion coverage.
Generation is deterministic (`do_sample: false`), and every successful record
also stores generation latency, end-to-end latency, and CUDA memory statistics.

Install and connect the official Grounded-SAM-2 source after Stage 1 is stable:

```bash
conda activate grounded-vlm
bash scripts/install_grounded_sam2.sh
python scripts/check_grounded_sam2_env.py
```

Run the first official Grounding DINO to SAM 2.1 image pipeline:

```bash
CUDA_VISIBLE_DEVICES=3 python scripts/run_grounded_sam2.py \
  --image data/eval_v0/images/000000230993.jpg \
  --prompt "person. umbrella. backpack." \
  --local-files-only
```

This produces boxes, masks, individual mask PNGs, COCO RLE, confidence scores,
latency, peak CUDA memory, and overlay visualizations. The full offline setup is
documented in `docs/grounded_sam2_deployment.md`.

## Stage 3: Oracle Grounding Batch Evaluation

Grounded-SAM-2 already performs box-prompted SAM 2.1 segmentation. Establish a
reproducible oracle-prompt baseline before changing prompts or model internals.
The evaluator selects the one `object_listing` record per image, so the 300
question records become 100 unique image-level evaluations.

Run five images first:

```bash
CUDA_VISIBLE_DEVICES=3 python scripts/batch_eval_grounded_sam2.py \
  --max-images 5 \
  --local-files-only
```

Then rerun without the limit. The same output directory is reused and the five
completed image IDs are skipped:

```bash
CUDA_VISIBLE_DEVICES=3 python scripts/batch_eval_grounded_sam2.py \
  --local-files-only
```

Outputs are written under
`outputs/eval_grounding_v0/<dataset__detector__sam2__oracle>/`:

```text
predictions.jsonl
errors.jsonl
metrics.json
run_config.json
visualizations/
```

The baseline protocol and next experiments are:

1. Use `eval_v0` target categories as oracle text prompts.
2. Match detections to same-class COCO boxes at IoU 0.5 and report mAP50.
3. Recover COCO masks for mask IoU evaluation.
4. Replace oracle prompts with Qwen-generated target phrases.
5. Compare official baseline and proposed improvements.

## Stage 4: Standard COCO Box and Mask Metrics

The compact VLM benchmark retains only objects above an area-ratio threshold.
That is useful for question generation but can incorrectly count valid small
detections as false positives. Restore every original COCO instance belonging
to each image's oracle-prompted categories:

```bash
python scripts/build_coco_grounding_gt.py
```

For the current `eval_v0`, this converts 486 filtered evidence boxes into 712
full instances across the same 100 images. All restored annotations include
COCO segmentation ground truth. The generated file is:

```text
data/eval_v0/coco_grounding_gt.json
```

Evaluate the already completed predictions without loading either model:

```bash
python scripts/eval_grounded_sam2_coco.py --require-complete
```

If multiple grounding runs exist, select one explicitly:

```bash
python scripts/eval_grounded_sam2_coco.py \
  --predictions outputs/eval_grounding_v0/<run-name>/predictions.jsonl \
  --require-complete
```

Outputs are saved beside the selected prediction file:

```text
coco_eval/
  coco_bbox_results.json
  coco_segm_results.json
  coco_metrics.json
```

`coco_metrics.json` contains standard bbox and segmentation AP@[0.50:0.95],
AP50, AP75, AP by object size, AR, per-category AP, coverage, and unmapped-label
diagnostics. The protocol is oracle-conditioned by image and is not equivalent
to evaluating an unprompted 80-class COCO detector.

## Stage 5: Fixed Split and Post-Processing Calibration

Create the split once and keep the generated files unchanged for every later
ablation:

```bash
python scripts/build_eval_splits.py --dev-size 20 --seed 2026
```

The files are:

```text
data/eval_v0/splits/dev_image_ids.json
data/eval_v0/splits/test_image_ids.json
```

The development split is the only split used for NMS and threshold selection.
The test split must remain untouched until the final configuration is locked.

First evaluate the existing no-NMS prediction file on dev without repeating
model inference:

```bash
python scripts/eval_grounded_sam2_coco.py \
  --predictions outputs/eval_grounding_v0/<baseline-run>/predictions.jsonl \
  --image-ids data/eval_v0/splits/dev_image_ids.json \
  --output-dir outputs/eval_grounding_v0/<baseline-run>/coco_eval/dev \
  --require-complete
```

The class-aware NMS 0.60 dev experiment suppressed 0/118 candidates and changed
neither bbox nor mask AP. The maximum same-class box IoU was 0.2945, so NMS
0.50 and 0.70 cannot affect this split. Keep NMS disabled for threshold tuning.

Run one low box-threshold inference with the original text threshold:

```bash
CUDA_VISIBLE_DEVICES=3 python scripts/batch_eval_grounded_sam2.py \
  --image-ids data/eval_v0/splits/dev_image_ids.json \
  --box-threshold 0.25 \
  --text-threshold 0.30 \
  --run-name dev__box-0.25__text-0.30__nms-none \
  --local-files-only
```

The low-threshold run retains every candidate needed for higher box cutoffs.
Evaluate all cutoffs offline without loading either model again:

```bash
python scripts/eval_grounding_score_sweep.py \
  --predictions outputs/eval_grounding_v0/dev__box-0.25__text-0.30__nms-none/predictions.jsonl
```

Outputs are written under:

```text
outputs/eval_grounding_v0/dev__box-0.25__text-0.30__nms-none/
  coco_eval/box_score_sweep/
    box-0.25/coco_metrics.json
    box-0.30/coco_metrics.json
    box-0.35/coco_metrics.json
    box-0.40/coco_metrics.json
    box-0.45/coco_metrics.json
    summary.json
    summary.csv
```

Use mask AP as the primary selection metric, followed by bbox AP, small mask AP,
and fewer retained detections as the tie-breaker. The offline 0.40 result must
match the existing 0.40 dev baseline closely;
otherwise investigate candidate generation before selecting a threshold. After
locking box threshold, repeat one low-box run for text thresholds 0.20 and 0.40.
Do not run the test split until both thresholds are fixed.

Run the two remaining text thresholds and reuse the offline box sweep for each:

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
```

Combine text thresholds 0.20/0.30/0.40 and all box cutoffs:

```bash
python scripts/compare_grounding_score_sweeps.py
```

The combined files are:

```text
outputs/eval_grounding_v0/dev__box-text-threshold-sweep/
  summary.json
  summary.csv
```

The selector rejects configurations with unmapped labels when a zero-unmapped
alternative exists, then compares mask AP, bbox AP, small-mask AP, and retained
detection count. Treat this as a dev-only choice; the test split remains sealed.

## Stage 6: VLM-Prompt End-to-End Grounding

The oracle experiment measures the detector and segmenter upper bound because it
uses ground-truth image categories as prompts. The first end-to-end experiment
reuses Qwen3-VL's saved `object_listing` answers, parses canonical COCO classes,
and feeds only those predicted classes to Grounding DINO. Oracle categories are
retained solely as evaluation targets.

Run Dev20 first with the already selected detector thresholds:

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

The existing Qwen predictions produce this CPU-only prompt preflight on Dev20:

```text
images=20, target_categories=58, predicted_categories=51
TP=44, FP=7, FN=14, micro P=0.862745, R=0.758621, F1=0.807339
empty_prompt_images=1
```

An empty parsed prompt is not dropped. It is saved as a valid zero-detection
record so every missed target becomes a false negative. Each prediction stores:

```text
prompt_categories
prompt_evaluation
vlm_prediction
pipeline_latency_seconds
```

`metrics.json` adds `prompt_quality` and `end_to_end_latency_seconds` to the
existing Grounded-SAM-2 metrics. The COCO evaluator carries both sections into
`coco_metrics.json` and labels the protocol as
`vlm_prompted_against_oracle_target_instances`. Run standard COCO evaluation
after inference:

```bash
python scripts/eval_grounded_sam2_coco.py \
  --predictions outputs/eval_grounding_v0/dev__vlm-prompt__box-0.30__text-0.30__nms-none/predictions.jsonl \
  --image-ids data/eval_v0/splits/dev_image_ids.json \
  --output-dir outputs/eval_grounding_v0/dev__vlm-prompt__box-0.30__text-0.30__nms-none/coco_eval/dev \
  --require-complete
```

Inspect Dev20 failures without changing the locked detector thresholds. Parser
changes are allowed only on dev. Once the parser is frozen, run Test80 exactly
once by replacing `dev_image_ids.json` with `test_image_ids.json` and using a new
`test__vlm-prompt__box-0.30__text-0.30__nms-none` run name. Compare that COCO
result with the oracle Test80 upper bound.

Generate a stage-attributed failure report before changing the prompt parser:

```bash
python scripts/analyze_vlm_grounding_failures.py \
  --predictions outputs/eval_grounding_v0/dev__vlm-prompt__box-0.30__text-0.30__nms-none/predictions.jsonl
```

The report is written beside the prediction file:

```text
failure_analysis/
  summary.json
  per_image.jsonl
  per_image.csv
  report.md
```

It attributes box false negatives to either an absent prompt category or a
Grounding DINO miss after the category was prompted. False positives are split
between prompted target categories and benchmark off-target categories. An
off-target category is absent from this benchmark's target set; it is not proof
that the object is visually absent. Non-terminal VLM answer endings are reported
as a generation-truncation heuristic, not as a definitive finish reason.

## Stage 7: Ontology-Constrained JSON Prompt Generation

The free-text Qwen baseline spends most of its 64-token budget on explanations.
On Dev20, 17/20 answers have non-terminal endings and the prompt stage causes
19/24 matched-box false negatives. Replace that interface with a dedicated
COCO-80 classifier prompt that requests one JSON array and no prose.

Run a five-image smoke test, then resume the same run over Dev20:

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

The output remains compatible with the existing grounding bridge and adds:

```text
structured_output.parse_source
structured_output.strict_json_array
structured_output.schema_valid
structured_output.parsed_categories
generated_tokens
hit_max_new_tokens
```

`metrics.json` reports strict JSON rate, schema-valid rate, recovery rate,
invalid and duplicate items, generated-token statistics, and the standard
object-listing category metrics. Require complete Dev20 coverage and compare
against the free-text prompt baseline before detector inference. The current
free-text target is micro P/R/F1 `0.862745/0.758621/0.807339`.

Only if structured prompt quality improves, run the fixed downstream pipeline:

```bash
CUDA_VISIBLE_DEVICES=3 python scripts/batch_eval_grounded_sam2.py \
  --prompt-source vlm \
  --vlm-predictions outputs/eval_v0/dev__qwen3-vl__coco80-json-v1/predictions.jsonl \
  --image-ids data/eval_v0/splits/dev_image_ids.json \
  --box-threshold 0.30 \
  --text-threshold 0.30 \
  --run-name dev__structured-vlm-prompt__box-0.30__text-0.30__nms-none \
  --local-files-only
```

Do not change Grounding DINO thresholds during this ablation. The controlled
variable is the VLM prompt interface. Run COCOeval and the failure-analysis
script on the new grounding output before selecting the prompt method.

## Stage 8: Grounding-Verified Answering

The previous stages evaluate category proposals, boxes, and masks. This stage
turns those components into answers while retaining an auditable evidence path:

```text
image + question
  -> task-conditioned category query
  -> Grounding DINO boxes
  -> SAM 2.1 masks
  -> explicit evidence gate
  -> forced answer + selective answer
```

The three task policies are:

- `object_listing`: use the saved structured Qwen category list and retain only
  categories with accepted Grounded-SAM-2 evidence.
- `object_existence`: parse the queried category from the question. A detection
  supports `yes`; detector silence produces forced `no` but selective refusal,
  because failure to detect is not visual proof of absence.
- `spatial_relation`: parse the ordered entity pair, select the largest predicted
  mask for each category (box area fallback), and apply the benchmark's
  normalized-center dominant-axis rule. Refuse if either entity is missing or
  the displacement is below the relation margin.

Run two Dev20 images as a six-question smoke test:

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
```

Then remove `--max-images 2` and rerun the otherwise identical command. The
same output directory resumes by question ID and should finish with 60/60
records. Inspect these metric groups:

```text
closed_set_answers
selective_answers
evidence_support
question_conditioned_evidence_iou50
latency_seconds
cuda_memory_gb
```

`closed_set_answers` scores the forced answer on every question and is directly
comparable with the original VLM baseline. `selective_answers` reports accuracy
only where the evidence policy answered, together with coverage and abstention.
The selective unsupported-claim rate should be zero by construction; any
nonzero value is an implementation or policy bug.

Use Dev20 to inspect errors and, if justified, adjust only
`--evidence-score-threshold`, `--evidence-mask-score-threshold`,
`--evidence-min-mask-area-ratio`, and `--relation-margin`. Keep detector
thresholds fixed at box/text `0.30/0.30`. Once the evidence policy is frozen,
run Test80 once with:

```bash
CUDA_VISIBLE_DEVICES=3 python scripts/batch_eval_grounded_answering.py \
  --structured-predictions outputs/eval_v0/test__qwen3-vl__coco80-json-v1/predictions.jsonl \
  --image-ids data/eval_v0/splits/test_image_ids.json \
  --box-threshold 0.30 \
  --text-threshold 0.30 \
  --evidence-score-threshold 0.30 \
  --run-name test__evidence-answering__coco80-json-v1__box-0.30__text-0.30 \
  --local-files-only
```

Do not tune the policy after viewing Test80 results.

## Stage 9: Dev-Only Task-Aware Policy Calibration

The first evidence policy improves forced answers but its generic refusal rule
removes too many correct negative existence answers. Calibrate each task from
the saved Dev20 annotations instead of forcing one threshold onto all tasks:

```bash
python scripts/calibrate_evidence_answering.py
```

This is CPU-only and performs no model inference. It evaluates:

- Listing: structured VLM output versus grounding score, mask score, and mask
  area-ratio gates. Selection maximizes macro F1, then exact accuracy.
- Existence: Qwen provides the forced yes/no answer; the selective answer is
  emitted only when Qwen and query-conditioned grounding agree. Selection
  maximizes selective accuracy subject to at least `0.80` coverage.
- Spatial relation: score, mask, area, and geometry-margin gates. Selection
  uses the same selective-accuracy and coverage constraint.

The default grid evaluates 673 candidates and locks:

```text
object_listing:
  score=0.40, mask_score=None, min_mask_area_ratio=0.005
object_existence:
  Qwen/Grounding consensus, score=0.30
spatial_relation:
  score=0.45, relation_margin=0.08
```

The selected task-aware policy changes Dev20 metrics as follows:

| Policy | Forced mean score | Forced exact | Selective coverage | Selective exact |
|---|---:|---:|---:|---:|
| Original Qwen | 0.695675 | 0.550000 | 1.000000 | 0.550000 |
| Initial evidence policy | 0.835176 | 0.700000 | 0.816667 | 0.673469 |
| Task-aware policy | 0.867460 | 0.750000 | 0.900000 | 0.796296 |

Spatial selective accuracy is `1.0` at `0.80` coverage; existence selective
accuracy is `0.944444` at `0.90` coverage; listing macro F1 is `0.852381`.

The calibration directory contains the complete candidate table, selected
per-sample answers, comparison summary, Markdown report, and
`selected_policy.json`. Treat that policy file as immutable before Test80. No
Test80 record may contribute to candidate ranking or threshold changes.

## Stage 10: Single Locked Test80 Evaluation

The inference script can now consume the immutable Dev-selected policy. The
detector remains at box/text `0.30/0.30` so every candidate required by the
task-specific post-filters is retained. Listing applies score `0.40` and minimum
mask area ratio `0.005`; existence joins the saved original Qwen yes/no answer
and emits a selective answer only on Qwen/Grounding agreement; spatial relation
applies score `0.45` and relation margin `0.08`.

First run input validation only:

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

The expected preflight is exactly `240` questions from `80` images, with no
model load and no prediction. Then remove only `--preflight-only` and run the
same command once. Do not add `--max-images`; the script rejects partial locked
Test runs. An interrupted full run may be resumed with the identical command
because completed question IDs are preserved.

Results are written to:

```text
outputs/eval_answering_v0/
  test__locked-task-aware__box-0.30__text-0.30/
    predictions.jsonl
    errors.jsonl
    metrics.json
    run_config.json
    visualizations/
```

`run_config.json` records hashes for the policy file, Test80 split, structured
Qwen categories, and original three-task Qwen predictions. Each prediction
stores the effective task policy separately from the raw detector threshold.
After completion, report the frozen metrics as held-out results; do not run the
calibration script on Test80 and do not modify policy thresholds.

## Stage 11: Frozen Test80 Reporting and Failure Attribution

Generate the final comparison package from saved outputs only:

```bash
python scripts/finalize_locked_test_report.py
```

The script validates all `240` questions, requires zero recorded errors, checks
the dataset, split, structured-prediction, original-answer, and policy hashes,
and recomputes the locked metrics. It also replays the original shared score
`0.30` policy from the saved raw annotations as a pre-defined ablation. No model
is loaded and no Test threshold is selected.

The frozen held-out result is:

- original Qwen exact accuracy: `0.566667`;
- locked forced exact accuracy: `0.733333`;
- locked selective exact accuracy: `0.801980` at `0.841667` coverage;
- listing macro F1: `0.858804`;
- existence accuracy: `0.975000`;
- spatial forced accuracy: `0.675000`, with selective accuracy `0.913793` at
  `0.725000` coverage.

The generated directory is:

```text
outputs/eval_answering_v0/
  test__locked-task-aware__box-0.30__text-0.30/
    final_report/
      final_report.md
      final_summary.json
      policy_comparison.csv
      task_metrics.csv
      failure_analysis.jsonl
      failure_analysis.csv
      failure_summary.json
      figures/
        policy_comparison.png
        task_performance.png
        failure_breakdown.png
```

The Dev-selected spatial score `0.45` is more conservative on Test80 than the
pre-defined shared score `0.30` ablation. Keep the locked result as the official
held-out result. Any spatial-policy revision must use a new validation protocol
and a new untouched test set, never post-hoc Test80 tuning.

## Stage 12: Cross-Dataset Hard-Case Candidate Index

Use the frozen Test80 failure taxonomy to find new examples rather than copying
failed Test80 records into a tuning set. The first cross-dataset version uses
Open Images validation for broader boxable categories and Visual Genome for
explicit spatial relationships.

Download these official metadata files on a connected machine:

```text
Open Images validation boxes:
https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv

Open Images boxable class names:
https://storage.googleapis.com/openimages/v5/class-descriptions-boxable.csv

Open Images V7 human-verified validation image labels:
https://storage.googleapis.com/openimages/v7/oidv7-val-annotations-human-imagelabels.csv

Open Images V7 complete class names:
https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions.csv

Visual Genome relationships:
https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/relationships.json.zip

Visual Genome image metadata:
https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/image_data.json.zip
```

The Visual Genome URLs above use the dataset author's University of Washington
mirror. The v1.4 relationship link still shown on the primary Visual Genome
download page currently returns HTTP 404.

After extraction, use this layout on the server:

```text
data/raw/
  open_images/
    validation-annotations-bbox.csv
    class-descriptions-boxable.csv
    oidv7-val-annotations-human-imagelabels.csv
    oidv7-class-descriptions.csv
  visual_genome/
    relationships.json
    image_data.json
```

Build the deterministic candidate manifest:

```bash
python scripts/build_cross_dataset_hard_v1.py \
  --open-images-count 200 \
  --visual-genome-count 200 \
  --dev-fraction 0.5 \
  --seed 2026
```

The loader streams the large Visual Genome relationship array instead of
loading it all into memory. Candidate difficulty is transparent and based on
small/tiny objects, occlusion, truncation, grouped or repeated instances, scene
density, category diversity, source-relative long-tail classes, and spatial
relationships. Selection rewards difficult records while retaining tag and
category diversity.

Generated files:

```text
data/cross_dataset_hard_v1/
  candidates.jsonl
  download_manifest.jsonl
  manifest.json
  splits/
    dev_sample_ids.json
    test_sample_ids.json
```

Important annotation rules:

- Open Images boxes are exhaustive only for verified boxable classes. Any
  listing question must use that declared vocabulary rather than claim to list
  every visible noun.
- Visual Genome candidates contain only question-safe explicit
  `left/right/above/below` relationships. Each endpoint must be either the only
  annotated instance of its category or its unique largest instance across all
  relationship endpoints in the image. They are valid for spatial questions,
  not negative existence or exhaustive object-listing questions.
- Visual Genome records with a `coco_id` from the existing 100-image benchmark
  are excluded before selection.
- Open Images and cross-source duplicate photographs cannot be ruled out by
  source IDs. Download only the selected images from `download_manifest.jsonl`,
  then run exact and perceptual content deduplication before freezing Hard-Test.
- Do not evaluate or tune on the provisional Hard-Test split until deduplication,
  image validation, and question generation are complete.

## Stage 13: Selected-Image Download and Content Deduplication

Download only the 400 selected candidates and compare them with the existing
COCO100 image pixels:

```bash
python scripts/download_and_dedup_hard_images.py --workers 8
```

The command resumes valid existing files and retries missing or corrupt images.
Every decoded image receives its byte size, dimensions, format, SHA-256, and
64-bit difference hash. Exact hashes are safe for automatic exclusion; near
matches use dHash distance `<=4` plus a 3% aspect-ratio tolerance and remain
manual-review items.

Open Images pixels are fetched from CVDF's public
`open-images-dataset.s3.amazonaws.com/validation/` bucket. The older
`storage.googleapis.com/openimages/2018_04/validation/` path now returns HTTP
403 and must not be used for new manifests.

Audit outputs:

```text
data/cross_dataset_hard_v1/
  images/
    open_images/
    visual_genome/
  image_audit/
    downloads.jsonl
    reference_images.jsonl
    duplicate_pairs.jsonl
    sample_status.jsonl
    accepted_sample_ids.json
    review_sample_ids.json
    excluded_sample_ids.json
    summary.json
```

Do not freeze Hard-Test while `summary.json` reports `review_required`. Review
the near-duplicate pairs, replenish download failures and exclusions from the
eligible source pools, rerun the audit, and freeze only when all 400 final image
IDs are valid and split-disjoint.

## Stage 14: Immutable Freeze and Source-Aware Questions

Candidate regeneration invalidates an older pixel audit whenever its IDs differ.
Refresh the pixels and audit first, then freeze:

```bash
python scripts/download_and_dedup_hard_images.py --workers 8
python scripts/freeze_cross_dataset_hard_v1.py
```

The freeze command requires exactly 400 accepted IDs, two disjoint 200-image
splits, 200 images per source, and matching SHA-256 hashes for every pixel file.
It copies the candidate and image manifests into `frozen/` and refuses to
overwrite a different snapshot. Rerunning against unchanged inputs performs a
verification pass.

Generate questions only after the frozen snapshot exists:

```bash
python scripts/build_cross_dataset_hard_questions.py
```

Question validity is source-dependent:

- Open Images listing uses a declared vocabulary containing annotated positives
  and up to four human-verified absent distractors. A sample without a verified
  absent boxable category uses a positive-only vocabulary and is explicitly
  flagged; it never infers absence from a missing box. The task does not claim
  exhaustive coverage of arbitrary visible nouns.
- Open Images existence is exactly balanced. Positive answers require bounding
  boxes; negative answers require an official human-verification row with
  `Confidence=0`.
- Open Images spatial relations use the largest non-group annotated instance of
  each category and box-center geometry.
- Visual Genome contributes only explicit spatial relationships with uniquely
  referable endpoints; it contributes no listing or negative-existence tasks.

The deterministic default contains 800 questions: 200 Open Images listing, 200
balanced Open Images existence, 200 Open Images relation, and 200 Visual Genome
relation questions. `questions_v1/manifest.json` hashes the frozen pixels,
candidate annotations, official label metadata, and generated artifacts.

## Stage 15: Hard-Dev Qwen3-VL Baseline

Run a no-model preflight first:

```bash
python scripts/batch_eval_vlm.py \
  --dataset data/cross_dataset_hard_v1/questions_v1/dev_questions.jsonl \
  --output-dir outputs/cross_dataset_hard_v1/vlm \
  --run-name hard-dev400__qwen3-vl-8b-instruct \
  --required-split dev \
  --preflight-only \
  --local-files-only
```

The expected summary is 400 questions over 200 images: 100 listing, 100
existence, and 200 relation questions. It must report one `dev` split and a
verified artifact hash. No run directory or model is created by preflight.

On the server, run eight questions first and then resume the same run name:

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

The second command skips the eight completed records. `metrics.json` reports
overall, per-task, per-source, and source-by-task metrics. The restricted
Open Images listing parser only scores labels from each question's frozen
`allowed_categories`; previous COCO80 results retain their original parser.
Keep `test_questions.jsonl` untouched until the full Dev analysis and policy
choices are frozen.

## Stage 16: Hard-Dev Failure Analysis

Analyze the completed predictions entirely offline:

```bash
python scripts/analyze_hard_vlm_failures.py
```

The command requires exact 400-question coverage, rejects non-Dev records, and
replays every saved score. It writes `summary.json`, `per_sample.jsonl`,
`per_sample.csv`, and `report.md` under the run's `failure_analysis/` directory.
Relation metrics include per-label confusion, balanced accuracy, and the
majority-class baseline for each source.

The first Qwen3-VL baseline shows:

- Open Images relation accuracy/balanced accuracy: `0.43 / 0.419666`, with 26
  refusal or category-absence claims and 22 token-limit hits.
- Visual Genome relation accuracy/balanced accuracy: `0.58 / 0.490193`; the raw
  accuracy is below its `0.61` majority-class baseline because 61/100 labels are
  `above`.
- Positive/negative existence accuracy: `0.98 / 0.64`.
- Listing macro F1: `0.875745`; 59 records add at least one allowed distractor
  and 35 miss at least one target category.

Do not increase generation length to hide relation failures. The intended
answer is a single label; use a Dev-only forced-choice prompt revision and
compare it against this immutable baseline.

## Stage 17: Dev-Only Relation Prompt V2

Build the deterministic prompt variant:

```bash
python scripts/build_hard_dev_relation_prompt_v2.py
```

Only the 200 Dev relation questions change. The prompt explicitly treats both
named instances as present, defines the relation using visual centers, and
requires exactly one of the four benchmark labels. Listing and existence text,
all answers, images, model settings, and scoring remain unchanged.

Run eight records first, then resume all 200 relation records:

```bash
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

Keep v1 predictions unchanged for paired comparison. Do not generate or inspect
a Test prompt v2 until the Dev comparison rule and acceptance criteria are
written and frozen.

After the v2 run completes, generate the paired report:

```bash
python scripts/compare_hard_relation_prompts.py
```

The observed Dev result is:

- Overall accuracy: `0.505 -> 0.56`; balanced accuracy:
  `0.503657 -> 0.590457`; parse-valid rate: `0.815 -> 1.0`.
- Token-limit hits: `25 -> 0`; mean relation latency:
  `0.997669s -> 0.392451s`.
- Open Images accuracy: `0.43 -> 0.59`, with 19 v2-only correct versus 3
  v1-only correct (`p=0.000855`).
- Visual Genome accuracy: `0.58 -> 0.53`, with 6 v2-only correct versus 11
  v1-only correct (`p=0.332306`); balanced accuracy rises only from
  `0.490193` to `0.508782`.

Therefore v2 passes the Open Images source gate but not the Visual Genome gate.
Do not globally replace v1. The next policy candidate should be source-aware,
and Visual Genome requires a separate Dev decision before any Test artifact is
created.

## Stage 18: Visual Genome Relation Prompt V3

Visual Genome relation labels come from explicit human relationship
annotations. They should not inherit the box-center definition used by the
Open Images relation task. Build the VG-only Dev candidate:

```bash
python scripts/build_hard_dev_vg_relation_prompt_v3.py
```

The immutable artifact contains exactly 100 Visual Genome Dev relation
questions under `data/cross_dataset_hard_v1/questions_v3_dev_vg/`. It keeps
images, answers, evidence, and instance-selection rules unchanged. The prompt
treats both instances as present, asks for their depicted semantic spatial
relationship, and requires exactly one of four labels. It never mentions
object centers and does not create a Test artifact.

Run an eight-record smoke test and then resume the same run over all 100
questions:

```bash
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

The acceptance criteria were fixed in the v3 manifest before inference:

- Parse-valid rate must be at least `0.98`.
- No answer may hit `max_new_tokens`.
- Balanced accuracy must be at least the v2 VG result, `0.508782`.
- Exact accuracy must be at least the v1 VG result, `0.58`.

All four conditions must pass before v3 can replace v1 for Visual Genome. If it
passes, freeze a source-aware relation policy: v2 for Open Images and v3 for
Visual Genome. Keep Hard-Test untouched until that decision and its comparison
report are complete.

## Stage 19: Lock the Source-Aware Relation Policy

After all three Dev prediction files are present, recompute the paired
comparisons and freeze the selected policy:

```bash
python scripts/select_hard_relation_prompt_policy.py
```

The command rejects incomplete runs, nonempty error files, non-Dev run
configurations, mismatched IDs, changed ground truth, and a v3 manifest that is
not the frozen Visual Genome Dev100 artifact. It writes:

```text
outputs/cross_dataset_hard_v1/relation_prompt_policy_dev_v1/
  summary.json
  selected_policy.json
  paired_transitions.jsonl
  report.md
```

The observed paired results are:

- Open Images v1 to v2: accuracy `0.43 -> 0.59`, balanced accuracy
  `0.419666 -> 0.564289`, and McNemar `p=0.000855`.
- Visual Genome v1 to v3: accuracy `0.58 -> 0.62`, balanced accuracy
  `0.490193 -> 0.591774`, and McNemar `p=0.454498`.
- Visual Genome v2 to v3: accuracy `0.53 -> 0.62`, balanced accuracy
  `0.508782 -> 0.591774`, and McNemar `p=0.035156`.

The locked policy is source-aware: Open Images uses center-based prompt v2 and
Visual Genome uses semantic prompt v3. Report the non-significant v1-to-v3
McNemar result rather than describing every comparison as significant. The v3
selection remains valid under the acceptance gate fixed before its predictions
were generated. Rerunning the command verifies byte-identical locked artifacts
and refuses changed output.

## Stage 20: Locked Hard-Test Construction and Preflight

Mechanically apply the immutable Dev-selected policy to all 400 Hard-Test
questions:

```bash
python scripts/build_locked_hard_test_questions.py
```

The builder verifies the policy lock and original Test hash. Exactly 200
relation prompts change: Open Images receives v2 and Visual Genome receives v3.
The 100 listing and 100 existence records remain unchanged, and all IDs,
images, answers, categories, and evidence boxes are preserved. The resulting
manifest declares a single complete held-out evaluation protocol.

Run the no-model preflight:

```bash
python scripts/batch_eval_vlm.py \
  --dataset data/cross_dataset_hard_v1/questions_locked_test_v1/test_questions.jsonl \
  --output-dir outputs/cross_dataset_hard_v1/vlm \
  --run-name hard-test400__qwen3-vl-8b-instruct__source-aware-policy-v1 \
  --required-split test \
  --max-new-tokens 64 \
  --preflight-only \
  --local-files-only
```

Expected preflight counts are 400 questions, 200 images, 100 listing, 100
existence, and 200 relation questions. It must report only the `test` split and
a verified artifact hash without creating a run directory.

Do not run a Test smoke subset. The evaluator rejects `--max-samples` whenever
`--required-split test` is active. Start the complete locked server run:

```bash
CUDA_VISIBLE_DEVICES=3 python scripts/batch_eval_vlm.py \
  --dataset data/cross_dataset_hard_v1/questions_locked_test_v1/test_questions.jsonl \
  --output-dir outputs/cross_dataset_hard_v1/vlm \
  --run-name hard-test400__qwen3-vl-8b-instruct__source-aware-policy-v1 \
  --required-split test \
  --max-new-tokens 64 \
  --local-files-only
```

An interrupted run may be resumed only with the identical command. Do not
inspect partial metrics or retune any prompt, threshold, model, decoding
setting, or evaluation rule after Test inference begins.

## Stage 21: Immutable Hard-Test Final Report

After the locked Test run reaches 400/400 with zero errors, finalize it:

```bash
python scripts/finalize_locked_hard_test_report.py
```

If a workspace sync omitted the derived Dev policy directory, first reconstruct
it from the unchanged Dev predictions:

```bash
python scripts/select_hard_relation_prompt_policy.py
```

The finalizer verifies the complete chain from frozen Test questions through
the selected Dev policy and server run configuration. It rescores all 400
predictions rather than trusting the saved metrics file. The immutable output
contains:

```text
final_report/
  manifest.json
  summary.json
  report.md
  generalization.csv
  relation_confusion.csv
  per_sample_analysis.jsonl
  per_sample_analysis.csv
```

The held-out primary results are:

- Existence exact accuracy: `0.85`.
- Listing macro F1: `0.873664`.
- Relation exact accuracy: `0.59`.
- Relation balanced accuracy: `0.610559`.
- Open Images relation accuracy/balanced accuracy: `0.62 / 0.592040`.
- Visual Genome relation accuracy/balanced accuracy: `0.56 / 0.491330`.

Dev-to-Test listing F1 is stable (`-0.002081`), while Open Images relation
accuracy improves by `0.03`. Visual Genome relation accuracy drops by `0.06`
and balanced accuracy drops by `0.100444`; its raw Test accuracy is also below
the `0.60` majority-class baseline. Report this source-specific
generalization weakness explicitly.

Read-only attribution finds 13 negative-existence false positives, 2 positive
false negatives, 53 listings with an extra allowed category, 41 listings with
a missed target, and 82 wrong-direction relation predictions. These findings
may guide a future benchmark version trained and selected without reusing
Hard-Test, but they must not alter the frozen policy or its reported result.

## Stage 22: Interactive Demo and Frozen Result Explorer

Install the separate demo dependencies after the base environment:

```bash
python -m pip install -r requirements-demo.txt
```

Start the complete application on the selected physical GPU:

```bash
CUDA_VISIBLE_DEVICES=3 python scripts/launch_demo.py \
  --server-name 0.0.0.0 \
  --server-port 7860
```

The application has three operational views:

- `Assistant` runs Qwen3-VL, parses structured evidence targets, and invokes
  Grounding DINO plus SAM 2.1 for visible box and mask evidence.
- `Benchmark Explorer` filters and inspects the immutable Hard-Test400
  predictions and evidence without rerunning or changing the benchmark.
- `Evaluation` summarizes Hard-Test400, COCO Test80, and grounded-answering
  Test240 and exposes the frozen reports.

Both model stacks are loaded lazily on the first live request. The Gradio queue
uses a concurrency limit of one so the 8B VLM and Grounded-SAM-2 do not compete
for the same GPU during separate requests. The default runtime paths and
thresholds are recorded in `configs/demo.yaml`.

Use read-only mode for UI inspection on a CPU workstation or before the model
weights are available:

```bash
python scripts/launch_demo.py \
  --results-only \
  --server-name 127.0.0.1 \
  --server-port 7860
```

Read-only mode intentionally disables the live `Run` button while retaining
the complete benchmark and evaluation views. It does not alter any frozen
artifact.

## Stage 23: Live Answer-to-Grounding Dev Evaluation

The earlier grounded-answering protocol reuses one saved structured COCO80
listing per image. The interactive Demo instead asks Qwen to generate an answer
and evidence targets for every individual question. Stage 23 evaluates that
exact live chain without replacing or modifying the frozen Test240 report.

Run the no-model preflight:

```bash
python scripts/batch_eval_live_pipeline.py --preflight-only
```

Expected coverage is 20 Dev images and 60 questions, with 20 questions for each
task. There are 49 questions with positive evidence and 11 questions where an
empty evidence result is correct. The preflight also verifies each image and
each referenced COCO segmentation annotation.

Run a two-image, six-question smoke test:

```bash
CUDA_VISIBLE_DEVICES=3 python scripts/batch_eval_live_pipeline.py \
  --max-images 2
```

The automatic smoke run name contains `smoke-2`, so it cannot contaminate or
block the complete Dev run. After the smoke reaches 6/6 with zero errors, run
the full protocol:

```bash
CUDA_VISIBLE_DEVICES=3 python scripts/batch_eval_live_pipeline.py
```

The script loads Qwen3-VL and Grounded-SAM-2 once, runs them serially on GPU 3,
and supports exact-command resume. It writes:

```text
outputs/eval_live_pipeline_v0/
  dev__live-answer-grounding-v1__qwen3-vl-8b-instruct__box-0.30__text-0.30/
    run_config.json
    predictions.jsonl
    errors.jsonl
    metrics.json
    artifacts/
```

`metrics.json` reports:

- answer accuracy/F1 and relation balanced accuracy;
- JSON schema-valid rate and parse-source counts;
- evidence-target micro/macro precision, recall, F1, misses, and hallucinations;
- question-conditioned Box IoU50 and SAM Mask IoU50;
- correct empty evidence and false-positive behavior on negative questions;
- answer-plus-evidence end-to-end success rates;
- Qwen, Grounding DINO, SAM 2, and total latency plus CUDA peak allocation.

The Test split is locked by default. `--allow-test` is an explicit gate and
partial Test runs are prohibited. Do not open that gate until all Stage 23 Dev
analysis and the next evidence-verification policy are frozen.

## Stage 24: Task-Aware Live Prompt Policy

The generic Stage 23 prompt produced valid JSON for all 60 Dev questions, but
its task contract was too loose. The frozen baseline achieved relation
accuracy and parse-valid rate of only `0.15 / 0.15`; it also produced open
phrases outside COCO-80 and reached target micro F1 `0.706827`, Box IoU50 F1
`0.421286`, Mask IoU50 F1 `0.416851`, and end-to-end any/complete success
`0.483333 / 0.383333`.

Stage 24 changes one variable: the system prompt. Detector thresholds, model
weights, decoding, dataset, split, evaluator, and runtime remain unchanged.
Prompt construction reads only `task_type` and `question`:

- listing must use exact COCO-80 category names;
- existence must answer exactly `yes` or `no`, with evidence only for `yes`;
- relation must choose one of four labels and return both question entities.

The acceptance thresholds and immutable generic-baseline hashes were fixed
before candidate inference in `configs/live_prompt_policy_v1.yaml`.

Stop the Gradio process so both model stacks have a free GPU, then preflight:

```bash
python scripts/batch_eval_live_pipeline.py \
  --preflight-only \
  --prompt-policy task-aware-coco-v1
```

Expected output is 60 Dev questions, 20 images, 20 questions per task, 49
positive-evidence questions, and 11 negative-evidence questions. Run the
separate six-question smoke test:

```bash
CUDA_VISIBLE_DEVICES=3 python scripts/batch_eval_live_pipeline.py \
  --prompt-policy task-aware-coco-v1 \
  --max-images 2
```

After 6/6 completes with zero errors, run or resume the full candidate:

```bash
CUDA_VISIBLE_DEVICES=3 python scripts/batch_eval_live_pipeline.py \
  --prompt-policy task-aware-coco-v1
```

The full candidate directory is:

```text
outputs/eval_live_pipeline_v0/
  dev__live-answer-grounding-v1__qwen3-vl-8b-instruct__box-0.30__text-0.30__task-aware-coco-v1/
```

New runs store gallery files as project-relative paths. For the original
generic run, create a portable sidecar while preserving its locked
`predictions.jsonl` hash:

```bash
python scripts/export_portable_live_predictions.py
```

After synchronizing the candidate output to the local workspace, compare it:

```bash
python scripts/compare_live_prompt_policies.py
```

The comparator rejects incomplete or mismatched runs, verifies baseline and
manifest hashes, independently replays both metric files, computes per-task
paired answer transitions with exact McNemar tests, and writes:

```text
outputs/eval_live_pipeline_v0/prompt_policy_dev_v1/
  summary.json
  paired_transitions.jsonl
  report.md
```

Adopt `task-aware-coco-v1` only if every pre-registered gate passes. A failed
gate means revise on Dev under a new policy name; it is not permission to open
or tune on Test.

## Stage 25: Compact Listing Prompt V2

Stage 24 v1 improved relation accuracy from `0.15` to `0.70`, target F1 from
`0.706827` to `0.756303`, Box IoU50 F1 from `0.421286` to `0.548673`, and
Mask IoU50 F1 from `0.416851` to `0.539823`. It nevertheless failed its
pre-registered schema gate because two listing responses copied long portions
of the vocabulary and stopped at the 192-token generation limit. Schema
validity was `58/60 = 0.966667`, below the required `0.98`.

The v2 experiment changes only the listing system prompt. Dev listing samples
contain at most five ground-truth categories, so v2 caps predictions at eight
without truncating a target answer. It requires high-confidence visible
objects, prohibits copying the vocabulary, and tells the model to omit
uncertain categories. Existence and relation prompts remain byte-identical.

The frozen v1 prediction and metrics hashes, v2 prompt-template hash, and all
non-regression thresholds are recorded before inference in
`configs/live_prompt_policy_v2.yaml`.

Run the no-model preflight and separate smoke test:

```bash
python scripts/batch_eval_live_pipeline.py \
  --preflight-only \
  --prompt-policy task-aware-coco-v2

CUDA_VISIBLE_DEVICES=3 python scripts/batch_eval_live_pipeline.py \
  --prompt-policy task-aware-coco-v2 \
  --max-images 2
```

After smoke reaches 6/6 with zero errors, run the complete Dev60 candidate:

```bash
CUDA_VISIBLE_DEVICES=3 python scripts/batch_eval_live_pipeline.py \
  --prompt-policy task-aware-coco-v2
```

The automatic full output directory ends in `__task-aware-coco-v2`; it cannot
overwrite v1. After synchronizing the output, compare it against frozen v1:

```bash
python scripts/compare_live_prompt_policies.py \
  --manifest configs/live_prompt_policy_v2.yaml
```

The comparator independently replays both metric files and writes the v1-v2
summary, paired transitions, and report under:

```text
outputs/eval_live_pipeline_v0/prompt_policy_v1_vs_v2_dev/
```

The v2 policy may be locked only if all pre-registered gates pass. Keep Test
closed until that decision is recorded.

## Stage 26: Locked Live-Pipeline Held-Out Test

The complete v1-v2 Dev comparison passed all ten pre-registered gates. V2
restored schema validity to `1.0`, improved listing macro F1 to `0.823903` and
target micro F1 to `0.878505`, retained relation accuracy at `0.70`, and
reduced mean latency to `1.453971` seconds. Small box and mask F1 decreases
remained above their frozen non-regression thresholds.

Freeze the selected policy:

```bash
python scripts/lock_live_prompt_policy.py
```

The freezer independently replays both Dev metric files and the paired
comparison. It verifies v1, v2, manifest, template, run-config, error-file,
prediction, metric, and transition hashes. It then writes deterministic
artifacts under:

```text
outputs/eval_live_pipeline_v0/locked_policy_v1/
  selected_policy.json
  test_protocol.json
  report.md
```

Rerunning the freezer must report `verified`; any byte difference is rejected.
The locked Test contains 80 images and 240 questions: 80 listing, 80
existence, and 80 relation. It includes 201 positive-evidence questions and 39
negative-evidence questions.

After synchronizing the lock directory to the server, preflight without model
loading:

```bash
python scripts/batch_eval_live_pipeline.py \
  --split-image-ids data/eval_v0/splits/test_image_ids.json \
  --prompt-policy task-aware-coco-v2 \
  --allow-test \
  --preflight-only
```

The evaluator validates the locked policy, prompt template, dataset, Test
image IDs, selected 240 question IDs, COCO ground truth, expected coverage, and
frozen run name. A missing or changed lock fails before either model loads.

Run the one complete held-out evaluation:

```bash
CUDA_VISIBLE_DEVICES=3 python scripts/batch_eval_live_pipeline.py \
  --split-image-ids data/eval_v0/splits/test_image_ids.json \
  --prompt-policy task-aware-coco-v2 \
  --allow-test
```

Do not run a Test smoke subset. Do not inspect partial metrics. If interrupted,
resume only with the identical command and do not change the prompt, model,
thresholds, decoding, evaluator, protocol, or run name.

## Stage 27: Immutable Live-Pipeline Test240 Report

After the locked Test reaches 240/240 with zero errors, finalize it locally:

```bash
python scripts/finalize_locked_live_test_report.py
```

The finalizer verifies the Test protocol, selected policy, dataset, split,
COCO ground truth, runtime source hashes, run configuration, unique ordered
prediction IDs, and every referenced box/mask visualization. It independently
replays all saved metrics from `predictions.jsonl`; a mismatch stops
finalization.

The immutable output is written under the locked Test run:

```text
final_report/
  manifest.json
  summary.json
  report.md
  generalization.csv
  relation_confusion.csv
  per_sample_analysis.jsonl
  per_sample_analysis.csv
```

Run the finalizer a second time. It must report `verified`, proving every
derived artifact is byte-identical.

The final held-out results are:

- Overall exact accuracy: `0.658333`.
- Listing macro F1: `0.749713`.
- Existence exact accuracy: `0.95`.
- Relation exact/balanced accuracy: `0.7375 / 0.719907`.
- Structured target micro F1: `0.840580`.
- Box and mask IoU50 micro F1: `0.502114 / 0.502114`.
- Complete end-to-end success: `0.591667`.
- Mean latency: `1.296948` seconds.

Compared with Dev, listing F1 drops `0.074190`, target F1 drops `0.037925`,
box/mask F1 drop `0.033218 / 0.028935`, while relation exact accuracy improves
`0.0375` and complete end-to-end success improves `0.025`. One listing
response reaches the 192-token limit, leaving schema validity at
`239/240 = 0.995833`. Preserve this as a final limitation; Test-driven prompt,
threshold, or parser changes are prohibited.

## Stage 28: Final Dashboard and Career Package

The Gradio `Evaluation` tab now treats the locked live-pipeline Test240 report
as the primary result. It displays answer, target, box, mask, end-to-end,
runtime, and integrity metrics; the Dev-to-Test table; the relation confusion
table; balanced success/failure evidence; and downloadable final artifacts.
Earlier frozen experiments remain visible as supporting context.

The career-facing materials are:

- `docs/resume_project_description.md`: Chinese and English resume bullets,
  concise variants, and claim boundaries.
- `docs/system_architecture.md`: online inference and offline evaluation
  Mermaid diagrams, module responsibilities, and engineering decisions.
- `docs/interview_notes.md`: 30-second and two-minute pitches, STAR framing,
  metrics, limitations, and common technical follow-ups.

All career-facing metrics come from the immutable Test240 report. The project
must be described as answer-followed-by-evidence localization, not as
evidence-conditioned answer correction or an official benchmark result.

## Stage 29: Official COCO POPE Data Preparation

`scripts/prepare_pope_data.py` downloads the three official COCO POPE question
files and only the COCO val2014 images referenced by them. It deliberately
avoids the full val2014 archive.

The validated local dataset contains:

- `9000` questions and `500` unique images.
- `3000` random, `3000` popular, and `3000` adversarial questions.
- `1500` yes and `1500` no labels within every strategy.
- `500/500` images passing Pillow decoding and SHA-256 audit.
- `0` missing or invalid images.

The generated layout is:

```text
data/pope/
  annotations/
    coco_pope_random.json
    coco_pope_popular.json
    coco_pope_adversarial.json
  images/
    COCO_val2014_*.jpg
  questions.jsonl
  image_manifest.jsonl
  image_audit.jsonl
  summary.json
  manifest.json
```

`questions.jsonl` uses the project schema and prefixes question IDs with the
sampling strategy, preventing collisions across the three official files.
`manifest.json` records source URLs, content hashes, derived artifact hashes,
and expected counts. Use `--audit-only` after transferring the directory to
the offline server.

## Stage 30: Official-Compatible POPE Baseline

`scripts/batch_eval_pope.py` is the dedicated Qwen3-VL baseline runner. It does
not reuse the generic VLM smoke result because POPE requires strategy-level
binary metrics and a specific answer conversion.

The frozen baseline protocol includes:

- A system prompt requesting exactly one `Yes` or `No` answer.
- Deterministic decoding with `max_new_tokens=4`.
- The official POPE answer conversion and metrics.
- A separate strict Yes/No parser to expose malformed model outputs.
- Resumable predictions, error logging, selected-ID hashing, and immutable run
  configuration.
- Per-strategy and overall Accuracy, Precision, Recall, F1, and Yes Ratio.
- Latency, generated-token, token-limit, CUDA memory, and runtime metadata.

Use the balanced smoke90 run first:

```bash
CUDA_VISIBLE_DEVICES=3 python scripts/batch_eval_pope.py \
  --samples-per-strategy 30 \
  --run-name pope-smoke90__qwen3-vl-8b-instruct \
  --local-files-only
```

This selects 30 questions per strategy with 15 positive and 15 negative labels
within each strategy. It is a runtime validation, not a reportable benchmark
result.

After checking zero errors and strict parse validity, run the complete official
selection:

```bash
CUDA_VISIBLE_DEVICES=3 python scripts/batch_eval_pope.py \
  --require-complete \
  --run-name pope-full9000__qwen3-vl-8b-instruct \
  --local-files-only
```

Resume an interrupted evaluation only with the identical command. The
`--require-complete` gate verifies 3000 questions and balanced labels for every
strategy before model loading.

## Stage 31: POPE Error Attribution

`scripts/analyze_pope_errors.py` performs a no-model audit of the completed
POPE baseline:

```bash
python scripts/analyze_pope_errors.py
```

Before producing any report, it independently replays the saved answer parser,
checks all 9000 unique prediction IDs, reproduces the confusion matrix, and
requires a completed all-strategy run config. The generated
`error_analysis/` directory contains:

```text
error_analysis/
  summary.json
  errors.jsonl
  per_object.csv
  per_image.csv
  representative_cases.jsonl
  false_negative_cases.jpg
  false_positive_cases.jpg
  report.md
```

The official confusion matrix remains question based. For qualitative review,
the analyzer additionally deduplicates semantic queries because every positive
question is repeated in the random, popular, and adversarial files. It reports
raw and unique FN/FP counts, object-level recall and false-positive rate,
hardest images, cross-strategy consistency, and balanced representative cases.

Treat a POPE false positive as a benchmark disagreement, not automatically as
proven hallucination. Negative labels are derived from COCO annotations, so
annotation omissions, synonyms, and category boundaries must be checked in the
rendered case sheets before assigning a causal failure label.

## Stage 32: Grounding-Aware Answer Verification V1

The previous task-aware consensus policy only used detector disagreement to
abstain; it did not change the forced VLM answer. Stage 32 introduces the first
answer-correction policy:

```text
saved Qwen Yes/No answer
        +
question object -> Grounding DINO -> SAM 2.1 evidence
        |
high-confidence localized evidence (score >= 0.45)
        |
rescue No -> Yes
```

The asymmetric `grounding_positive_rescue_v1` rule follows two constraints:

- A negative Qwen answer is promoted only when accepted localized evidence
  reaches the frozen promotion threshold.
- A positive Qwen answer is never demoted solely because the detector returned
  no box. Detector silence is not visual proof of absence.

The policy configuration is stored in
`configs/grounding_answer_verifier_v1.yaml`. Run a known POPE false-negative
case with:

```bash
CUDA_VISIBLE_DEVICES=3 python scripts/run_grounding_answer_verifier.py \
  --sample-id pope_coco_random_1139 \
  --local-files-only
```

The command reads the existing full POPE Qwen baseline, so it does not reload
or rerun the VLM. It writes `verification.json`, Grounding boxes, segmentation
masks, and compact evidence diagnostics under
`outputs/grounding_answer_verifier_v1/<sample-id>/`.

This is an implementation milestone, not an accuracy claim. The next stage
must pre-register batch selection, comparison baselines, threshold ablations,
paired significance tests, and runtime accounting before reporting method
improvements.

## Stage 33: POPE Verifier Batch Comparison Protocol

The one-sample verifier is extended to a resumable paired evaluator in
`scripts/batch_eval_pope_verifier.py`. It reads the frozen Qwen POPE
predictions instead of rerunning the VLM and evaluates the original and
verified answer on the same question.

POPE repeats some semantic queries across random, popular, and adversarial
files. The batch protocol hashes `(image_id, object, question)`, excludes the
ground-truth label and strategy from the key, and runs Grounded-SAM-2 only once
per unique query. The raw question-level protocol and per-strategy metrics are
preserved when cached evidence is projected back onto every source record.

Run the balanced engineering smoke test:

```bash
python scripts/batch_eval_pope_verifier.py \
  --samples-per-strategy 30 \
  --preflight-only \
  --local-files-only

CUDA_VISIBLE_DEVICES=3 python scripts/batch_eval_pope_verifier.py \
  --samples-per-strategy 30 \
  --run-name pope-verifier-smoke90__positive-rescue-v1 \
  --local-files-only
```

The current frozen baseline yields 90 questions and 53 unique grounding
queries in this selection. A completed run writes:

- `evidence.jsonl`: one Grounded-SAM-2 result per unique query;
- `predictions.jsonl`: paired baseline and verified outcomes per question;
- `metrics.json`: baseline, verified, delta, correction, latency, memory, and
  coverage summaries;
- `errors.jsonl`: failed attempts retained for restart diagnostics;
- `run_config.json`: source hashes, selected IDs, selected query keys, model
  paths, thresholds, and runtime options.

The run configuration is immutable on resume. A changed source hash, selected
set, model, or threshold requires a new run directory. `Smoke90` validates
execution, caching, recovery, and metrics only; it must not be used to tune
thresholds. Threshold candidates are pre-registered and selected on a
separate development split before the single locked full-9000 test.

## Stage 34: Semantic Crop Verifier V2

The V1 Smoke90 run completed correctly but promoted five true negative answers
and rescued no false negatives. The failure cases showed why a detector score
alone is insufficient: a red bus was accepted for `orange`, vehicles were
confused with `train` or `truck`, and one `dining table` mask occupied nearly
the complete snow image. False detections had scores as high as `0.767`, while
a true `car` candidate scored `0.449`.

V2 therefore introduces a second semantic gate:

```text
frozen Qwen No answer
        |
cached Grounding DINO + SAM 2.1 candidates
        |
evidence and maximum-area geometry gates
        |
up to two context crops per unique query
        |
deterministic Qwen exact Yes/No category review
        |
promote No -> Yes only after exact semantic Yes
```

The implementation is split across
`src/grounded_visual_assistant/semantic_answer_verifier.py`,
`src/grounded_visual_assistant/pope_semantic_verifier_evaluation.py`, and
`scripts/batch_eval_pope_verifier_v2.py`. The versioned configuration is
`configs/grounding_answer_verifier_v2.yaml`.

Run the input-only preflight and Smoke90:

```bash
python scripts/batch_eval_pope_verifier_v2.py \
  --samples-per-strategy 30 \
  --preflight-only \
  --local-files-only

CUDA_VISIBLE_DEVICES=3 python scripts/batch_eval_pope_verifier_v2.py \
  --samples-per-strategy 30 \
  --run-name pope-verifier-v2-smoke90__semantic-rescue \
  --local-files-only
```

The expected workload is 90 questions, 53 unique POPE queries, 40 unique
negative-baseline queries, 17 candidate queries, and 20 semantic crop reviews.
V1 evidence is reused, so Grounding DINO and SAM 2.1 are not loaded in this
stage. The output directory contains:

- `crops/`: deterministic context crops named by query and annotation index;
- `semantic_reviews.jsonl`: raw Qwen answer, parse, latency, memory, crop hash,
  and candidate diagnostics;
- `predictions.jsonl`: paired baseline and V2 question-level predictions;
- `metrics.json`: baseline/V2 deltas, corrections, McNemar p-value, semantic
  review statistics, optimized latency projection, and memory;
- `errors.jsonl` and `run_config.json`: restart diagnostics and immutable
  experiment provenance.

V2 is still an engineering candidate. Smoke90 can reject a broken mechanism,
but it cannot establish or tune the final method. The next stage must construct
the separate development protocol, pre-register ablations, and lock one V2
configuration before held-out evaluation.

## Stage 35: POPE-Isolated Verifier Dev110

The original `eval_v0` development split contains 20 COCO val2017 images, but
only one existence question per image. One of those images also appears in the
official POPE Full500 image set. Stage 35 creates a larger method-development
protocol while preserving POPE as held-out evaluation:

1. Exclude every Dev image ID referenced by POPE Full500.
2. Keep every annotated image/category pair from the remaining 19 images.
3. Create one positive existence question for each pair.
4. Pair it with an absent category from the same official COCO
   `supercategory` whenever possible.
5. For the singleton `person` supercategory or exhausted groups, choose a
   deterministic least-used absent category.

This produces 55 positive and 55 hard-negative questions over 19 images and 64
queried categories. Forty-two negatives share the positive object's
supercategory and 13 use the balanced fallback. The selection is algorithmic;
no question is chosen from V1/V2 success or failure outcomes.

Build and independently audit the data:

```bash
python scripts/build_verifier_dev_v1.py
python scripts/build_verifier_dev_v1.py --audit-only
```

The canonical files are `data/verifier_dev_v1/questions.jsonl` and
`data/verifier_dev_v1/manifest.json`. The builder is byte deterministic and the
audit checks source hashes, the 110-question hash, balance, unique IDs and
queries, official annotation consistency, image availability, pair structure,
and zero overlap with all POPE images.

Freeze the Qwen Dev110 baseline:

```bash
python scripts/batch_eval_verifier_dev.py \
  --preflight-only \
  --local-files-only

CUDA_VISIBLE_DEVICES=3 python scripts/batch_eval_verifier_dev.py \
  --run-name qwen-baseline-dev110__qwen3-vl-8b-instruct \
  --local-files-only
```

The runner uses deterministic exact `Yes`/`No` decoding, immutable input/model
hashes, incremental JSONL persistence, and restart-safe metrics. It reports
overall binary metrics, positive/hard-negative accuracy, supercategory
breakdown, both-correct pair rate, strict parse validity, latency, and GPU
memory. The next stage consumes only its frozen negative answers for
Grounded-SAM-2 evidence generation, then performs semantic-review ablations.

## Stage 36: Verifier Dev57 Grounding Evidence

The frozen Dev110 baseline predicts `No` for 57 questions: 54 true negatives
and three false negatives. The asymmetric verifier never demotes a positive
baseline answer, so processing the remaining 53 `Yes` answers would add cost
without changing any output.

`scripts/batch_ground_verifier_dev.py` builds jobs solely from strict baseline
`No` answers. Jobs contain image, question, target, baseline ID, and a GT-free
query key; they do not contain labels or pair roles. The ordered selection hash
is `117e46c4596700e3db55129305a8a1adaa1a82775e105fd9e9e18d36bb32f265`.

Run the preflight and evidence cache:

```bash
python scripts/batch_ground_verifier_dev.py \
  --preflight-only \
  --local-files-only

CUDA_VISIBLE_DEVICES=3 python scripts/batch_ground_verifier_dev.py \
  --run-name grounding-dev57__grounding-dino-base__sam2.1-base-plus \
  --visualize-limit 12 \
  --local-files-only
```

The versioned `configs/verifier_dev_grounding_v1.yaml` fixes Grounding DINO at
`box=0.30, text=0.30` with SAM 2.1 Base+. This is the lowest pre-registered
candidate threshold; score and geometry ablations reuse the same evidence
instead of rerunning the detector.

The run writes `evidence.jsonl`, `metrics.json`, `errors.jsonl`,
`run_config.json`, and optional qualitative artifacts. Evidence records contain
no GT. Metrics join labels only after inference and separately summarize
candidate presence for false-negative and true-negative baseline outcomes,
along with candidate counts, score bins, latency, and peak memory. The next
stage generates the union of semantic candidate crops from this cache and runs
the pre-registered V1/V2 module ablation.

## Stage 37: Verifier Dev23 Semantic Review

The completed Dev57 evidence contains 23 accepted candidates across 19
queries. `configs/verifier_dev_semantic_review_v1.yaml` defines the candidate
union used by every later ablation:

- grounding score at least `0.30`;
- mask score at least `0.50`;
- no maximum-area filtering during review generation;
- at most two candidates per query;
- `0.25` context padding and a minimum 160-pixel crop.

Keeping the unfiltered union is important: the `0.90` maximum mask-area gate
and one-versus-two-candidate policies can then be compared offline without
rerunning Qwen. The ordered 23-candidate hash is
`186c3a3e15bc9c901cdfe528518e2a422ffe5e334709ef53d68cc83001ab4f51`.

Run preflight and semantic review:

```bash
python scripts/batch_review_verifier_dev.py \
  --preflight-only \
  --local-files-only

CUDA_VISIBLE_DEVICES=3 python scripts/batch_review_verifier_dev.py \
  --run-name semantic-review-dev23__qwen3-vl-8b-instruct \
  --local-files-only
```

The script writes deterministic JPEG crops, crop SHA-256 values,
`semantic_reviews.jsonl`, `metrics.json`, `errors.jsonl`, and an immutable
`run_config.json`. Candidate jobs include no GT or pair role. The same strict
semantic prompt used by V2 Smoke requires an exact `Yes` or `No`; color, text,
context, and a related category are explicitly insufficient evidence.

Metrics join labels only after inference and report semantic-Yes rates for the
three false-negative and 54 true-negative baseline queries, exact parse
validity, token-limit hits, latency, and memory. After completion, all
pre-registered module ablations run solely from the frozen baseline, evidence,
and review artifacts.

## Stage 38: Verifier Dev110 Offline Ablation

Stage 38 consumes only the three frozen development artifacts:

- 110 Qwen baseline predictions;
- 57 Grounded-SAM-2 negative-query records;
- 23 deterministic semantic crop reviews.

The versioned `configs/verifier_dev_ablation_v1.yaml` pre-registers four score
thresholds (`0.30`, `0.40`, `0.45`, and `0.50`) and five controlled module
families. Together with the unchanged baseline, this creates 21 policies. The
grid isolates the grounding-only policy, the `0.90` maximum mask-area gate,
Top-1 versus Top-2 review cost, and the exact semantic confirmation gate.

Audit the frozen source chain, then run the comparison:

```bash
python scripts/compare_verifier_dev_ablations.py --audit-only
python scripts/compare_verifier_dev_ablations.py
```

No model is loaded. The script verifies all source/config hashes, all crop
hashes, exact semantic answers, GT-free inference artifacts, and the ordered
policy hash before evaluating labels offline. A policy is eligible only when:

1. accuracy strictly exceeds the frozen baseline;
2. F1 does not decrease;
3. beneficial corrections exceed harmful corrections.

The completed result rejects every verifier candidate:

| Policy result | Accuracy | Beneficial | Harmful | Net |
|---|---:|---:|---:|---:|
| Qwen baseline | 0.963636 | 0 | 0 | 0 |
| Best V1 score-only (`0.50`) | 0.936364 | 1 | 4 | -3 |
| V2 semantic (`0.30`) | 0.954545 | 1 | 2 | -1 |
| V2 semantic (`0.40`/`0.45`) | 0.954545 | 0 | 1 | -1 |
| V2 semantic (`0.50`) | 0.963636 | 0 | 0 | 0 |

The area gate does not change predictions because no Dev candidate exceeds
the `0.90` mask-area ratio. Top-1 reduces the score-`0.30` semantic workload
from 23 to 19 crops but does not change corrections. A score of `0.50` merely
turns V2 into a costly no-op, so a tie is not presented as an improvement.

The locked decision in `selected_policy.json` is
`retain_baseline_no_eligible_verifier`. Held-out POPE remains untouched for
selection. The next mechanism iteration must be designed and assessed on Dev
only; the full held-out evaluation is permitted only after one verifier
passes the pre-registered gates.

## Stage 39: Contrastive Category Verifier V3

Stage 38 shows that another target-biased Yes/No question is not enough:
`chair` is confused with a bed/couch region, and school buses are accepted as
`truck`. V3 adds a target-neutral category decision after V2:

```text
frozen Qwen No
    -> Grounding/SAM candidate
    -> V2 exact crop Yes
    -> red-box same-supercategory classification
    -> promote only when the exact V3 label equals the target
```

The cascade fixes `score=0.30`, mask score `0.50`, maximum mask-area ratio
`0.90`, and Top-1 before any V3 output is observed. This replays 19 V2 Top-1
candidates and sends only three exact V2-Yes candidates (`book`, `chair`, and
`truck`) to contrastive review.

The options come from `configs/coco_80_supercategories_v1.yaml`, a standalone
copy of the official COCO-80 ontology. It contains no image annotations or
experiment outcomes. Every option set contains all categories in the target's
supercategory plus `none`; options are sorted and do not mark the target.

Run local validation and optionally inspect the prepared crops:

```bash
python scripts/batch_review_verifier_dev_v3.py \
  --preflight-only \
  --local-files-only

python scripts/batch_review_verifier_dev_v3.py \
  --prepare-only \
  --local-files-only
```

Run the three deterministic reviews on the server:

```bash
CUDA_VISIBLE_DEVICES=3 python scripts/batch_review_verifier_dev_v3.py \
  --local-files-only
```

The run is resumable and source-hash locked. `jobs.jsonl` and all model inputs
exclude GT, pair roles, and expected labels. Each marked crop is hashed. After
all reviews complete, the script joins Dev labels offline and writes:

- `contrastive_reviews.jsonl` and `errors.jsonl`;
- `predictions.jsonl` for all 110 paired questions;
- `metrics.json` with baseline/V3 metrics and runtime projection;
- `v3_decision.json` with the same three Stage 38 acceptance gates.

The ordered three-job hash is
`c50a00cf39fdca04ac3a46053817569bcb983dabb4cec14919da8bb6ce485172`.
No held-out POPE record is read during V3 design or selection.

The completed exact labels are `chair`, `none`, and `car`. V3 correctly
rejects the V2 `truck` promotion by identifying a related vehicle, but it also
rejects the beneficial `book` rescue and retains the harmful `chair`
promotion. It changes one final answer, and that change is harmful. Accuracy
falls from `0.963636` to `0.954545`, F1 falls from `0.962963` to `0.954128`,
and the locked decision is `reject_v3_on_dev`. V3 therefore does not advance
to held-out evaluation.

## Stage 40: Final Verifier Freeze and Failure Audit

Stage 40 closes the answer-rewriting branch instead of continuing to tune on
three reviewed examples. The versioned configuration
`configs/verifier_dev_final_report_v1.yaml` fixes the representative V1, V2,
and V3 rows and declares the final deployment boundary:

- keep the frozen Qwen Dev110 answer policy;
- disable evidence-based answer rewriting;
- retain Grounding DINO and SAM 2.1 for localization and audit evidence;
- do not run a rejected verifier on held-out POPE.

Audit the complete source chain and generate the final report:

```bash
python scripts/finalize_verifier_dev_report.py --audit-only
python scripts/finalize_verifier_dev_report.py
```

The script does not load a model. It validates and recomputes the following
chain before writing output:

| Frozen source | Coverage |
|---|---:|
| Qwen baseline | 110 questions |
| Grounded-SAM-2 evidence | 57 negative queries |
| V2 semantic review | 23 crops |
| Stage 38 ablation | 21 policies |
| V3 contrastive review | 3 crops |

The controlled comparison retained in the report is:

| Variant | Accuracy | F1 | Beneficial | Harmful | Net |
|---|---:|---:|---:|---:|---:|
| Frozen Qwen baseline | 0.963636 | 0.962963 | 0 | 0 | 0 |
| V1 grounding-only (`0.50`) | 0.936364 | 0.938053 | 1 | 4 | -3 |
| V2 semantic rescue (`0.30`, Top-1) | 0.954545 | 0.954955 | 1 | 2 | -1 |
| V2 conservative no-op (`0.50`, Top-1) | 0.963636 | 0.962963 | 0 | 0 | 0 |
| V3 contrastive category review | 0.954545 | 0.954128 | 0 | 1 | -1 |

The six-case audit traces all four baseline errors (`remote`, `book`,
`handbag`, and `stop sign`) and both representative V2 regression risks
(`chair` and `truck`). It separates an unrechecked baseline false positive,
a detector recall miss, semantic false rejection, contrastive false
rejection, category ambiguity, and a cross-category confusion caught by V3.

Canonical outputs live in
`outputs/eval_verifier_final_v1/verifier-dev110-final/`:

- `final_policy.json`: the immutable answer-rewrite decision and claim bound;
- `variant_summary.csv/jsonl`: the compact five-variant comparison;
- `failure_analysis.json` and `failure_cases.csv/jsonl`: reproducible cases;
- `report.md`: engineering and interview interpretation;
- `artifact_manifest.json`: SHA-256 values for every generated artifact.

The scientific conclusion is deliberately negative: evidence improves
inspectability, but no tested verifier improves Dev answer quality under the
frozen gates. This is still a useful project result because the system makes
the rejection reproducible and prevents a weaker module from being deployed
or selectively reported.

## Stage 41: Final Gradio and Interview Package

The Gradio `Evaluation` page consumes the immutable Stage 40 artifacts through
`load_demo_metrics`. It has two nested result views:

1. `Held-Out Test240` retains the final live-pipeline metrics, Dev-to-Test
   generalization, relation confusion, evidence gallery, and report files.
2. `Verifier Audit` exposes the frozen deployment decision, five representative
   V1/V2/V3 rows, six traced failure/risk cases, and downloadable Stage 40
   artifacts.

The results-only mode remains the recommended interview path because it needs
no GPU and cannot accidentally trigger model inference:

```bash
python scripts/launch_demo.py \
  --results-only \
  --server-name 0.0.0.0 \
  --server-port 7860
```

`docs/resume_project_description.md`, `docs/system_architecture.md`, and
`docs/interview_notes.md` now share the same final claim boundary: the system
provides auditable evidence; three answer-rewrite mechanisms were evaluated
and rejected; the deployed policy retains the frozen Qwen answer. The compact
launch and reproduction procedure is recorded in
`docs/interview_demo_reproduction.md`.

This stage completes the required interview-project mainline. Further POPE,
RefCOCOg/GroundingME comparisons, larger models, or learned verification are
optional research extensions and must not retroactively change the frozen
Test240 or Verifier Dev110 results.
