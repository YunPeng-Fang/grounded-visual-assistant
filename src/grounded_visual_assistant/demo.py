"""Runtime and frozen-result helpers for the Gradio project demo."""

from __future__ import annotations

import json
import re
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml


GROUNDED_DEMO_SYSTEM_PROMPT = (
    "You are a grounded visual assistant. Answer the user's image question and "
    "identify the visible object instances needed as evidence. Return exactly "
    "one valid JSON object with this schema: "
    '{"answer":"concise answer","evidence_targets":["object phrase"]}. '
    "Use short concrete object noun phrases, include no more than six unique "
    "targets, and return an empty list when no object grounding is useful. Do "
    "not use Markdown or add text outside the JSON object."
)
EVIDENCE_MODE = "Answer + grounded evidence"
ANSWER_ONLY_MODE = "Answer only"
ALL_SOURCES = "All sources"
ALL_TASKS = "All tasks"
ALL_OUTCOMES = "All outcomes"
LIVE_TEST_RUN_NAME = (
    "test__live-answer-grounding-v1__qwen3-vl-8b-instruct__"
    "box-0.30__text-0.30__task-aware-coco-v2__locked-v1"
)
VERIFIER_FINAL_RUN_NAME = "verifier-dev110-final"


def _recover_json_object(value: str) -> tuple[dict[str, Any] | None, str]:
    stripped = value.strip()
    try:
        payload = json.loads(stripped)
        return (payload, "direct_json") if isinstance(payload, dict) else (None, "invalid")
    except json.JSONDecodeError:
        pass

    fenced = re.fullmatch(
        r"```(?:json)?\s*(.*?)\s*```",
        stripped,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if fenced:
        try:
            payload = json.loads(fenced.group(1))
            if isinstance(payload, dict):
                return payload, "code_fence"
        except json.JSONDecodeError:
            pass

    object_start = stripped.find("{")
    if object_start >= 0:
        try:
            payload, _ = json.JSONDecoder().raw_decode(stripped[object_start:])
            if isinstance(payload, dict):
                return payload, "embedded_json"
        except json.JSONDecodeError:
            pass
    return None, "unparseable"


def normalize_evidence_targets(
    values: Iterable[Any], *, limit: int = 6
) -> list[str]:
    """Normalize concise Grounding DINO target phrases."""
    targets = []
    seen = set()
    for raw_value in values:
        value = " ".join(str(raw_value).strip().lower().split())
        value = value.strip(" \t\r\n.,;:")
        if not value or len(value) > 80 or value in seen:
            continue
        seen.add(value)
        targets.append(value)
        if len(targets) >= limit:
            break
    return targets


def parse_manual_targets(value: str, *, limit: int = 6) -> list[str]:
    """Parse comma, newline, period, or semicolon separated target phrases."""
    return normalize_evidence_targets(
        re.split(r"[,;\n.]+", str(value)), limit=limit
    )


def parse_grounded_vlm_answer(
    raw_answer: str, *, target_limit: int = 6
) -> dict[str, Any]:
    """Parse the demo JSON response while preserving a safe text fallback."""
    payload, parse_source = _recover_json_object(raw_answer)
    if payload is None:
        return {
            "answer": raw_answer.strip(),
            "evidence_targets": [],
            "parse_source": parse_source,
            "schema_valid": False,
        }
    answer = payload.get("answer")
    target_values = payload.get("evidence_targets")
    valid_targets = isinstance(target_values, list)
    targets = normalize_evidence_targets(
        target_values if valid_targets else [],
        limit=target_limit,
    )
    schema_valid = isinstance(answer, str) and valid_targets
    return {
        "answer": answer.strip() if isinstance(answer, str) else raw_answer.strip(),
        "evidence_targets": targets,
        "parse_source": parse_source,
        "schema_valid": schema_valid,
    }


def grounding_annotation_rows(
    annotations: Iterable[Mapping[str, Any]],
) -> list[list[Any]]:
    """Format Grounded-SAM-2 annotations for a compact UI table."""
    rows = []
    for item in annotations:
        box = [round(float(value), 1) for value in item.get("bbox", [])]
        rows.append(
            [
                str(item.get("class_name", "unknown")),
                round(float(item.get("score", 0.0)), 4),
                (
                    round(float(item["mask_score"]), 4)
                    if item.get("mask_score") is not None
                    else None
                ),
                box,
                int(item.get("mask_area", 0)),
            ]
        )
    return rows


@dataclass(frozen=True)
class DemoPaths:
    project_root: Path
    config_path: Path

    @classmethod
    def resolve(
        cls, project_root: str | Path, config_path: str | Path
    ) -> "DemoPaths":
        root = Path(project_root).resolve()
        config = Path(config_path)
        if not config.is_absolute():
            config = root / config
        return cls(project_root=root, config_path=config)


class DemoRuntime:
    """Lazy, serialized Qwen3-VL and Grounded-SAM-2 demo runtime."""

    def __init__(
        self,
        project_root: str | Path,
        config_path: str | Path = "configs/demo.yaml",
        *,
        inference_enabled: bool = True,
    ) -> None:
        self.paths = DemoPaths.resolve(project_root, config_path)
        self.config = yaml.safe_load(
            self.paths.config_path.read_text(encoding="utf-8")
        )
        self.inference_enabled = inference_enabled
        self._vlm = None
        self._grounder = None
        self._lock = threading.Lock()

    def _project_path(self, value: str | Path) -> Path:
        path = Path(value)
        return path if path.is_absolute() else self.paths.project_root / path

    def _load_yaml(self, value: str | Path) -> dict[str, Any]:
        return yaml.safe_load(
            self._project_path(value).read_text(encoding="utf-8")
        )

    def _get_vlm(self):
        if self._vlm is not None:
            return self._vlm
        from grounded_visual_assistant.vlm_baseline import (
            VlmBaseline,
            VlmBaselineConfig,
        )

        demo_vlm = self.config["vlm"]
        source = self._load_yaml(demo_vlm["config"])["model"]
        self._vlm = VlmBaseline(
            VlmBaselineConfig(
                model_id=source["model_id"],
                torch_dtype=source.get("torch_dtype", "auto"),
                device_map=source.get("device_map", "auto"),
                max_new_tokens=int(
                    demo_vlm.get(
                        "max_new_tokens",
                        source.get("max_new_tokens", 256),
                    )
                ),
                do_sample=False,
                local_files_only=bool(source.get("local_files_only", True)),
            )
        )
        return self._vlm

    def _get_grounder(self):
        if self._grounder is not None:
            return self._grounder
        from grounded_visual_assistant.grounded_sam2 import (
            GroundedSam2,
            GroundedSam2Config,
        )

        demo_grounding = self.config["grounding"]
        source = self._load_yaml(demo_grounding["config"])
        grounding = source["grounding"]
        sam2 = source["sam2"]
        runtime = source["runtime"]
        self._grounder = GroundedSam2(
            GroundedSam2Config(
                grounding_model_id=grounding["model_id"],
                sam2_checkpoint=sam2["checkpoint"],
                sam2_model_config=sam2["model_config"],
                box_threshold=float(
                    demo_grounding.get(
                        "box_threshold",
                        grounding.get("box_threshold", 0.4),
                    )
                ),
                text_threshold=float(
                    demo_grounding.get(
                        "text_threshold",
                        grounding.get("text_threshold", 0.3),
                    )
                ),
                nms_iou_threshold=demo_grounding.get(
                    "nms_iou_threshold",
                    grounding.get("nms_iou_threshold"),
                ),
                device=runtime.get("device", "cuda"),
                dtype=runtime.get("dtype", "float16"),
                local_files_only=bool(
                    grounding.get("local_files_only", True)
                ),
            )
        )
        return self._grounder

    def run(
        self,
        image_path: str | Path,
        question: str,
        evidence_mode: str,
        manual_targets: str = "",
        *,
        system_prompt: str | None = None,
        evidence_target_limit: int = 6,
    ) -> dict[str, Any]:
        """Run one serialized live answer and optional grounding request."""
        if not self.inference_enabled:
            raise RuntimeError("Live inference is disabled for this demo process.")
        image = Path(image_path) if image_path else None
        if image is None or not image.is_file():
            raise ValueError("Select a valid input image.")
        question = " ".join(str(question).strip().split())
        if not question:
            raise ValueError("Enter an image question.")
        if evidence_mode not in {EVIDENCE_MODE, ANSWER_ONLY_MODE}:
            raise ValueError(f"Unsupported evidence mode: {evidence_mode}")

        with self._lock:
            vlm = self._get_vlm()
            if evidence_mode == ANSWER_ONLY_MODE:
                vlm_result = vlm.answer(image, question)
                parsed = {
                    "answer": vlm_result["answer"],
                    "evidence_targets": [],
                    "parse_source": "answer_only",
                    "schema_valid": True,
                }
            else:
                vlm_result = vlm.answer(
                    image,
                    question,
                    system_prompt=system_prompt or GROUNDED_DEMO_SYSTEM_PROMPT,
                )
                parsed = parse_grounded_vlm_answer(
                    vlm_result["answer"],
                    target_limit=evidence_target_limit,
                )

            manual = parse_manual_targets(manual_targets)
            targets = manual or parsed["evidence_targets"]
            result: dict[str, Any] = {
                "answer": parsed["answer"],
                "vlm_raw_answer": vlm_result["answer"],
                "targets": targets,
                "target_source": "manual" if manual else "vlm",
                "gallery": [],
                "annotations": [],
                "raw_annotations": [],
                "diagnostics": {
                    "status": "answered",
                    "vlm_parse_source": parsed["parse_source"],
                    "vlm_schema_valid": parsed["schema_valid"],
                    "vlm_latency_seconds": vlm_result.get(
                        "end_to_end_latency_seconds"
                    ),
                    "vlm_generated_tokens": vlm_result.get("generated_tokens"),
                    "vlm_model": vlm_result.get("model"),
                },
            }
            if evidence_mode == ANSWER_ONLY_MODE:
                return result
            if not targets:
                result["diagnostics"]["status"] = "answered_without_targets"
                return result

            output_root = self._project_path(self.config["runtime"]["output_dir"])
            output_dir = output_root / uuid.uuid4().hex
            prompt = ". ".join(targets) + "."
            grounded = self._get_grounder().predict(
                image, prompt, output_dir=output_dir
            )
            mask_path = output_dir / "grounded_sam2_masks.jpg"
            boxes_path = output_dir / "grounding_boxes.jpg"
            result["gallery"] = [
                (str(mask_path), "Segmentation masks"),
                (str(boxes_path), "Grounding boxes"),
            ]
            result["annotations"] = grounding_annotation_rows(
                grounded["annotations"]
            )
            result["raw_annotations"] = grounded["annotations"]
            result["diagnostics"].update(
                {
                    "status": "grounded",
                    "grounding_prompt": grounded["text_prompt"],
                    "detections": len(grounded["annotations"]),
                    "grounding_latency_seconds": grounded["latency_seconds"],
                    "grounding_models": grounded["models"],
                    "thresholds": grounded["thresholds"],
                    "grounding_postprocessing": grounded.get(
                        "postprocessing", {}
                    ),
                    "cuda_peak_memory_allocated_gb": grounded.get(
                        "cuda_peak_memory_allocated_gb"
                    ),
                }
            )
            return result


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


class FrozenBenchmarkStore:
    """Read-only index over the finalized Hard-Test400 result."""

    def __init__(self, project_root: str | Path) -> None:
        self.project_root = Path(project_root).resolve()
        run = (
            self.project_root
            / "outputs/cross_dataset_hard_v1/vlm/"
            "hard-test400__qwen3-vl-8b-instruct__source-aware-policy-v1"
        )
        dataset_path = (
            self.project_root
            / "data/cross_dataset_hard_v1/questions_locked_test_v1/"
            "test_questions.jsonl"
        )
        analysis_path = run / "final_report/per_sample_analysis.jsonl"
        predictions_path = run / "predictions.jsonl"
        dataset = _read_jsonl(dataset_path)
        predictions = _read_jsonl(predictions_path)
        analyses = _read_jsonl(analysis_path)
        prediction_by_id = {str(item["id"]): item for item in predictions}
        analysis_by_id = {str(item["id"]): item for item in analyses}
        if not (
            len(dataset)
            == len(prediction_by_id)
            == len(analysis_by_id)
            == 400
        ):
            raise RuntimeError("Frozen benchmark explorer requires Test400 artifacts.")
        self.records = []
        for sample in dataset:
            question_id = str(sample["id"])
            if question_id not in prediction_by_id or question_id not in analysis_by_id:
                raise RuntimeError(f"Missing frozen result for {question_id}.")
            self.records.append(
                {
                    **sample,
                    "saved_prediction": prediction_by_id[question_id],
                    "analysis": analysis_by_id[question_id],
                }
            )
        self.by_id = {str(item["id"]): item for item in self.records}
        self.overlay_dir = (
            self.project_root / "outputs/demo/benchmark_overlays"
        )

    @staticmethod
    def sources() -> list[str]:
        return [
            ALL_SOURCES,
            "open_images_v7_validation",
            "visual_genome_v1_4",
        ]

    @staticmethod
    def tasks() -> list[str]:
        return [
            ALL_TASKS,
            "object_listing",
            "object_existence",
            "spatial_relation",
        ]

    @staticmethod
    def outcomes() -> list[str]:
        return [ALL_OUTCOMES, "Correct", "Incorrect"]

    def filter_ids(
        self,
        source: str = ALL_SOURCES,
        task: str = ALL_TASKS,
        outcome: str = ALL_OUTCOMES,
    ) -> list[str]:
        records = self.records
        if source != ALL_SOURCES:
            records = [item for item in records if item["source"] == source]
        if task != ALL_TASKS:
            records = [item for item in records if item["task_type"] == task]
        if outcome != ALL_OUTCOMES:
            expected = outcome == "Correct"
            records = [
                item
                for item in records
                if bool(item["analysis"]["is_correct"]) == expected
            ]
        return [
            str(item["id"])
            for item in sorted(
                records,
                key=lambda item: (
                    -int(item["analysis"]["severity"]),
                    str(item["id"]),
                ),
            )
        ]

    def choices(
        self,
        source: str = ALL_SOURCES,
        task: str = ALL_TASKS,
        outcome: str = ALL_OUTCOMES,
    ) -> list[tuple[str, str]]:
        choices = []
        for question_id in self.filter_ids(source, task, outcome):
            item = self.by_id[question_id]
            source_short = "OI" if str(item["source"]).startswith("open_") else "VG"
            task_short = {
                "object_listing": "listing",
                "object_existence": "existence",
                "spatial_relation": "relation",
            }[item["task_type"]]
            result = "correct" if item["analysis"]["is_correct"] else "incorrect"
            choices.append(
                (
                    f"{source_short} | {task_short} | {result} | {question_id}",
                    question_id,
                )
            )
        return choices

    def _image_path(self, value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else self.project_root / path

    def _evidence_overlay(self, record: Mapping[str, Any]) -> Path:
        evidence = list(record.get("evidence_boxes") or [])
        image_path = self._image_path(str(record["image"]))
        if not evidence:
            return image_path
        output_path = self.overlay_dir / f"{record['id']}.jpg"
        if output_path.is_file():
            return output_path

        from PIL import Image, ImageDraw

        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        palette = (
            "#2563eb",
            "#0f766e",
            "#b45309",
            "#be123c",
            "#6d28d9",
        )
        for index, item in enumerate(evidence):
            if len(item.get("bbox_xywh") or []) == 4:
                x, y, width, height = [
                    float(value) for value in item["bbox_xywh"]
                ]
                box = (x, y, x + width, y + height)
            else:
                x1, y1, x2, y2 = [
                    float(value)
                    for value in item["bbox_xyxy_normalized"]
                ]
                box = (
                    x1 * image.width,
                    y1 * image.height,
                    x2 * image.width,
                    y2 * image.height,
                )
            color = palette[index % len(palette)]
            draw.rectangle(box, outline=color, width=4)
            label = str(item.get("category", "evidence"))
            text_box = draw.textbbox((box[0], box[1]), label)
            draw.rectangle(
                (
                    text_box[0] - 3,
                    text_box[1] - 2,
                    text_box[2] + 3,
                    text_box[3] + 2,
                ),
                fill=color,
            )
            draw.text((box[0], box[1]), label, fill="white")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path, quality=92)
        return output_path

    def sample_view(
        self, question_id: str
    ) -> tuple[str, str, str, str, list[list[Any]], dict[str, Any]]:
        if question_id not in self.by_id:
            raise ValueError("Select a benchmark sample.")
        item = self.by_id[question_id]
        prediction = item["saved_prediction"]
        analysis = item["analysis"]
        evidence_rows = []
        for evidence in item.get("evidence_boxes") or []:
            evidence_rows.append(
                [
                    evidence.get("category"),
                    evidence.get("annotation_id"),
                    evidence.get("bbox_xywh"),
                ]
            )
        details = {
            "id": item["id"],
            "source": item["source"],
            "task_type": item["task_type"],
            "score": analysis["score"],
            "is_correct": analysis["is_correct"],
            "flags": analysis["flags"],
            "prompt_version": (item.get("metadata") or {}).get(
                "prompt_version", "v1"
            ),
            "generated_tokens": prediction.get("generated_tokens"),
        }
        return (
            str(self._evidence_overlay(item)),
            str(item["question"]),
            str(item["gt_answer"]),
            str(prediction.get("prediction", "")),
            evidence_rows,
            details,
        )


def load_demo_metrics(project_root: str | Path) -> dict[str, Any]:
    """Load the final live-pipeline result and supporting frozen metrics."""
    root = Path(project_root).resolve()
    hard_run = (
        root
        / "outputs/cross_dataset_hard_v1/vlm/"
        "hard-test400__qwen3-vl-8b-instruct__source-aware-policy-v1/"
        "final_report"
    )
    live_run = (
        root / "outputs/eval_live_pipeline_v0" / LIVE_TEST_RUN_NAME
    )
    live_report = live_run / "final_report"
    grounding_path = (
        root
        / "outputs/eval_grounding_v0/"
        "test__structured-vlm-prompt__box-0.30__text-0.30__nms-none/"
        "coco_eval/test/coco_metrics.json"
    )
    answering_path = (
        root
        / "outputs/eval_answering_v0/"
        "test__locked-task-aware__box-0.30__text-0.30/metrics.json"
    )
    verifier_run = (
        root
        / "outputs/eval_verifier_final_v1"
        / VERIFIER_FINAL_RUN_NAME
    )
    verifier_policy = json.loads(
        (verifier_run / "final_policy.json").read_text(encoding="utf-8")
    )
    verifier_analysis = json.loads(
        (verifier_run / "failure_analysis.json").read_text(
            encoding="utf-8"
        )
    )
    return {
        "hard": json.loads(
            (hard_run / "summary.json").read_text(encoding="utf-8")
        ),
        "live": json.loads(
            (live_report / "summary.json").read_text(encoding="utf-8")
        ),
        "grounding": json.loads(grounding_path.read_text(encoding="utf-8")),
        "answering": json.loads(answering_path.read_text(encoding="utf-8")),
        "generalization": list(
            csv_dict_rows(live_report / "generalization.csv")
        ),
        "hard_generalization": list(
            csv_dict_rows(hard_run / "generalization.csv")
        ),
        "relation_confusion": list(
            csv_dict_rows(live_report / "relation_confusion.csv")
        ),
        "verifier": {
            "policy": verifier_policy,
            "analysis": verifier_analysis,
            "variants": list(
                csv_dict_rows(verifier_run / "variant_summary.csv")
            ),
            "cases": list(
                csv_dict_rows(verifier_run / "failure_cases.csv")
            ),
            "report_files": [
                str(verifier_run / "report.md"),
                str(verifier_run / "final_policy.json"),
                str(verifier_run / "variant_summary.csv"),
                str(verifier_run / "failure_cases.csv"),
                str(verifier_run / "artifact_manifest.json"),
            ],
        },
        "report_files": [
            str(live_report / "report.md"),
            str(live_report / "summary.json"),
            str(live_report / "generalization.csv"),
            str(live_report / "relation_confusion.csv"),
            str(live_report / "manifest.json"),
        ],
        "evidence_gallery": discover_evidence_gallery(root, live_run),
    }


def csv_dict_rows(path: Path) -> Iterable[dict[str, Any]]:
    import csv

    with path.open("r", encoding="utf-8", newline="") as handle:
        yield from csv.DictReader(handle)


def generalization_table_rows(
    records: Iterable[Mapping[str, Any]],
) -> list[list[Any]]:
    """Normalize old scoped and final flat generalization reports for Gradio."""
    return [
        [
            item.get("scope") or "Live Test240",
            item["metric"],
            item["dev"],
            item["test"],
            item["delta_test_minus_dev"],
        ]
        for item in records
    ]


def verifier_variant_table_rows(
    records: Iterable[Mapping[str, Any]],
) -> list[list[Any]]:
    """Convert the frozen verifier comparison into Gradio table rows."""
    return [
        [
            item["label"],
            float(item["accuracy"]),
            float(item["f1"]),
            int(item["beneficial"]),
            int(item["harmful"]),
            int(item["net_correct"]),
            int(item["model_reviews"]),
            float(item["incremental_latency_seconds"]),
            item["decision"],
        ]
        for item in records
    ]


def verifier_failure_table_rows(
    records: Iterable[Mapping[str, Any]],
) -> list[list[Any]]:
    """Convert final verifier cases into compact diagnostic rows."""
    return [
        [
            item["object"],
            item["scope"],
            item["gt_answer"],
            item["baseline_prediction"],
            item["v2_prediction"],
            item.get("v3_selected_label") or "-",
            item["final_frozen_prediction"],
            item["taxonomy"],
        ]
        for item in records
    ]


def render_verifier_markdown(metrics: Mapping[str, Any]) -> str:
    """Render the frozen answer-rewrite decision for the dashboard."""
    verifier = metrics["verifier"]
    policy = verifier["policy"]
    summary = verifier["analysis"]["summary"]
    baseline = policy["baseline_metrics"]
    return "\n".join(
        [
            "## Evidence Verifier Dev110 Final Decision",
            "",
            (
                "**Decision:** retain the frozen Qwen baseline | "
                "**Answer rewriting:** disabled | "
                "**Grounded evidence:** localization and audit only"
            ),
            "",
            "| Frozen policy | Result |",
            "|---|---:|",
            f"| Baseline accuracy | {baseline['accuracy']:.6f} |",
            f"| Baseline F1 | {baseline['f1']:.6f} |",
            f"| Baseline errors | {summary['baseline_errors']} |",
            f"| Traced failure/risk cases | {len(verifier['cases'])} |",
            "| Eligible verifier variants | 0 |",
            "| Held-out verifier run permitted | no |",
            "",
            (
                "All V1/V2/V3 answer-rewrite variants failed the frozen "
                "Dev gates: strictly higher accuracy, non-decreasing F1, "
                "and positive net corrections."
            ),
        ]
    )


def discover_evidence_gallery(
    project_root: Path, live_run: Path | None = None
) -> list[tuple[str, str]]:
    """Select deterministic success and failure examples from frozen results."""
    images: list[tuple[str, str]] = []
    if live_run is not None:
        predictions_path = live_run / "predictions.jsonl"
        if predictions_path.is_file():
            predictions = _read_jsonl(predictions_path)
            for task in (
                "object_listing",
                "object_existence",
                "spatial_relation",
            ):
                for complete in (True, False):
                    for item in predictions:
                        artifacts = (
                            item.get("grounding") or {}
                        ).get("artifacts") or []
                        if (
                            item.get("task_type") != task
                            or bool(
                                item.get("end_to_end_complete_success")
                            )
                            is not complete
                            or not artifacts
                        ):
                            continue
                        path = Path(str(artifacts[0][0]))
                        if not path.is_absolute():
                            path = project_root / path
                        if not path.is_file():
                            continue
                        outcome = (
                            "complete success"
                            if complete
                            else "incomplete evidence"
                        )
                        images.append(
                            (
                                str(path),
                                f"{task} | {outcome} | {item['id']}",
                            )
                        )
                        break
            if images:
                return images

    roots = [
        project_root
        / "outputs/eval_answering_v0/"
        "test__locked-task-aware__box-0.30__text-0.30/visualizations",
        project_root
        / "outputs/eval_grounding_v0/"
        "test__structured-vlm-prompt__box-0.30__text-0.30__nms-none/"
        "visualizations",
    ]
    for root in roots:
        for path in sorted(root.rglob("grounded_sam2_masks.jpg")):
            images.append((str(path), path.parent.name))
            if len(images) >= 6:
                return images
    return images


def render_metrics_markdown(metrics: Mapping[str, Any]) -> str:
    """Render the fixed evaluation dashboard tables."""
    live_summary = metrics["live"]
    live = live_summary["test_result"]
    integrity = live_summary["integrity"]
    hard = metrics["hard"]["test_result"]
    grounding = metrics["grounding"]
    answering = metrics["answering"]
    bbox = grounding["bbox"]["summary"]
    segm = grounding["segmentation"]["summary"]
    selective = answering["selective_answers"]
    return "\n".join(
        [
            "## Final Held-Out Live-Pipeline Test240",
            "",
            (
                "**Status:** finalized | **Coverage:** "
                f"{integrity['coverage']}/240 | **Errors:** "
                f"{integrity['prediction_errors']} | **Metrics replayed:** "
                f"{integrity['saved_metrics_replayed']}"
            ),
            "",
            "| Outcome | Metric | Result |",
            "|---|---|---:|",
            (
                "| Answer quality | Overall exact accuracy | "
                f"{live['overall']['exact_accuracy']:.4f} |"
            ),
            (
                "| Answer quality | Listing macro F1 | "
                f"{live['tasks']['object_listing']['macro_f1']:.4f} |"
            ),
            (
                "| Answer quality | Existence accuracy | "
                f"{live['tasks']['object_existence']['exact_accuracy']:.4f} |"
            ),
            (
                "| Answer quality | Relation balanced accuracy | "
                f"{live['tasks']['spatial_relation']['balanced_accuracy']:.4f} |"
            ),
            (
                "| Structured output | Schema valid rate | "
                f"{live['structured_targets']['schema_valid_rate']:.4f} |"
            ),
            (
                "| Evidence targets | Target micro F1 | "
                f"{live['structured_targets']['micro_f1']:.4f} |"
            ),
            (
                "| Visual evidence | Box / Mask IoU50 micro F1 | "
                f"{live['box_micro_f1']:.4f} / "
                f"{live['mask_micro_f1']:.4f} |"
            ),
            (
                "| End-to-end | Any / complete evidence success | "
                f"{live['end_to_end']['answer_and_any_evidence_success_rate']:.4f} / "
                f"{live['end_to_end']['answer_and_complete_evidence_success_rate']:.4f} |"
            ),
            (
                "| Negative evidence | Correct empty-target rate | "
                f"{live['negative_evidence_behavior']['correct_empty_rate']:.4f} |"
            ),
            (
                "| Runtime | Mean latency / peak CUDA memory | "
                f"{live['mean_latency_seconds']:.3f} s / "
                f"{live['peak_cuda_memory_gb']:.2f} GB |"
            ),
            "",
            "### Task Breakdown",
            "",
            "| Task | Samples | Primary metric | Result |",
            "|---|---:|---|---:|",
            (
                "| Object listing | "
                f"{live['tasks']['object_listing']['count']} | Macro F1 | "
                f"{live['tasks']['object_listing']['macro_f1']:.4f} |"
            ),
            (
                "| Object existence | "
                f"{live['tasks']['object_existence']['count']} | Accuracy | "
                f"{live['tasks']['object_existence']['exact_accuracy']:.4f} |"
            ),
            (
                "| Spatial relation | "
                f"{live['tasks']['spatial_relation']['count']} | "
                "Balanced accuracy | "
                f"{live['tasks']['spatial_relation']['balanced_accuracy']:.4f} |"
            ),
            "",
            (
                "**Locked-test limitation:** "
                f"{live_summary['decision']['reported_limitation']}."
            ),
            "",
            "### Supporting Frozen Benchmarks",
            "",
            "| Benchmark | Metric | Result |",
            "|---|---|---:|",
            (
                "| Hard-Test400 | Existence accuracy | "
                f"{hard['tasks']['object_existence']['exact_accuracy']:.4f} |"
            ),
            (
                "| Hard-Test400 | Listing macro F1 | "
                f"{hard['tasks']['object_listing']['macro_f1']:.4f} |"
            ),
            (
                "| Hard-Test400 | Relation balanced accuracy | "
                f"{hard['tasks']['spatial_relation']['balanced_accuracy']:.4f} |"
            ),
            f"| COCO Test80 | BBox AP / AP50 | {bbox['ap']:.4f} / {bbox['ap50']:.4f} |",
            f"| COCO Test80 | Mask AP / AP50 | {segm['ap']:.4f} / {segm['ap50']:.4f} |",
            (
                "| Grounded answering Test240 | Selective accuracy / coverage | "
                f"{selective['exact_accuracy']:.4f} / {selective['coverage']:.4f} |"
            ),
            "",
            "### Earlier Grounded Answering Study",
            "",
            "| Task | Forced accuracy | Selective accuracy | Coverage |",
            "|---|---:|---:|---:|",
            *[
                (
                    f"| {task} | "
                    f"{answering['closed_set_answers']['tasks'][task]['exact_accuracy']:.4f} | "
                    f"{selective['tasks'][task]['exact_accuracy']:.4f} | "
                    f"{selective['tasks'][task]['coverage']:.4f} |"
                )
                for task in (
                    "object_existence",
                    "object_listing",
                    "spatial_relation",
                )
            ],
        ]
    )
