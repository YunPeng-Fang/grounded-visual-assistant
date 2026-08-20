"""Adapter for the official Grounded-SAM-2 image pipeline."""

from __future__ import annotations

import json
import re
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class GroundedSam2Config:
    """Configuration for local Grounding DINO and SAM 2.1 inference."""

    grounding_model_id: str
    sam2_checkpoint: str
    sam2_model_config: str
    box_threshold: float = 0.4
    text_threshold: float = 0.3
    nms_iou_threshold: float | None = None
    device: str = "cuda"
    dtype: str = "float16"
    local_files_only: bool = True

    def __post_init__(self) -> None:
        for name, value in (
            ("box_threshold", self.box_threshold),
            ("text_threshold", self.text_threshold),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1, got {value}.")
        if self.nms_iou_threshold is not None and not (
            0.0 <= self.nms_iou_threshold <= 1.0
        ):
            raise ValueError(
                "nms_iou_threshold must be between 0 and 1 or None, got "
                f"{self.nms_iou_threshold}."
            )
        if self.dtype not in {"float16", "bfloat16", "float32"}:
            raise ValueError(f"Unsupported dtype: {self.dtype}")


def normalize_grounding_prompt(prompt: str) -> str:
    """Match Grounding DINO's lowercase, period-separated prompt format."""
    phrases = [part.strip().lower() for part in re.split(r"[.;]+", prompt)]
    phrases = [phrase for phrase in phrases if phrase]
    if not phrases:
        raise ValueError("Grounding prompt must contain at least one phrase.")
    return ". ".join(phrases) + "."


def class_aware_nms_indices(
    boxes: np.ndarray,
    scores: np.ndarray,
    labels: list[str],
    iou_threshold: float,
) -> np.ndarray:
    """Return score-ordered indices after suppressing same-label overlaps."""
    boxes = np.asarray(boxes, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    if boxes.ndim != 2 or boxes.shape[1:] != (4,):
        raise ValueError(f"boxes must have shape [N, 4], got {boxes.shape}.")
    if len(boxes) != len(scores) or len(boxes) != len(labels):
        raise ValueError("boxes, scores, and labels must have the same length.")
    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be between 0 and 1.")
    if not len(boxes):
        return np.empty((0,), dtype=np.int64)

    x1, y1, x2, y2 = boxes.T
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = np.argsort(-scores, kind="stable")
    suppressed = np.zeros(len(boxes), dtype=bool)
    keep: list[int] = []
    normalized_labels = [label.strip().lower() for label in labels]

    for position, index in enumerate(order):
        if suppressed[index]:
            continue
        keep.append(int(index))
        remaining = order[position + 1 :]
        same_label = np.asarray(
            [
                candidate
                for candidate in remaining
                if not suppressed[candidate]
                and normalized_labels[candidate] == normalized_labels[index]
            ],
            dtype=np.int64,
        )
        if not len(same_label):
            continue
        intersection_x1 = np.maximum(x1[index], x1[same_label])
        intersection_y1 = np.maximum(y1[index], y1[same_label])
        intersection_x2 = np.minimum(x2[index], x2[same_label])
        intersection_y2 = np.minimum(y2[index], y2[same_label])
        intersection = (
            np.maximum(0.0, intersection_x2 - intersection_x1)
            * np.maximum(0.0, intersection_y2 - intersection_y1)
        )
        union = areas[index] + areas[same_label] - intersection
        iou = np.divide(
            intersection,
            union,
            out=np.zeros_like(intersection),
            where=union > 0,
        )
        suppressed[same_label[iou > iou_threshold]] = True

    return np.asarray(keep, dtype=np.int64)


def _safe_label(value: str) -> str:
    label = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
    return label or "object"


class GroundedSam2:
    """Official Grounding DINO (HF) to SAM 2.1 box-prompt pipeline."""

    def __init__(self, config: GroundedSam2Config) -> None:
        self.config = config
        checkpoint = Path(config.sam2_checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"SAM 2 checkpoint not found: {checkpoint}")

        try:
            import pycocotools.mask as mask_util
            import torch
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        except ImportError as exc:
            raise RuntimeError(
                "Grounded-SAM-2 dependencies are unavailable. Install "
                "requirements-grounded-sam2.txt and the official repository "
                "with scripts/install_grounded_sam2.sh."
            ) from exc

        self.torch = torch
        self.mask_util = mask_util
        self.device = torch.device(config.device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")
        self.amp_dtype = self._resolve_dtype(config.dtype)

        if (
            self.device.type == "cuda"
            and torch.cuda.get_device_properties(self.device).major >= 8
        ):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        sam2_model = build_sam2(
            config.sam2_model_config,
            str(checkpoint),
            device=str(self.device),
        )
        sam2_model.eval()
        self.sam2_predictor = SAM2ImagePredictor(sam2_model)

        self.processor = AutoProcessor.from_pretrained(
            config.grounding_model_id,
            local_files_only=config.local_files_only,
        )
        self.grounding_model = (
            AutoModelForZeroShotObjectDetection.from_pretrained(
                config.grounding_model_id,
                local_files_only=config.local_files_only,
            )
            .to(self.device)
            .eval()
        )

    def _resolve_dtype(self, dtype: str):
        if dtype == "float16":
            return self.torch.float16
        if dtype == "bfloat16":
            return self.torch.bfloat16
        return self.torch.float32

    def _autocast(self):
        if self.device.type == "cuda" and self.amp_dtype != self.torch.float32:
            return self.torch.autocast(device_type="cuda", dtype=self.amp_dtype)
        return nullcontext()

    def _synchronize(self) -> None:
        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)

    def _mask_to_rle(self, mask: np.ndarray) -> dict[str, Any]:
        encoded = self.mask_util.encode(
            np.array(mask[:, :, None], order="F", dtype="uint8")
        )[0]
        counts = encoded["counts"]
        if isinstance(counts, bytes):
            counts = counts.decode("utf-8")
        return {
            "size": [int(value) for value in encoded["size"]],
            "counts": counts,
        }

    def predict(
        self,
        image_path: str | Path,
        prompt: str,
        output_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        """Detect text targets, segment each box, and optionally save artifacts."""
        image_path = Path(image_path)
        if not image_path.is_file():
            raise FileNotFoundError(f"Input image not found: {image_path}")
        text_prompt = normalize_grounding_prompt(prompt)
        image = Image.open(image_path).convert("RGB")
        # Pillow may expose a read-only buffer; SAM/torchvision expects a
        # writable array when converting the image to a tensor.
        image_array = np.asarray(image).copy()

        if self.device.type == "cuda":
            self.torch.cuda.reset_peak_memory_stats(self.device)
        total_start = time.perf_counter()

        detector_inputs = self.processor(
            images=image,
            text=text_prompt,
            return_tensors="pt",
        ).to(self.device)
        self._synchronize()
        detector_start = time.perf_counter()
        with self.torch.inference_mode(), self._autocast():
            detector_outputs = self.grounding_model(**detector_inputs)
        self._synchronize()
        detector_latency = time.perf_counter() - detector_start

        detections = self.processor.post_process_grounded_object_detection(
            detector_outputs,
            detector_inputs.input_ids,
            threshold=self.config.box_threshold,
            text_threshold=self.config.text_threshold,
            target_sizes=[image.size[::-1]],
        )[0]
        boxes = detections["boxes"].detach().cpu().numpy()
        grounding_scores = detections["scores"].detach().cpu().numpy()
        raw_labels = detections.get("text_labels", detections.get("labels", []))
        if hasattr(raw_labels, "detach"):
            raw_labels = raw_labels.detach().cpu().tolist()
        labels = [str(label) for label in raw_labels]
        candidate_count = len(boxes)
        if self.config.nms_iou_threshold is not None and candidate_count:
            keep = class_aware_nms_indices(
                boxes,
                grounding_scores,
                labels,
                self.config.nms_iou_threshold,
            )
            boxes = boxes[keep]
            grounding_scores = grounding_scores[keep]
            labels = [labels[index] for index in keep]
        kept_count = len(boxes)

        sam_latency = 0.0
        masks = np.empty((0, image.height, image.width), dtype=bool)
        mask_scores = np.empty((0,), dtype=np.float32)
        if len(boxes):
            self._synchronize()
            sam_start = time.perf_counter()
            with self.torch.inference_mode(), self._autocast():
                self.sam2_predictor.set_image(image_array)
                masks, mask_scores, _ = self.sam2_predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=boxes,
                    multimask_output=False,
                )
            self._synchronize()
            sam_latency = time.perf_counter() - sam_start
            if masks.ndim == 4:
                masks = masks.squeeze(1)
            mask_scores = np.asarray(mask_scores).reshape(-1)
            masks = masks.astype(bool)

        annotations = []
        for index, (label, box, grounding_score, mask) in enumerate(
            zip(labels, boxes, grounding_scores, masks)
        ):
            mask_score = float(mask_scores[index]) if index < len(mask_scores) else None
            annotations.append(
                {
                    "class_name": label,
                    "bbox": [round(float(value), 3) for value in box],
                    "score": round(float(grounding_score), 6),
                    "mask_score": round(mask_score, 6) if mask_score is not None else None,
                    "mask_area": int(mask.sum()),
                    "segmentation": self._mask_to_rle(mask),
                }
            )

        self._synchronize()
        total_latency = time.perf_counter() - total_start
        result: dict[str, Any] = {
            "image_path": str(image_path),
            "text_prompt": text_prompt,
            "annotations": annotations,
            "box_format": "xyxy",
            "img_width": image.width,
            "img_height": image.height,
            "models": {
                "grounding": self.config.grounding_model_id,
                "sam2_checkpoint": self.config.sam2_checkpoint,
                "sam2_config": self.config.sam2_model_config,
            },
            "thresholds": {
                "box": self.config.box_threshold,
                "text": self.config.text_threshold,
                "nms_iou": self.config.nms_iou_threshold,
            },
            "postprocessing": {
                "candidate_count": candidate_count,
                "kept_count": kept_count,
                "suppressed_count": candidate_count - kept_count,
            },
            "latency_seconds": {
                "grounding": round(detector_latency, 4),
                "sam2": round(sam_latency, 4),
                "total": round(total_latency, 4),
            },
            "device": str(self.device),
        }
        if self.device.type == "cuda":
            gibibyte = 1024**3
            result["cuda_peak_memory_allocated_gb"] = round(
                self.torch.cuda.max_memory_allocated(self.device) / gibibyte, 4
            )

        if output_dir is not None:
            self._save_artifacts(
                image_array=image_array,
                masks=masks,
                result=result,
                output_dir=Path(output_dir),
            )
        return result

    def _save_artifacts(
        self,
        *,
        image_array: np.ndarray,
        masks: np.ndarray,
        result: dict[str, Any],
        output_dir: Path,
    ) -> None:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError(
                "OpenCV is required to save Grounded-SAM-2 visualizations. "
                "Install requirements.txt first."
            ) from exc

        output_dir.mkdir(parents=True, exist_ok=True)
        mask_dir = output_dir / "masks"
        mask_dir.mkdir(parents=True, exist_ok=True)

        boxes_frame = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        mask_frame = boxes_frame.copy()
        palette = (
            (44, 160, 255),
            (80, 200, 120),
            (220, 120, 80),
            (180, 90, 220),
            (70, 200, 220),
            (220, 170, 60),
        )

        for index, (annotation, mask) in enumerate(zip(result["annotations"], masks)):
            color = palette[index % len(palette)]
            x1, y1, x2, y2 = [int(round(value)) for value in annotation["bbox"]]
            label = f"{annotation['class_name']} {annotation['score']:.2f}"
            cv2.rectangle(boxes_frame, (x1, y1), (x2, y2), color, 2)
            cv2.rectangle(mask_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                boxes_frame,
                label,
                (x1, max(y1 - 8, 18)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                mask_frame,
                label,
                (x1, max(y1 - 8, 18)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )
            color_array = np.asarray(color, dtype=np.float32)
            mask_frame[mask] = (
                mask_frame[mask].astype(np.float32) * 0.55 + color_array * 0.45
            ).astype(np.uint8)
            mask_name = f"{index:03d}_{_safe_label(annotation['class_name'])}.png"
            cv2.imwrite(str(mask_dir / mask_name), mask.astype(np.uint8) * 255)

        cv2.imwrite(str(output_dir / "grounding_boxes.jpg"), boxes_frame)
        cv2.imwrite(str(output_dir / "grounded_sam2_masks.jpg"), mask_frame)
        (output_dir / "result.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
