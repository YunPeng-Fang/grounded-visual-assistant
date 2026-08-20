"""Single-image VLM baseline runner."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor

try:
    from transformers import Qwen2_5_VLForConditionalGeneration
except ImportError:  # pragma: no cover - depends on installed transformers version
    Qwen2_5_VLForConditionalGeneration = None

try:
    from transformers import Qwen3VLForConditionalGeneration
except ImportError:  # pragma: no cover - depends on installed transformers version
    Qwen3VLForConditionalGeneration = None

try:
    from transformers import AutoModelForMultimodalLM
except ImportError:  # pragma: no cover - depends on installed transformers version
    AutoModelForMultimodalLM = None

from grounded_visual_assistant.prompts import build_vlm_messages


@dataclass
class VlmBaselineConfig:
    model_id: str = "Qwen/Qwen3-VL-8B-Instruct"
    torch_dtype: str = "auto"
    device_map: str = "auto"
    max_new_tokens: int = 256
    do_sample: bool = False
    local_files_only: bool = False


class VlmBaseline:
    """Thin wrapper around a Qwen-VL style image-text generation model."""

    def __init__(self, config: VlmBaselineConfig) -> None:
        self.config = config
        dtype = self._resolve_dtype(config.torch_dtype)
        model_cls = self._resolve_model_class(config.model_id)
        model_kwargs = {
            "device_map": config.device_map,
            "local_files_only": config.local_files_only,
        }
        if "Qwen3-VL" in config.model_id:
            model_kwargs["dtype"] = dtype
        else:
            model_kwargs["torch_dtype"] = dtype
        self.model = model_cls.from_pretrained(config.model_id, **model_kwargs)
        self.processor = AutoProcessor.from_pretrained(
            config.model_id,
            local_files_only=config.local_files_only,
        )

    @staticmethod
    def _resolve_model_class(model_id: str):
        if "Qwen3-VL" in model_id:
            if Qwen3VLForConditionalGeneration is not None:
                return Qwen3VLForConditionalGeneration
            if AutoModelForMultimodalLM is not None:
                return AutoModelForMultimodalLM
            raise RuntimeError(
                "Qwen3-VL requires a newer transformers build. Install the latest "
                "transformers from source, or switch to Qwen2.5-VL."
            )

        if "Qwen2.5-VL" in model_id or "Qwen2-VL" in model_id:
            if Qwen2_5_VLForConditionalGeneration is not None:
                return Qwen2_5_VLForConditionalGeneration
            if AutoModelForMultimodalLM is not None:
                return AutoModelForMultimodalLM
            raise RuntimeError(
                "Qwen2.5-VL model class is unavailable. Upgrade transformers."
            )

        if AutoModelForMultimodalLM is not None:
            return AutoModelForMultimodalLM
        raise RuntimeError(
            "No supported multimodal model class is available in transformers."
        )

    def answer(
        self,
        image_path: str | Path,
        question: str,
        *,
        system_prompt: str | None = None,
    ) -> dict[str, Any]:
        """Answer a question about one image."""
        total_start = time.perf_counter()
        image_path = str(image_path)
        messages = build_vlm_messages(
            image_path=image_path,
            question=question,
            system_prompt=system_prompt,
        )

        if "Qwen3-VL" in self.config.model_id:
            result = self._answer_qwen3(messages, image_path, question)
        else:
            result = self._answer_qwen25(messages, image_path, question)
        result["end_to_end_latency_seconds"] = round(
            time.perf_counter() - total_start, 4
        )
        return result

    def _answer_qwen3(
        self,
        messages: list[dict],
        image_path: str,
        question: str,
    ) -> dict[str, Any]:
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        start = self._start_measurement()
        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=self.config.do_sample,
            )
        latency, cuda_metrics = self._finish_measurement(start)

        generated_ids_trimmed = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        generated_tokens = int(generated_ids_trimmed[0].shape[-1])
        answer = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        return self._build_result(
            image_path,
            question,
            answer,
            latency,
            cuda_metrics,
            generated_tokens,
        )

    def _answer_qwen25(
        self,
        messages: list[dict],
        image_path: str,
        question: str,
    ) -> dict[str, Any]:
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        start = self._start_measurement()
        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=self.config.do_sample,
            )
        latency, cuda_metrics = self._finish_measurement(start)

        generated_ids_trimmed = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        generated_tokens = int(generated_ids_trimmed[0].shape[-1])
        answer = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        return self._build_result(
            image_path,
            question,
            answer,
            latency,
            cuda_metrics,
            generated_tokens,
        )

    def _start_measurement(self) -> float:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        return time.perf_counter()

    @staticmethod
    def _finish_measurement(start: float) -> tuple[float, dict[str, float]]:
        cuda_metrics: dict[str, float] = {}
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            gibibyte = 1024**3
            cuda_metrics = {
                "cuda_memory_allocated_gb": round(
                    torch.cuda.memory_allocated() / gibibyte, 4
                ),
                "cuda_peak_memory_allocated_gb": round(
                    torch.cuda.max_memory_allocated() / gibibyte, 4
                ),
                "cuda_memory_reserved_gb": round(
                    torch.cuda.memory_reserved() / gibibyte, 4
                ),
            }
        return time.perf_counter() - start, cuda_metrics

    def _build_result(
        self,
        image_path: str,
        question: str,
        answer: str,
        latency: float,
        cuda_metrics: dict[str, float],
        generated_tokens: int,
    ) -> dict[str, Any]:
        result = {
            "image": image_path,
            "question": question,
            "answer": answer,
            "model": self.config.model_id,
            "latency_seconds": round(latency, 4),
            "generated_tokens": generated_tokens,
            "max_new_tokens": self.config.max_new_tokens,
            "hit_max_new_tokens": generated_tokens >= self.config.max_new_tokens,
            "device": str(self.model.device),
            "cuda_available": torch.cuda.is_available(),
        }
        result.update(cuda_metrics)
        return result

    @staticmethod
    def _resolve_dtype(dtype: str) -> str | torch.dtype:
        if dtype == "auto":
            return "auto"
        if dtype in {"float16", "fp16"}:
            return torch.float16
        if dtype in {"bfloat16", "bf16"}:
            return torch.bfloat16
        if dtype in {"float32", "fp32"}:
            return torch.float32
        raise ValueError(f"Unsupported torch dtype: {dtype}")
