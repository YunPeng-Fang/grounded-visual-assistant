"""Official-compatible POPE answer parsing, selection, and metrics."""

from __future__ import annotations

import hashlib
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping

from .evaluation import parse_yes_no
from .pope_dataset import POPE_STRATEGIES


POPE_PROTOCOL = "official_coco_pope_qwen_baseline_v1"
POPE_SYSTEM_PROMPT = (
    "Answer the image question using exactly one word: Yes or No. "
    "Do not add explanations or punctuation."
)


def official_parse_answer(value: str) -> str:
    """Reproduce the yes/no conversion used by POPE's official evaluator."""
    first_sentence = str(value).split(".", maxsplit=1)[0]
    words = first_sentence.replace(",", "").split(" ")
    return "no" if any(word in {"No", "no", "not"} for word in words) else "yes"


def evaluate_answer(answer: str, target: str) -> dict[str, Any]:
    label = str(target).strip().lower()
    if label not in {"yes", "no"}:
        raise ValueError(f"POPE target must be yes/no, found {target!r}.")
    official = official_parse_answer(answer)
    strict = parse_yes_no(answer)
    return {
        "official_prediction": official,
        "strict_prediction": strict,
        "strict_parse_valid": strict is not None,
        "parsed_target": label,
        "is_correct": official == label,
    }


def select_records(
    records: Iterable[Mapping[str, Any]],
    *,
    strategy: str = "all",
    samples_per_strategy: int | None = None,
) -> list[dict[str, Any]]:
    if strategy not in {"all", *POPE_STRATEGIES}:
        raise ValueError(f"Unsupported POPE strategy: {strategy}")
    if samples_per_strategy is not None and samples_per_strategy <= 0:
        raise ValueError("samples_per_strategy must be positive.")
    requested = (
        POPE_STRATEGIES if strategy == "all" else (strategy,)
    )
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for raw_item in records:
        item = dict(raw_item)
        item_strategy = str(item.get("strategy", ""))
        if item_strategy in requested:
            grouped[item_strategy].append(item)
    missing = [item for item in requested if not grouped[item]]
    if missing:
        raise ValueError(f"POPE dataset is missing strategies: {missing}")

    selected = []
    for item_strategy in requested:
        strategy_records = grouped[item_strategy]
        if samples_per_strategy is not None:
            strategy_records = strategy_records[:samples_per_strategy]
        selected.extend(strategy_records)
    ids = [str(item["id"]) for item in selected]
    if len(ids) != len(set(ids)):
        raise ValueError("Selected POPE records contain duplicate IDs.")
    return selected


def selected_ids_sha256(records: Iterable[Mapping[str, Any]]) -> str:
    payload = "\n".join(str(item["id"]) for item in records) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def binary_metrics(predictions: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    records = list(predictions)
    tp = fp = tn = fn = 0
    strict_valid = 0
    for item in records:
        evaluation = item["evaluation"]
        prediction = str(evaluation["official_prediction"])
        target = str(evaluation["parsed_target"])
        strict_valid += int(bool(evaluation["strict_parse_valid"]))
        if prediction == "yes" and target == "yes":
            tp += 1
        elif prediction == "yes" and target == "no":
            fp += 1
        elif prediction == "no" and target == "no":
            tn += 1
        elif prediction == "no" and target == "yes":
            fn += 1
        else:
            raise ValueError(
                f"Unexpected POPE prediction/target: {prediction}/{target}"
            )
    count = len(records)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "count": count,
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "accuracy": round((tp + tn) / count, 6) if count else 0.0,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "yes_ratio": round((tp + fp) / count, 6) if count else 0.0,
        "strict_parse_valid_rate": (
            round(strict_valid / count, 6) if count else 0.0
        ),
    }


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return round(statistics.fmean(values), 6) if values else 0.0


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return round(ordered[lower] * (1 - weight) + ordered[upper] * weight, 6)


def aggregate_metrics(
    predictions: list[dict[str, Any]],
    *,
    expected_samples: int,
    error_attempts: int = 0,
    status: str = "completed",
) -> dict[str, Any]:
    strategies: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in predictions:
        strategies[str(item["strategy"])].append(item)
    latencies = [float(item["latency_seconds"]) for item in predictions]
    total_latency = sum(latencies)
    generated_tokens = [
        int(item["generated_tokens"])
        for item in predictions
        if item.get("generated_tokens") is not None
    ]
    token_limit_hits = sum(
        bool(item.get("hit_max_new_tokens")) for item in predictions
    )
    completed = len(predictions)
    result: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": POPE_PROTOCOL,
        "status": status,
        "coverage": {
            "expected": expected_samples,
            "completed": completed,
            "remaining": max(expected_samples - completed, 0),
            "completion_rate": (
                round(completed / expected_samples, 6)
                if expected_samples
                else 0.0
            ),
            "error_attempts": error_attempts,
        },
        "overall": binary_metrics(predictions),
        "strategies": {
            name: binary_metrics(items)
            for name, items in sorted(strategies.items())
        },
        "generation": {
            "system_prompt": POPE_SYSTEM_PROMPT,
            "mean_generated_tokens": _mean(generated_tokens),
            "token_limit_hits": token_limit_hits,
            "token_limit_hit_rate": (
                round(token_limit_hits / completed, 6)
                if completed
                else 0.0
            ),
        },
        "latency_seconds": {
            "total": round(total_latency, 6),
            "mean": _mean(latencies),
            "median": (
                round(statistics.median(latencies), 6)
                if latencies
                else 0.0
            ),
            "p95": _percentile(latencies, 0.95),
            "throughput_samples_per_second": (
                round(completed / total_latency, 6)
                if total_latency
                else 0.0
            ),
        },
    }
    peak_memory = [
        float(item["cuda_peak_memory_allocated_gb"])
        for item in predictions
        if item.get("cuda_peak_memory_allocated_gb") is not None
    ]
    reserved_memory = [
        float(item["cuda_memory_reserved_gb"])
        for item in predictions
        if item.get("cuda_memory_reserved_gb") is not None
    ]
    if peak_memory or reserved_memory:
        result["cuda_memory_gb"] = {
            "peak_allocated_mean": _mean(peak_memory),
            "peak_allocated_max": (
                round(max(peak_memory), 6) if peak_memory else 0.0
            ),
            "reserved_mean": _mean(reserved_memory),
            "reserved_max": (
                round(max(reserved_memory), 6)
                if reserved_memory
                else 0.0
            ),
        }
    return result

