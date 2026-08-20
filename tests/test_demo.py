from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from grounded_visual_assistant.demo import (
    ALL_OUTCOMES,
    ALL_SOURCES,
    ALL_TASKS,
    ANSWER_ONLY_MODE,
    EVIDENCE_MODE,
    DemoRuntime,
    FrozenBenchmarkStore,
    generalization_table_rows,
    grounding_annotation_rows,
    load_demo_metrics,
    parse_grounded_vlm_answer,
    parse_manual_targets,
    render_metrics_markdown,
    render_verifier_markdown,
    verifier_failure_table_rows,
    verifier_variant_table_rows,
)


class DemoTest(unittest.TestCase):
    def test_parses_direct_and_fenced_grounded_answers(self) -> None:
        direct = parse_grounded_vlm_answer(
            '{"answer":"A person holds an umbrella.",'
            '"evidence_targets":["Person","umbrella","person"]}'
        )
        self.assertTrue(direct["schema_valid"])
        self.assertEqual(direct["evidence_targets"], ["person", "umbrella"])

        fenced = parse_grounded_vlm_answer(
            '```json\n{"answer":"Yes","evidence_targets":["dog"]}\n```'
        )
        self.assertEqual(fenced["parse_source"], "code_fence")
        self.assertEqual(fenced["answer"], "Yes")

        fallback = parse_grounded_vlm_answer("A plain answer.")
        self.assertFalse(fallback["schema_valid"])
        self.assertEqual(fallback["answer"], "A plain answer.")

    def test_manual_targets_are_normalized_and_limited(self) -> None:
        targets = parse_manual_targets(
            "Person; umbrella.\nDOG, person; bicycle; car; bus"
        )
        self.assertEqual(
            targets, ["person", "umbrella", "dog", "bicycle", "car", "bus"]
        )

    def test_annotation_rows_are_compact(self) -> None:
        rows = grounding_annotation_rows(
            [
                {
                    "class_name": "person",
                    "score": 0.912345,
                    "mask_score": 0.87654,
                    "bbox": [1.01, 2.02, 10.03, 20.04],
                    "mask_area": 45,
                }
            ]
        )
        self.assertEqual(
            rows[0], ["person", 0.9123, 0.8765, [1.0, 2.0, 10.0, 20.0], 45]
        )

    def test_benchmark_filter_orders_failures_first(self) -> None:
        store = FrozenBenchmarkStore.__new__(FrozenBenchmarkStore)
        store.records = [
            {
                "id": "correct",
                "source": "open_images_v7_validation",
                "task_type": "object_listing",
                "analysis": {"is_correct": True, "severity": 0},
            },
            {
                "id": "incorrect",
                "source": "open_images_v7_validation",
                "task_type": "object_listing",
                "analysis": {"is_correct": False, "severity": 3},
            },
        ]
        self.assertEqual(
            store.filter_ids(ALL_SOURCES, ALL_TASKS, ALL_OUTCOMES),
            ["incorrect", "correct"],
        )
        self.assertEqual(
            store.filter_ids(
                "open_images_v7_validation",
                "object_listing",
                "Incorrect",
            ),
            ["incorrect"],
        )

    def test_loads_final_live_test_metrics_for_evaluation_page(self) -> None:
        metrics = load_demo_metrics(PROJECT_ROOT)
        live = metrics["live"]
        rendered = render_metrics_markdown(metrics)
        generalization = generalization_table_rows(
            metrics["generalization"]
        )

        self.assertEqual(live["status"], "finalized")
        self.assertEqual(live["integrity"]["coverage"], 240)
        self.assertAlmostEqual(
            live["test_result"]["end_to_end"][
                "answer_and_complete_evidence_success_rate"
            ],
            0.591667,
        )
        self.assertEqual(len(metrics["generalization"]), 14)
        self.assertEqual(generalization[0][0], "Live Test240")
        self.assertEqual(generalization[0][1], "overall_exact_accuracy")
        self.assertEqual(len(metrics["relation_confusion"]), 13)
        self.assertGreaterEqual(len(metrics["evidence_gallery"]), 3)
        self.assertIn("Final Held-Out Live-Pipeline Test240", rendered)
        self.assertIn("0.5917", rendered)
        self.assertTrue(
            all(Path(path).is_file() for path in metrics["report_files"])
        )

    def test_loads_final_verifier_audit_for_evaluation_page(self) -> None:
        metrics = load_demo_metrics(PROJECT_ROOT)
        verifier = metrics["verifier"]
        variants = verifier_variant_table_rows(verifier["variants"])
        cases = verifier_failure_table_rows(verifier["cases"])
        rendered = render_verifier_markdown(metrics)

        self.assertEqual(
            verifier["policy"]["decision"],
            "retain_qwen_baseline_disable_answer_rewrite",
        )
        self.assertFalse(verifier["policy"]["answer_rewrite_enabled"])
        self.assertFalse(
            verifier["policy"]["held_out_verifier_run_permitted"]
        )
        self.assertEqual(len(variants), 5)
        self.assertEqual(variants[0][0], "Frozen Qwen baseline")
        self.assertEqual(variants[0][-1], "retained")
        self.assertEqual(len(cases), 6)
        self.assertIn("grounding_recall_miss", {item[-1] for item in cases})
        self.assertIn("Answer rewriting:** disabled", rendered)
        self.assertTrue(
            all(
                Path(path).is_file()
                for path in verifier["report_files"]
            )
        )

    def test_runtime_runs_answer_only_without_grounder(self) -> None:
        class FakeVlm:
            def answer(self, image_path, question, *, system_prompt=None):
                self.call = (Path(image_path), question, system_prompt)
                return {
                    "answer": "A red cup.",
                    "end_to_end_latency_seconds": 0.1,
                    "generated_tokens": 4,
                    "model": "fake-vlm",
                }

        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "image.jpg"
            image_path.write_bytes(b"test")
            runtime = DemoRuntime(PROJECT_ROOT)
            runtime._vlm = FakeVlm()
            result = runtime.run(
                image_path,
                "What is visible?",
                ANSWER_ONLY_MODE,
            )

        self.assertEqual(result["answer"], "A red cup.")
        self.assertEqual(result["targets"], [])
        self.assertEqual(result["vlm_raw_answer"], "A red cup.")
        self.assertEqual(result["raw_annotations"], [])
        self.assertEqual(result["diagnostics"]["status"], "answered")
        self.assertIsNone(runtime._vlm.call[2])
        self.assertIsNone(runtime._grounder)

    def test_runtime_connects_structured_targets_to_grounder(self) -> None:
        class FakeVlm:
            def answer(self, image_path, question, *, system_prompt=None):
                self.system_prompt = system_prompt
                return {
                    "answer": (
                        '{"answer":"A person holds an umbrella.",'
                        '"evidence_targets":["person","umbrella"]}'
                    ),
                    "end_to_end_latency_seconds": 0.2,
                    "generated_tokens": 12,
                    "model": "fake-vlm",
                }

        class FakeGrounder:
            def predict(self, image_path, prompt, output_dir=None):
                self.prompt = prompt
                Path(output_dir).mkdir(parents=True)
                return {
                    "text_prompt": prompt,
                    "annotations": [
                        {
                            "class_name": "person",
                            "score": 0.9,
                            "mask_score": 0.8,
                            "bbox": [1, 2, 3, 4],
                            "mask_area": 10,
                        }
                    ],
                    "latency_seconds": {"total": 0.3},
                    "models": {"grounding": "fake", "sam2": "fake"},
                    "thresholds": {"box": 0.3, "text": 0.3},
                    "cuda_peak_memory_allocated_gb": 1.0,
                }

        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "image.jpg"
            image_path.write_bytes(b"test")
            runtime = DemoRuntime(PROJECT_ROOT)
            runtime.config["runtime"]["output_dir"] = str(
                Path(temp_dir) / "outputs"
            )
            runtime._vlm = FakeVlm()
            runtime._grounder = FakeGrounder()
            result = runtime.run(
                image_path,
                "What is the person holding?",
                EVIDENCE_MODE,
                system_prompt="Return grounded JSON.",
            )

        self.assertEqual(result["answer"], "A person holds an umbrella.")
        self.assertEqual(result["targets"], ["person", "umbrella"])
        self.assertEqual(runtime._grounder.prompt, "person. umbrella.")
        self.assertEqual(result["diagnostics"]["status"], "grounded")
        self.assertEqual(result["annotations"][0][0], "person")
        self.assertEqual(result["raw_annotations"][0]["class_name"], "person")
        self.assertEqual(runtime._vlm.system_prompt, "Return grounded JSON.")


if __name__ == "__main__":
    unittest.main()
