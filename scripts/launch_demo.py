"""Launch the Grounded Visual Assistant Gradio application."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

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
    load_demo_metrics,
    render_metrics_markdown,
    render_verifier_markdown,
    verifier_failure_table_rows,
    verifier_variant_table_rows,
)


APP_CSS = """
:root {
  --surface: #ffffff;
  --canvas: #f5f7fa;
  --line: #d8dee8;
  --text: #172033;
  --muted: #5e6879;
  --blue: #2563eb;
  --green: #0f766e;
  --amber: #b45309;
}

.gradio-container {
  max-width: 1440px !important;
  min-width: 0 !important;
  width: 100% !important;
  flex-basis: 0 !important;
  margin: 0 auto !important;
  background: var(--canvas) !important;
  color: var(--text) !important;
}

.gradio-container .main,
.gradio-container .wrap,
.gradio-container main,
.gradio-container .form,
.gradio-container .block {
  min-width: 0 !important;
}

.app-header {
  padding: 18px 2px 10px 2px;
  border-bottom: 1px solid var(--line);
  margin-bottom: 10px;
}

.app-header h1 {
  font-size: 30px !important;
  line-height: 1.2 !important;
  letter-spacing: 0 !important;
  margin: 0 0 6px 0 !important;
}

.app-header p {
  color: var(--muted);
  margin: 0 !important;
}

.panel {
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: 6px;
  padding: 12px;
}

.primary-action button {
  min-height: 42px;
  font-weight: 650;
}

.status-ok {
  color: var(--green);
  font-weight: 650;
}

.status-offline {
  color: var(--amber);
  font-weight: 650;
}

.compact-table {
  min-height: 180px;
  overflow-x: auto;
}

.benchmark-image img,
.live-image img {
  object-fit: contain !important;
  max-height: 520px !important;
}

@media (max-width: 760px) {
  .gradio-container main {
    padding-left: 12px !important;
    padding-right: 12px !important;
  }
  .app-header h1 {
    font-size: 25px !important;
  }
  .app-header p {
    white-space: normal !important;
  }
  .filter-control {
    flex: 1 1 100% !important;
    width: 100% !important;
  }
  .panel {
    padding: 8px;
  }
}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch the grounded visual assistant Gradio demo."
    )
    parser.add_argument("--config", default="configs/demo.yaml")
    parser.add_argument("--server-name", default="0.0.0.0")
    parser.add_argument("--server-port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--inbrowser", action="store_true")
    parser.add_argument(
        "--results-only",
        action="store_true",
        help="Disable live model loading and expose only frozen result views.",
    )
    parser.add_argument("--auth-user", default=None)
    parser.add_argument("--auth-password", default=None)
    args = parser.parse_args()
    if (args.auth_user is None) != (args.auth_password is None):
        parser.error("--auth-user and --auth-password must be provided together.")
    if not 1 <= args.server_port <= 65535:
        parser.error("--server-port must be between 1 and 65535.")
    return args


def build_app(
    runtime: DemoRuntime,
    store: FrozenBenchmarkStore,
    metrics: dict[str, Any],
    *,
    inference_enabled: bool,
):
    try:
        import gradio as gr
    except ImportError as exc:
        raise RuntimeError(
            "Gradio is not installed. Run "
            "`python -m pip install -r requirements-demo.txt`."
        ) from exc

    initial_choices = store.choices()
    initial_id = initial_choices[0][1]
    initial_view = store.sample_view(initial_id)
    status_class = "status-ok" if inference_enabled else "status-offline"
    status_text = (
        "Live inference enabled"
        if inference_enabled
        else "Frozen results mode"
    )

    def run_live(
        image_path: str | None,
        question: str,
        mode: str,
        manual_targets: str,
    ):
        try:
            result = runtime.run(
                image_path or "",
                question,
                mode,
                manual_targets,
            )
            return (
                result["answer"],
                "; ".join(result["targets"]),
                result["gallery"],
                result["annotations"],
                result["diagnostics"],
            )
        except Exception as exc:
            return (
                f"Error: {exc}",
                "",
                [],
                [],
                {"status": "error", "error": str(exc)},
            )

    def update_sample_choices(source: str, task: str, outcome: str):
        choices = store.choices(source, task, outcome)
        value = choices[0][1] if choices else None
        return gr.Dropdown(choices=choices, value=value)

    def load_sample(question_id: str | None):
        if question_id is None:
            return None, "", "", "", [], {"status": "no_match"}
        return store.sample_view(question_id)

    def clear_live():
        return None, "", EVIDENCE_MODE, "", "", "", [], [], {}

    with gr.Blocks(title="Grounded Visual Assistant") as app:
        gr.Markdown(
            (
                "# Grounded Visual Assistant\n"
                "Qwen3-VL-8B-Instruct | Grounding DINO | SAM 2.1 | "
                f"<span class='{status_class}'>{status_text}</span>"
            ),
            elem_classes="app-header",
        )
        with gr.Tabs():
            with gr.Tab("Assistant"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=5, elem_classes="panel"):
                        input_image = gr.Image(
                            label="Input image",
                            type="filepath",
                            sources=["upload", "webcam", "clipboard"],
                            height=430,
                            buttons=["fullscreen"],
                            elem_classes="live-image",
                        )
                        question = gr.Textbox(
                            label="Question",
                            lines=2,
                            placeholder="What is the person holding?",
                        )
                        mode = gr.Radio(
                            choices=[EVIDENCE_MODE, ANSWER_ONLY_MODE],
                            value=EVIDENCE_MODE,
                            label="Response mode",
                        )
                        manual_targets = gr.Textbox(
                            label="Evidence targets (optional)",
                            placeholder="person; umbrella",
                        )
                        with gr.Row():
                            run_button = gr.Button(
                                "Run",
                                variant="primary",
                                interactive=inference_enabled,
                                elem_classes="primary-action",
                            )
                            clear_button = gr.Button("Clear", variant="secondary")
                    with gr.Column(scale=7, elem_classes="panel"):
                        answer = gr.Textbox(
                            label="Answer",
                            lines=4,
                            interactive=False,
                            buttons=["copy"],
                        )
                        resolved_targets = gr.Textbox(
                            label="Resolved evidence targets",
                            interactive=False,
                            buttons=["copy"],
                        )
                        evidence_gallery = gr.Gallery(
                            label="Visual evidence",
                            columns=2,
                            rows=1,
                            height=430,
                            object_fit="contain",
                            buttons=["fullscreen", "download"],
                        )
                evidence_table = gr.Dataframe(
                    headers=[
                        "Label",
                        "Grounding score",
                        "Mask score",
                        "Box (xyxy)",
                        "Mask area",
                    ],
                    datatype=["str", "number", "number", "str", "number"],
                    value=[],
                    type="array",
                    interactive=False,
                    buttons=["copy", "fullscreen"],
                    label="Evidence instances",
                    elem_classes="compact-table",
                )
                runtime_details = gr.JSON(label="Runtime details", value={})

                run_button.click(
                    run_live,
                    inputs=[input_image, question, mode, manual_targets],
                    outputs=[
                        answer,
                        resolved_targets,
                        evidence_gallery,
                        evidence_table,
                        runtime_details,
                    ],
                    api_visibility="private",
                )
                clear_button.click(
                    clear_live,
                    outputs=[
                        input_image,
                        question,
                        mode,
                        manual_targets,
                        answer,
                        resolved_targets,
                        evidence_gallery,
                        evidence_table,
                        runtime_details,
                    ],
                    api_visibility="private",
                )

            with gr.Tab("Benchmark Explorer"):
                gr.Markdown("## Frozen Benchmark Explorer")
                with gr.Row():
                    source_filter = gr.Dropdown(
                        choices=store.sources(),
                        value=ALL_SOURCES,
                        label="Source",
                        elem_classes="filter-control",
                    )
                    task_filter = gr.Dropdown(
                        choices=store.tasks(),
                        value=ALL_TASKS,
                        label="Task",
                        elem_classes="filter-control",
                    )
                    outcome_filter = gr.Dropdown(
                        choices=store.outcomes(),
                        value=ALL_OUTCOMES,
                        label="Outcome",
                        elem_classes="filter-control",
                    )
                sample_id = gr.Dropdown(
                    choices=initial_choices,
                    value=initial_id,
                    label="Frozen Test sample",
                    filterable=True,
                )
                with gr.Row(equal_height=False):
                    with gr.Column(scale=6, elem_classes="panel"):
                        benchmark_image = gr.Image(
                            value=initial_view[0],
                            label="Annotated evidence",
                            height=520,
                            buttons=["fullscreen", "download"],
                            elem_classes="benchmark-image",
                        )
                    with gr.Column(scale=6, elem_classes="panel"):
                        benchmark_question = gr.Textbox(
                            value=initial_view[1],
                            label="Question",
                            lines=3,
                            interactive=False,
                            buttons=["copy"],
                        )
                        with gr.Row():
                            benchmark_gt = gr.Textbox(
                                value=initial_view[2],
                                label="Ground truth",
                                interactive=False,
                                buttons=["copy"],
                            )
                            benchmark_prediction = gr.Textbox(
                                value=initial_view[3],
                                label="Prediction",
                                interactive=False,
                                buttons=["copy"],
                            )
                        benchmark_boxes = gr.Dataframe(
                            value=initial_view[4],
                            headers=["Category", "Annotation ID", "Box (xywh)"],
                            datatype=["str", "str", "str"],
                            type="array",
                            interactive=False,
                            label="Frozen evidence",
                            buttons=["copy", "fullscreen"],
                        )
                        benchmark_details = gr.JSON(
                            value=initial_view[5], label="Sample diagnostics"
                        )

                filter_inputs = [
                    source_filter,
                    task_filter,
                    outcome_filter,
                ]
                sample_outputs = [
                    benchmark_image,
                    benchmark_question,
                    benchmark_gt,
                    benchmark_prediction,
                    benchmark_boxes,
                    benchmark_details,
                ]
                for component in (
                    source_filter,
                    task_filter,
                    outcome_filter,
                ):
                    component.change(
                        update_sample_choices,
                        inputs=filter_inputs,
                        outputs=sample_id,
                        api_visibility="private",
                    ).then(
                        load_sample,
                        inputs=sample_id,
                        outputs=sample_outputs,
                        api_visibility="private",
                    )
                sample_id.change(
                    load_sample,
                    inputs=sample_id,
                    outputs=sample_outputs,
                    api_visibility="private",
                )

            with gr.Tab("Evaluation"):
                with gr.Tabs():
                    with gr.Tab("Held-Out Test240"):
                        gr.Markdown(render_metrics_markdown(metrics))
                        gr.Dataframe(
                            value=generalization_table_rows(
                                metrics["generalization"]
                            ),
                            headers=[
                                "Scope",
                                "Metric",
                                "Dev",
                                "Test",
                                "Delta",
                            ],
                            datatype=[
                                "str",
                                "str",
                                "number",
                                "number",
                                "number",
                            ],
                            type="array",
                            interactive=False,
                            buttons=["copy", "fullscreen"],
                            label="Final Test240 Dev to Test generalization",
                        )
                        gr.Dataframe(
                            value=[
                                [
                                    item["target"],
                                    item["prediction"],
                                    item["count"],
                                ]
                                for item in metrics["relation_confusion"]
                            ],
                            headers=[
                                "Target relation",
                                "Prediction",
                                "Count",
                            ],
                            datatype=["str", "str", "number"],
                            type="array",
                            interactive=False,
                            buttons=["copy", "fullscreen"],
                            label=(
                                "Final Test240 spatial-relation confusion"
                            ),
                        )
                        gr.Gallery(
                            value=metrics["evidence_gallery"],
                            label="Final Test240 frozen visual evidence",
                            columns=3,
                            height=420,
                            object_fit="contain",
                            buttons=["fullscreen", "download"],
                        )
                        gr.File(
                            value=metrics["report_files"],
                            file_count="multiple",
                            interactive=False,
                            label="Final Test240 report artifacts",
                        )
                    with gr.Tab("Verifier Audit"):
                        gr.Markdown(render_verifier_markdown(metrics))
                        gr.Dataframe(
                            value=verifier_variant_table_rows(
                                metrics["verifier"]["variants"]
                            ),
                            headers=[
                                "Variant",
                                "Accuracy",
                                "F1",
                                "Beneficial",
                                "Harmful",
                                "Net",
                                "Reviews",
                                "Extra latency (s)",
                                "Decision",
                            ],
                            datatype=[
                                "str",
                                "number",
                                "number",
                                "number",
                                "number",
                                "number",
                                "number",
                                "number",
                                "str",
                            ],
                            type="array",
                            interactive=False,
                            buttons=["copy", "fullscreen"],
                            label="Frozen V1/V2/V3 comparison",
                        )
                        gr.Dataframe(
                            value=verifier_failure_table_rows(
                                metrics["verifier"]["cases"]
                            ),
                            headers=[
                                "Target",
                                "Scope",
                                "GT",
                                "Baseline",
                                "V2",
                                "V3 label",
                                "Frozen final",
                                "Failure type",
                            ],
                            datatype=["str"] * 8,
                            type="array",
                            interactive=False,
                            buttons=["copy", "fullscreen"],
                            label="Final failure and regression audit",
                        )
                        gr.JSON(
                            value=metrics["verifier"]["policy"],
                            label="Frozen deployment policy",
                        )
                        gr.File(
                            value=metrics["verifier"]["report_files"],
                            file_count="multiple",
                            interactive=False,
                            label="Verifier audit artifacts",
                        )
    return app, gr.themes.Base()


def main() -> None:
    args = parse_args()
    runtime = DemoRuntime(
        PROJECT_ROOT,
        args.config,
        inference_enabled=not args.results_only,
    )
    store = FrozenBenchmarkStore(PROJECT_ROOT)
    metrics = load_demo_metrics(PROJECT_ROOT)
    app, theme = build_app(
        runtime,
        store,
        metrics,
        inference_enabled=not args.results_only,
    )
    auth = (
        (args.auth_user, args.auth_password)
        if args.auth_user is not None
        else None
    )
    app.queue(default_concurrency_limit=1, max_size=8).launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
        inbrowser=args.inbrowser,
        auth=auth,
        show_error=True,
        theme=theme,
        css=APP_CSS,
        footer_links=["settings"],
        ssr_mode=False,
    )


if __name__ == "__main__":
    main()
