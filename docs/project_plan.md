# Project Plan

## Positioning

This project is designed as a resume-ready visual multimodal system rather than a simple model demo.

Target capabilities:

- Open-vocabulary image understanding
- Phrase grounding
- Instance segmentation
- Evidence-based VQA
- Hallucination checking
- Mask-guided reasoning
- Interactive demo and measurable evaluation

## Why This Project Matters

Generic VLMs can answer image questions, but their answers are often not tied to verifiable visual evidence. This project aims to make answers more grounded by connecting VLM reasoning with boxes, masks, and evidence checks.

## Milestones

### Stage 1: VLM Baseline

- Run single-image VLM inference with Qwen3-VL-8B or Qwen2.5-VL-7B fallback.
- Save JSON outputs.
- Collect 20 image-question examples.
- Summarize success and failure cases.

### Stage 2: Grounding

- Parse target phrases from user questions.
- Ground phrases to boxes.
- Visualize boxes.

### Stage 3: Segmentation

- Use box prompts to generate masks with SAM2.
- Visualize masks.
- Save structured visual evidence.

### Stage 4: Evidence-Based Answering

- Force final answers to cite detected boxes/masks.
- Mark unsupported claims.
- Add low-confidence or refusal behavior.

### Stage 5: Evaluation and Demo

- Build a hard-case image set.
- Evaluate grounding, segmentation, VQA, hallucination, and latency.
- Build a Gradio demo.
