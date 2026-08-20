"""Prompt templates for the visual assistant."""

DEFAULT_SYSTEM_PROMPT = (
    "You are a careful visual assistant. Answer based only on visible image "
    "evidence. If the image does not provide enough evidence, say so clearly."
)


def build_vlm_messages(
    image_path: str,
    question: str,
    system_prompt: str | None = None,
) -> list[dict]:
    """Build a Qwen-VL compatible message payload."""
    return [
        {
            "role": "system",
            "content": system_prompt or DEFAULT_SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": question},
            ],
        },
    ]
