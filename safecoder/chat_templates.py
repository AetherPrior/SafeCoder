"""Chat-template helpers for instruction-tuned models (Qwen3 no-think, etc.)."""

from __future__ import annotations

from typing import Any

# Matches LLaMA-Factory `qwen3_nothink`: Qwen3 chat format without reasoning/thinking blocks.
QWEN3_NOTHINK_KWARGS: dict[str, Any] = {"enable_thinking": False}

# Model aliases in SafeCoder that use the Qwen3 no-think template.
QWEN3_MODEL_PREFIXES = ("qwen3", "Qwen/Qwen3", "Qwen/Qwen2.5")


def is_qwen3_model(model_name: str) -> bool:
    name = model_name.lower()
    return any(p.lower() in name for p in QWEN3_MODEL_PREFIXES)


def get_chat_template_kwargs(model_name: str) -> dict[str, Any]:
    if is_qwen3_model(model_name) or is_qwen3_model(_base_model_name(model_name)):
        return dict(QWEN3_NOTHINK_KWARGS)
    return {}


def _base_model_name(model_name: str) -> str:
    for marker in ('-lora', '-sven'):
        if marker in model_name:
            return model_name.split(marker)[0]
    return model_name


def uses_chat_template(model_name: str, chat_models: dict[str, str] | None = None) -> bool:
    if chat_models and model_name in chat_models:
        return True
    base = _base_model_name(model_name)
    if chat_models and base in chat_models:
        return True
    return is_qwen3_model(model_name) or is_qwen3_model(base)


def configure_tokenizer(tokenizer, model_name: str) -> None:
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.pad_token = "<|endoftext|>"
    if is_qwen3_model(model_name) and getattr(tokenizer, "chat_template", None) is None:
        raise ValueError(
            f"Tokenizer for {model_name} has no chat_template; "
            "use a Qwen3 HuggingFace checkpoint with a built-in template."
        )


def apply_safecoder_chat_template(
    tokenizer,
    model_name: str,
    messages: list[dict[str, str]],
    *,
    tokenize: bool = False,
    add_generation_prompt: bool = False,
) -> str | list[int]:
    kwargs = get_chat_template_kwargs(model_name)
    return tokenizer.apply_chat_template(
        messages,
        tokenize=tokenize,
        add_generation_prompt=add_generation_prompt,
        **kwargs,
    )


def encode_chat_turn(tokenizer, model_name: str, user_content: str, assistant_content: str):
    """Tokenize a user/assistant turn and return (tokens, weights) for SafeCoder training."""
    messages = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content},
    ]
    full_text = apply_safecoder_chat_template(
        tokenizer, model_name, messages, tokenize=False, add_generation_prompt=False
    )
    prompt_text = apply_safecoder_chat_template(
        tokenizer,
        model_name,
        [{"role": "user", "content": user_content}],
        tokenize=False,
        add_generation_prompt=True,
    )

    be = tokenizer.encode_plus(full_text)
    tokens = be.data["input_ids"]
    weights = [0] * len(tokens)
    char_idx = max(0, len(prompt_text) - 1)
    token_start_idx = be.char_to_token(char_idx)
    if token_start_idx is None:
        token_start_idx = len(tokenizer.encode(prompt_text, add_special_tokens=False))
    else:
        token_start_idx += 1
    for token_idx in range(token_start_idx, len(tokens)):
        weights[token_idx] = 1
    return tokens, weights
