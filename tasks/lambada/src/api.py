"""OpenAI-compatible Responses streams with bounded, ordered concurrency."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor

from . import MAX_NEW_TOKENS
from .scoring import inspect_text


API_SETTINGS = {"temperature": 0, "store": False}
API_INSTRUCTIONS = (
    "Predict the single most likely next word immediately following the passage.\n"
    "Return only that word, without repeating the passage or adding explanations."
)


def request_settings(reasoning_effort=None):
    settings = dict(API_SETTINGS)
    if reasoning_effort is not None:
        settings["reasoning"] = {"effort": reasoning_effort}
    return settings


def metadata(model, concurrency, reasoning_effort=None):
    """Describe requested Responses settings without making an API call."""
    return {
        "model": model, "model_path": None, "vocab_size": None,
        "with_bos": None, "bos_token_id": None, "context_limit": None,
        "prompt_mode": "next_word_instruction", "instructions": API_INSTRUCTIONS,
        "concurrency": concurrency,
        "decoding": {**request_settings(reasoning_effort), "stream": True},
        "api_protocol": "responses", "model_type": None,
        "tokenizer_decoder": None, "bos_policy": None,
    }


def request(client, example, model, reasoning_effort=None):
    text = ""
    with client.responses.create(
        model=model, input=example["context"], instructions=API_INSTRUCTIONS,
        max_output_tokens=MAX_NEW_TOKENS, stream=True, **request_settings(reasoning_effort),
    ) as stream:
        for event in stream:
            result = None
            if event.type == "response.output_text.delta":
                text += event.delta
                result = inspect_text(text)
            elif event.type == "response.completed":
                result = inspect_text(text, final=True, terminal_reason="completed")
            elif event.type == "response.incomplete":
                details = event.response.incomplete_details
                reason = details.reason if details else "incomplete"
                result = {"prediction": "", "completed": False,
                          "stop_reason": "token_limit" if reason == "max_output_tokens" else reason}
            elif event.type in {"response.failed", "error"}:
                raise RuntimeError(f"API stream failed: {event}")
            if result is not None:
                return {"generated_ids": None, "generated_text": text,
                        "generation_budget": MAX_NEW_TOKENS, **result}
    raise RuntimeError("API stream ended without a word boundary or terminal event")


def generate(examples, model, concurrency=4, reasoning_effort=None):
    from openai import OpenAI

    with OpenAI() as client:
        executor = ThreadPoolExecutor(max_workers=concurrency)
        pending = deque()
        try:
            for example in examples:
                pending.append(executor.submit(request, client, example, model, reasoning_effort))
                if len(pending) == concurrency:
                    yield pending.popleft().result()
            while pending:
                yield pending.popleft().result()
        finally:
            executor.shutdown(wait=True, cancel_futures=True)
