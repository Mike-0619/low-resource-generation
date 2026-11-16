"""Local causal language models and lossless first-word decoding."""
import json
import re
from pathlib import Path

from . import MAX_NEW_TOKENS
from .scoring import inspect_text

# Invert the byte-level BPE alphabet without lossy UTF-8 replacement.
_VISIBLE_BYTES = [*range(33, 127), *range(161, 173), *range(174, 256)]
BYTE_DECODER = {chr(value): value for value in _VISIBLE_BYTES}
BYTE_DECODER.update({
    chr(256 + index): value
    for index, value in enumerate(value for value in range(256) if value not in _VISIBLE_BYTES)
})
BYTE_FALLBACK = re.compile(rb"<0x([0-9a-fA-F]{2})>")


class TokenBytes:
    """Apply supported decoder operations while retaining invalid/pending bytes."""
    def __init__(self, tokenizer):
        if not getattr(tokenizer, "is_fast", False):
            raise ValueError("A fast tokenizer with a supported decoder is required")
        state = json.loads(tokenizer.backend_tokenizer.decoder.__getstate__())
        self.steps = state["decoders"] if state["type"] == "Sequence" else [state]
        self.byte_level = bool(self.steps) and self.steps[0]["type"] == "ByteLevel"
        if self.byte_level:
            special = set(tokenizer.all_special_ids)
            if any(not piece or any(char not in BYTE_DECODER for char in piece)
                   for piece, token in tokenizer.get_vocab().items() if token not in special):
                raise ValueError("Unsupported non-byte-level token in ByteLevel vocabulary")
        for index, step in enumerate(self.steps):
            kind = step["type"]
            if kind == "ByteLevel" and index == 0:
                continue
            if kind == "Replace" and (step.get("pattern"), step.get("content")) in [({"String": "▁"}, " "), ({"String": "\n "}, "\n")]:
                continue
            if kind == "Metaspace" and step.get("prepend_scheme", "always") in {"always", "first", "never"}:
                continue
            if kind == "Strip" and step.get("content") == " " and step.get("start", 0) >= 0 and step.get("stop", 0) >= 0:
                continue
            if kind not in {"ByteFallback", "Fuse"}:
                raise ValueError(f"Unsupported tokenizer decoder: {state}")
        kinds = {step["type"] for step in self.steps}
        if not self.byte_level and not kinds & {"Metaspace", "ByteFallback"}:
            raise ValueError(f"Unsupported tokenizer decoder: {state}")
        self.name = "ByteLevel" if self.byte_level else ("SentencePieceByteFallback" if "ByteFallback" in kinds else "Metaspace")

    def decode(self, pieces):
        chunks = ([bytes(BYTE_DECODER[char] for char in piece) for piece in pieces]
                  if self.byte_level else [piece.encode("utf-8") for piece in pieces])
        for step in self.steps:
            kind = step["type"]
            if kind == "ByteLevel":
                chunks = [b"".join(chunks)]
            elif kind == "Replace":
                pattern, replacement = step["pattern"]["String"].encode(), step["content"].encode()
                chunks = [chunk.replace(pattern, replacement) for chunk in chunks]
            elif kind == "Metaspace":
                replacement = step.get("replacement", "▁").encode()
                chunks = [chunk.replace(replacement, b"" if index == 0 and step.get("prepend_scheme", "always") != "never" else b" ")
                          for index, chunk in enumerate(chunks)]
            elif kind == "ByteFallback":
                decoded, pending = [], bytearray()
                for chunk in chunks:
                    match = BYTE_FALLBACK.fullmatch(chunk)
                    if match:
                        pending.append(int(match[1], 16))
                    else:
                        if pending:
                            decoded.append(bytes(pending))
                            pending.clear()
                        decoded.append(chunk)
                if pending:
                    decoded.append(bytes(pending))
                chunks = decoded
            elif kind == "Fuse":
                chunks = [b"".join(chunks)]
            elif kind == "Strip":
                stripped = []
                for chunk in chunks:
                    for _ in range(step.get("start", 0)):
                        if chunk.startswith(b" "):
                            chunk = chunk[1:]
                    for _ in range(step.get("stop", 0)):
                        if chunk.endswith(b" "):
                            chunk = chunk[:-1]
                    stripped.append(chunk)
                chunks = stripped
        return b"".join(chunks)


def resolve_device(name="cuda"):
    import torch
    if name not in {"cuda", "cpu"}:
        raise ValueError("device must be cuda or cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; use a GPU allocation or explicitly select --device cpu for limited debugging")
    return torch.device(name)


class FirstWord:
    """Inspect lossless bytes before acting on later invalid/special tokens."""
    def __init__(self, tokenizer, eos_token_ids=None, *, decoder=None):
        self.tokenizer = tokenizer
        self.decoder = decoder or TokenBytes(tokenizer)
        self.known_ids = set(tokenizer.get_vocab().values())
        self.eos_ids = set(eos_token_ids if eos_token_ids is not None else
                           ([] if tokenizer.eos_token_id is None else [tokenizer.eos_token_id]))
        self.special_ids = set(tokenizer.all_special_ids) | self.eos_ids

    def inspect(self, generated):
        terminal = next((index for index, token in enumerate(generated)
                         if token in self.special_ids or token not in self.known_ids), None)
        payload = generated[:terminal] if terminal is not None else generated
        pieces = self.tokenizer.convert_ids_to_tokens(payload)
        text = self.decoder.decode(pieces).decode("utf-8", errors="surrogateescape")
        result = inspect_text(text)
        if result is not None or terminal is None:
            return result
        token = generated[terminal]
        if token in self.eos_ids:
            return inspect_text(text, final=True)
        return {"prediction": "", "completed": False,
                "stop_reason": "special_token" if token in self.special_ids else "invalid_token"}


class BatchFirstWord:
    """Freeze each row at its first terminal event, before later padding."""

    def __init__(self, inspector, prompt_width, batch_size):
        self.inspector = inspector
        self.prompt_width = prompt_width
        self.stops = [None] * batch_size

    def __call__(self, input_ids, scores, **kwargs):
        import torch

        for row, stop in enumerate(self.stops):
            if stop is None:
                tokens = input_ids[row, self.prompt_width:].tolist()
                result = self.inspector.inspect(tokens)
                if result is not None:
                    self.stops[row] = (len(tokens), result)
        return torch.tensor([stop is not None for stop in self.stops], device=input_ids.device)



class LocalModel:
    """Validated local model state shared by context preparation and generation."""

    def __init__(self, path, tokenizer, model, device, *, context_limit=None, bos="auto", decoder=None):
        self.path, self.tokenizer, self.model, self.device = Path(path), tokenizer, model, device
        self.bos = bos
        config = model.config
        if config.is_encoder_decoder:
            raise ValueError("Only decoder-only causal language models are supported")
        size = min(config.vocab_size, model.get_input_embeddings().num_embeddings)
        if max(tokenizer.get_vocab().values()) >= size:
            raise ValueError(f"Tokenizer IDs exceed model vocab size {size}")
        eos = config.eos_token_id if config.eos_token_id is not None else tokenizer.eos_token_id
        eos_ids = list(eos) if isinstance(eos, (list, tuple)) else ([] if eos is None else [eos])
        if tokenizer.eos_token_id is not None and tokenizer.eos_token_id not in eos_ids:
            raise ValueError("Model eos_token_id does not include tokenizer eos_token_id")
        self.tokens = {}
        for name in ("bos_token_id", "pad_token_id"):
            configured, actual = getattr(config, name), getattr(tokenizer, name)
            if configured is not None and actual is not None and configured != actual:
                raise ValueError(f"Model {name}={configured} does not match tokenizer {name}={actual}")
            self.tokens[name] = configured if actual is None else actual
        if self.tokens["pad_token_id"] is None:
            if not eos_ids:
                raise ValueError("A padding token or EOS token is required for batching")
            self.tokens["pad_token_id"] = eos_ids[0]
        for token in [*self.tokens.values(), *eos_ids]:
            if token is not None and (not isinstance(token, int) or not 0 <= token < config.vocab_size):
                raise ValueError(f"Special token ID {token!r} is outside model vocab size {config.vocab_size}")
        self.tokens.update(eos_token_ids=eos_ids, eos_token_id=eos_ids[0] if len(eos_ids) == 1 else eos_ids or None)
        limits = [value for value in [tokenizer.model_max_length,
                  *(getattr(config, name, None) for name in ("max_position_embeddings", "n_positions", "n_ctx"))]
                  if isinstance(value, int) and 0 < value < 10**20]
        if context_limit is not None and (context_limit < 1 or limits and context_limit > min(limits)):
            raise ValueError("context_limit must be positive and within the model/tokenizer limit")
        if context_limit is None and not limits:
            raise ValueError("Cannot infer context limit; specify --context_limit")
        self.context_limit = min(limits) if context_limit is None else context_limit
        tokenizer.bos_token_id, tokenizer.pad_token_id = self.tokens["bos_token_id"], self.tokens["pad_token_id"]
        tokenizer.padding_side = "left"
        self.inspector = FirstWord(tokenizer, eos_ids, decoder=decoder)

    @classmethod
    def load(cls, path, device="cuda", dtype="auto", *, context_limit=None, bos="auto"):
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        path = Path(path).expanduser().resolve()
        weights = ("model.safetensors", "model.safetensors.index.json", "pytorch_model.bin", "pytorch_model.bin.index.json")
        if not path.is_dir() or not (path / "config.json").is_file() or not any((path / name).is_file() for name in weights):
            raise FileNotFoundError(f"Complete exported model required; final model is not ready: {path}")
        device = resolve_device(device)
        offline = {"local_files_only": True, "trust_remote_code": False}
        config = AutoConfig.from_pretrained(path, **offline)
        if config.is_encoder_decoder:
            raise ValueError("Only decoder-only causal language models are supported")
        tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True, **offline)
        decoder = TokenBytes(tokenizer)
        dtype = torch.float32 if dtype == "auto" and device.type == "cpu" else ("auto" if dtype == "auto" else getattr(torch, dtype))
        model = AutoModelForCausalLM.from_pretrained(path, config=config, dtype=dtype, **offline).to(device)
        model.eval()
        return cls(path, tokenizer, model, device, context_limit=context_limit, bos=bos, decoder=decoder)

    def metadata(self, prepared_inputs, batch_size, requested_dtype):
        """Describe actual inputs and decoding without starting generation."""
        tokens = self.tokens
        with_bos = bool(prepared_inputs and tokens["bos_token_id"] is not None and
                        all(item["input_ids"][0] == tokens["bos_token_id"] for item in prepared_inputs))
        return {
            "model": str(self.path), "model_path": str(self.path), "vocab_size": len(self.tokenizer),
            "bos_token_id": tokens["bos_token_id"], "eos_token_id": tokens["eos_token_id"],
            "eos_token_ids": tokens["eos_token_ids"], "pad_token_id": tokens["pad_token_id"],
            "context_limit": self.context_limit, "training_sequence_length": None,
            "decoding": {"do_sample": False, "num_beams": 1, "use_cache": True, "batch_size": batch_size},
            "device": str(self.device), "batch_size": batch_size,
            "dtype": str(self.model.dtype), "requested_dtype": requested_dtype,
            "model_type": self.model.config.model_type, "tokenizer_decoder": self.inspector.decoder.name,
            "bos_policy": self.bos, "with_bos": with_bos,
            "prompt_mode": "raw_continuation_with_bos" if with_bos else "raw_continuation",
        }

    def prepare(self, context):
        ids = self.tokenizer.encode(context, add_special_tokens=self.bos == "auto")
        if not ids:
            raise ValueError("LAMBADA context tokenized to an empty sequence")
        if self.bos == "always":
            if self.tokens["bos_token_id"] is None:
                raise ValueError("bos=always requires a BOS token")
            ids = [self.tokens["bos_token_id"], *ids]
        available = self.context_limit - len(ids)
        if available <= 0:
            raise ValueError(f"Context leaves no generation space within {self.context_limit} tokens")
        return {"input_ids": ids, "available": available, "budget": min(MAX_NEW_TOKENS, available)}

    def generate(self, prepared_inputs, batch_size=8):
        """Preserve input order and batch only equal generation budgets."""
        import torch

        with torch.inference_mode():
            batch = []
            for prepared in prepared_inputs:
                if batch and (len(batch) == batch_size or prepared["budget"] != batch[0]["budget"]):
                    yield from self._generate_batch(batch)
                    batch = []
                batch.append(prepared)
            if batch:
                yield from self._generate_batch(batch)

    def _generate_batch(self, prepared):
        import torch
        from transformers import GenerationConfig, StoppingCriteriaList

        budget = prepared[0]["budget"]
        width = max(len(row["input_ids"]) for row in prepared)
        padding = [width - len(row["input_ids"]) for row in prepared]
        input_ids = torch.tensor([[self.tokens["pad_token_id"]] * pad + row["input_ids"]
                                  for row, pad in zip(prepared, padding)], device=self.device)
        attention_mask = torch.tensor([[0] * pad + [1] * (width - pad)
                                       for pad in padding], device=self.device)
        stopping = BatchFirstWord(self.inspector, width, len(prepared))
        # Fresh settings prevent saved generation_config values from changing the protocol.
        config = GenerationConfig(
            do_sample=False, num_beams=1, num_return_sequences=1, max_new_tokens=budget, use_cache=True,
            **{name: self.tokens[name] for name in ("bos_token_id", "eos_token_id", "pad_token_id")},
        )
        output = self.model.generate(input_ids=input_ids, attention_mask=attention_mask,
                                     generation_config=config, stopping_criteria=StoppingCriteriaList([stopping]))
        results = []
        for row, item in enumerate(prepared):
            stop = stopping.stops[row]
            if stop is None:
                length, result = budget, {"prediction": "", "completed": False,
                    "stop_reason": "context_limit" if item["available"] <= MAX_NEW_TOKENS else "token_limit"}
            else:
                length, result = stop
            generated = output[row, width:width + length].tolist()
            results.append({"generated_ids": generated,
                "generated_text": self.tokenizer.decode(generated, skip_special_tokens=False, clean_up_tokenization_spaces=False),
                "generation_budget": budget, **result})
        return results
