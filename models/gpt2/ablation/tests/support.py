from pathlib import Path

from src.training.utils import ModelSpec


def tiny_spec() -> ModelSpec:
    return ModelSpec(
        1, 64, 1, 32, "gelu_new", 0.02, 1e-5, 0.0, 0.0, 0.0, True
    )


class DummyTokenizer:
    def save_pretrained(self, directory) -> None:
        Path(directory, "tokenizer.json").write_text("{}", encoding="utf-8")


class DummyModel:
    def save_pretrained(self, directory, **_kwargs) -> None:
        Path(directory, "model.safetensors").write_bytes(b"")
