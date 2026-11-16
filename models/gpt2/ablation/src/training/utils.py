"""Shared configuration, path ownership, and filesystem helpers."""

import json
import os
import re
import shutil
from tempfile import TemporaryDirectory
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from ..models.attention import validate_attention


def _write_atomic(path: str | Path, text: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f".{target.name}.tmp-", dir=target.parent) as directory:
        temporary = Path(directory) / target.name
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(target)


def write_json_atomic(path: str | Path, payload: Any) -> None:
    _write_atomic(path, json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def write_yaml_atomic(path: str | Path, payload: Any) -> None:
    _write_atomic(path, yaml.safe_dump(payload, sort_keys=False, allow_unicode=True))


@contextmanager
def staged_directory(target: Path) -> Iterator[Path]:
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.parent / f".{target.name}.building-{os.getpid()}"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    try:
        yield staging
        if target.exists():
            shutil.rmtree(target)
        staging.replace(target)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INITIALIZATION_ROOT = PROJECT_ROOT / "initializations"
CONFIG_PATHS = {
    "10m_en_spans": PROJECT_ROOT / "configs/English/10M/shared_spans.yaml",
    "100m_en_spans": PROJECT_ROOT / "configs/English/100M/shared_spans.yaml",
    "100m_de_spans": PROJECT_ROOT / "configs/German/100M/shared_spans.yaml",
    "10m_en_balanced": PROJECT_ROOT / "configs/English/10M/balanced_batch.yaml",
    "100m_en_balanced": PROJECT_ROOT / "configs/English/100M/balanced_batch.yaml",
    "100m_de_balanced": PROJECT_ROOT / "configs/German/100M/balanced_batch.yaml",
}
CONDITION_GROUPS = {
    variant: tuple(name for name, path in CONFIG_PATHS.items() if path.stem == variant)
    for variant in ("shared_spans", "balanced_batch")
}
DEFAULT_CONFIG = CONFIG_PATHS["10m_en_spans"]
VOCAB_SIZES = (512, 1024, 2048, 4096, 8192, 16384)
SEEDS = (42, 43, 44)
SPECIAL_TOKEN_IDS = {"<unk>": 0, "<s>": 1, "</s>": 2, "<pad>": 3, "<mask>": 4}


@dataclass(frozen=True)
class ModelSpec:
    n_layer: int
    n_embd: int
    n_head: int
    n_positions: int
    activation_function: str
    initializer_range: float
    layer_norm_epsilon: float
    attn_pdrop: float
    resid_pdrop: float
    embd_pdrop: float
    tie_word_embeddings: bool


@dataclass(frozen=True)
class TrainingSpec:
    sequence_length: int
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    adam_beta1: float
    adam_beta2: float
    adam_epsilon: float
    gradient_clip_norm: float
    warmup_fraction: float
    num_workers: int


@dataclass(frozen=True)
class ConditionSpec:
    name: str
    size: str
    language: str
    variant: str
    dataset_repo_id: str
    dataset_revision: str
    corpus_path: str

    def __post_init__(self):
        if self.variant not in ("shared_spans", "balanced_batch"):
            raise ValueError(f"Unsupported route: {self.variant}")


@dataclass(frozen=True)
class ProjectConfig:
    model: ModelSpec
    training: TrainingSpec
    condition: ConditionSpec
    attention_backend: str = "sdpa"

    def __post_init__(self):
        validate_attention(self.attention_backend, self.model.attn_pdrop)

    def snapshot(self, vocab_size: int, seed: int) -> dict[str, Any]:
        return {**asdict(self), "vocab_size": vocab_size, "seed": seed}


@dataclass(frozen=True)
class ConditionPaths:
    cleaned: Path
    tokenizer_root: Path
    tokenized_root: Path
    samples: Path
    shared_tokenized_root: Path

    def tokenizer(self, vocab_size: int) -> Path:
        return self.tokenizer_root / f"vocab_{vocab_size}"

    def token_cache(self, vocab_size: int) -> Path:
        return self.tokenized_root / f"vocab_{vocab_size}"

    def shared_cache(self, vocab_size: int) -> Path:
        return self.shared_tokenized_root / f"vocab_{vocab_size}"


@dataclass(frozen=True)
class RunPaths:
    initialization: Path
    run: Path
    last: Path
    final: Path
    logs: Path
    records: Path
    tokenizer: Path
    token_cache: Path

    def relative_paths(self) -> dict[str, str]:
        names = ("initialization", "tokenizer", "token_cache", "last", "final", "logs", "records")
        return {name: str(getattr(self, name).relative_to(PROJECT_ROOT)) for name in names}


def load_config(path: str | Path = DEFAULT_CONFIG) -> ProjectConfig:
    source = Path(path).expanduser()
    if not source.is_absolute():
        source = PROJECT_ROOT / source
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    return ProjectConfig(
        model=ModelSpec(**payload["model"]),
        training=TrainingSpec(**payload["training"]),
        condition=ConditionSpec(**payload["condition"]),
        attention_backend=payload.get("attention_backend", "sdpa"),
    )


def configs_for_condition(condition: str) -> tuple[ProjectConfig, ...]:
    if condition not in CONDITION_GROUPS and condition not in CONFIG_PATHS:
        raise ValueError(f"Unsupported condition: {condition}")
    return tuple(
        load_config(CONFIG_PATHS[name]) for name in CONDITION_GROUPS.get(condition, (condition,))
    )


def condition_paths(condition: ConditionSpec) -> ConditionPaths:
    family = Path(condition.language) / condition.size
    data = PROJECT_ROOT / "data" / family
    return ConditionPaths(
        cleaned=data / condition.variant / "cleaned",
        tokenizer_root=PROJECT_ROOT / "tokenizer" / family,
        tokenized_root=data / condition.variant / "tokenized",
        samples=data / "shared_spans" / "samples",
        shared_tokenized_root=data / "shared_spans" / "tokenized",
    )


def run_paths(condition: ConditionSpec, vocab_size: int, seed: int) -> RunPaths:
    family = Path(condition.language) / condition.size
    run = PROJECT_ROOT / "results" / family / condition.variant / f"vocab_{vocab_size}" / f"seed_{seed}"
    data_paths = condition_paths(condition)
    return RunPaths(
        initialization=INITIALIZATION_ROOT / f"vocab_{vocab_size}" / f"seed_{seed}",
        run=run,
        last=run / "checkpoints" / "last",
        final=run / "final",
        records=run / "records",
        logs=PROJECT_ROOT / "logs" / family / condition.variant / "train",
        tokenizer=data_paths.tokenizer(vocab_size),
        token_cache=data_paths.token_cache(vocab_size),
    )


def require_fresh_run(results: Path, *log_dirs: Path) -> None:
    """Allow empty scaffolding and scheduler output, but never overwrite training."""
    for root in (results, *log_dirs):
        if root.is_symlink() or (root.exists() and not root.is_dir()):
            raise FileExistsError(f"Fresh run conflicts with existing path: {root}")
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_symlink():
                raise FileExistsError(f"Fresh run conflicts with existing path: {path}")
            if path.is_dir():
                continue
            if root != results and re.fullmatch(r"[0-9]+(?:_[0-9]+)?\.(?:out|err)", path.name):
                continue
            raise FileExistsError(f"Fresh run conflicts with existing artifact: {path}")
