"""Baseline configuration, experiment paths and filesystem safety helpers."""
import json
from dataclasses import dataclass
import os
import re
import shutil
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
import yaml

from .attention import validate_attention

def _write_atomic(path: str | Path, text: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(target)


def write_json_atomic(path: str | Path, payload: Any) -> None:
    _write_atomic(path, json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def write_yaml_atomic(path: str | Path, payload: Any) -> None:
    _write_atomic(path, yaml.safe_dump(payload, sort_keys=False, allow_unicode=True))


@contextmanager
def staged_directory(target: Path) -> Iterator[Path]:
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.building-", dir=target.parent))
    try:
        yield staging
        if target.is_symlink():
            raise FileExistsError(f"Refusing symbolic-link output: {target}")
        if target.exists():
            target.rmdir()  # Only empty scaffolding may be replaced.
        staging.replace(target)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def require_fresh_run(results: Path, logs: Path) -> None:
    """Allow empty scaffolding and scheduler output, but never overwrite training."""
    for root in (results, logs):
        if root.is_symlink() or (root.exists() and not root.is_dir()):
            raise FileExistsError(f"Fresh run conflicts with existing path: {root}")
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_symlink():
                raise FileExistsError(f"Fresh run conflicts with existing path: {path}")
            if path.is_dir():
                continue
            if root == logs and re.fullmatch(r"[0-9]+(?:_[0-9]+)?\.(?:out|err)", path.name):
                continue
            raise FileExistsError(f"Fresh run conflicts with existing artifact: {path}")


TEMPLATE_COMMIT = "127d0a55def41e878afcfbd7dd85eb6732c2abdf"


def load_experiment(path):
    path = Path(path).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    cfg.setdefault("attention_backend", "sdpa")
    validate_attention(cfg["attention_backend"], cfg["model"]["attn_pdrop"])
    root = PROJECT_ROOT
    experiment_paths(cfg, root)
    return cfg, root


@dataclass(frozen=True)
class BaselinePaths:
    cleaned: Path
    tokenizer: Path
    results: Path
    logs: Path


def experiment_paths(cfg, root):
    size, language = cfg["dataset"]["size"], cfg["dataset"]["language"]
    seed = cfg["training"]["seed"]
    if (size, language) not in (("10M", "English"), ("100M", "English"),
                               ("10M", "German"), ("108M", "German")):
        raise ValueError(f"Unsupported baseline language/size: {language}/{size}")
    condition = Path(language) / size
    return BaselinePaths(
        cleaned=root / "data" / condition / "cleaned",
        tokenizer=root / "tokenizer" / condition,
        results=root / "results" / condition / f"seed_{seed}",
        logs=root / "logs" / condition / "train",
    )


def require_empty(path):
    if path.is_symlink() or (path.exists() and (not path.is_dir() or any(path.iterdir()))):
        raise FileExistsError(f"Refusing to overwrite existing artifacts: {path}")
