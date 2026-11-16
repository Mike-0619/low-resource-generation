"""Save model artifacts and restore optimizer, scheduler, and RNG state."""

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .utils import staged_directory


def capture_rng_state(device: torch.device) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state(state["cuda"])


def save_model_artifact(
    target: Path, model, tokenizer, state: dict[str, Any] | None = None
) -> None:
    with staged_directory(target) as staging:
        model.save_pretrained(staging, safe_serialization=True)
        tokenizer.save_pretrained(staging)
        if state is not None:
            torch.save(state, staging / "training_state.pt")


def load_training_state(
    *,
    checkpoint: Path,
    optimizer,
    scheduler,
    expected_signature: dict[str, Any],
) -> tuple[int, int]:
    state = torch.load(checkpoint / "training_state.pt", map_location="cpu", weights_only=False)
    if state.get("format") != "single_gpu_v1":
        raise ValueError("Unsupported legacy checkpoint; start a new single-GPU run")
    if state["signature"] != expected_signature:
        raise ValueError("Resume protocol signature does not match this run")
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    restore_rng_state(state["rng_state"])
    return state["completed_epoch"], state["global_step"]
