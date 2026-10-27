"""Explicit attention selection for baseline training."""

import importlib.metadata
import platform

import torch
from transformers.utils import is_flash_attn_3_available


def validate_attention(backend: str, dropout: float) -> str:
    if backend not in ("sdpa", "flash_attention_3"):
        raise ValueError(f"Unsupported attention_backend: {backend!r}; choose sdpa or flash_attention_3")
    if not 0.0 <= dropout < 1.0:
        raise ValueError(f"attention dropout must be in [0, 1), got {dropout}")
    if backend == "flash_attention_3" and dropout != 0.0:
        raise ValueError("flash_attention_3 requires attention dropout 0.0; the configured dropout will not be changed")
    return backend


def require_local_fa3() -> None:
    # Prevent Transformers from selecting/downloading a Hub kernel when the
    # explicitly requested local package is absent.
    if not is_flash_attn_3_available(kernels_fallback_ok=False):
        raise RuntimeError("The local official flash-attn-3 package is unavailable; Hub fallback is disabled")


def require_training_runtime(backend: str, dropout: float) -> dict:
    validate_attention(backend, dropout)
    if not torch.cuda.is_available():
        raise RuntimeError("Training requires CUDA")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("Training requires BF16 support")
    capability = torch.cuda.get_device_capability()
    if backend == "flash_attention_3":
        if capability != (9, 0):
            raise RuntimeError(f"flash_attention_3 requires Hopper SM90, got sm_{capability[0]}{capability[1]}")
        require_local_fa3()
    runtime = {
        "requested_attention_backend": backend,
        "attention_dropout": dropout,
        "python_version": platform.python_version(),
        "gpu_name": torch.cuda.get_device_name(),
        "compute_capability": f"{capability[0]}.{capability[1]}",
        "torch_version": str(torch.__version__),
        "torch_cuda_version": torch.version.cuda,
        "transformers_version": importlib.metadata.version("transformers"),
        "tokenizers_version": importlib.metadata.version("tokenizers"),
    }
    if backend == "flash_attention_3":
        runtime["fa3_version"] = importlib.metadata.version("flash-attn-3")
    return runtime


def require_model_backend(model, requested: str) -> str:
    actual = model.config._attn_implementation
    if actual != requested:
        raise RuntimeError(f"Attention backend fallback detected: expected {requested}, got {actual}")
    return actual


def configure_model_attention(config, backend: str):
    validate_attention(backend, config.attn_pdrop)
    if backend == "flash_attention_3":
        require_local_fa3()
    config._attn_implementation = backend
    return config
