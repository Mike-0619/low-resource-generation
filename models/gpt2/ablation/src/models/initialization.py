"""Build shared GPT-2 initializations and load models for training."""

import argparse
from dataclasses import asdict
from fcntl import LOCK_EX, flock
from pathlib import Path

import torch
from transformers import GPT2Config, GPT2LMHeadModel, get_cosine_schedule_with_warmup

from ..training.utils import (
    DEFAULT_CONFIG,
    INITIALIZATION_ROOT,
    SEEDS,
    SPECIAL_TOKEN_IDS,
    VOCAB_SIZES,
    ModelSpec,
    ProjectConfig,
    load_config,
    staged_directory,
)
from .attention import configure_model_attention, require_model_backend


def make_gpt2_config(spec: ModelSpec, vocab_size: int) -> GPT2Config:
    return GPT2Config(
        **asdict(spec),
        vocab_size=vocab_size,
        n_ctx=spec.n_positions,
        bos_token_id=SPECIAL_TOKEN_IDS["<s>"],
        eos_token_id=SPECIAL_TOKEN_IDS["</s>"],
        pad_token_id=SPECIAL_TOKEN_IDS["<pad>"],
        use_cache=False,
    )


def derive_smaller_model(
    canonical: GPT2LMHeadModel, spec: ModelSpec, vocab_size: int
) -> GPT2LMHeadModel:
    derived = GPT2LMHeadModel(make_gpt2_config(spec, vocab_size))
    state = canonical.state_dict()
    for name in ("transformer.wte.weight", "lm_head.weight"):
        state[name] = state[name][:vocab_size]
    derived.load_state_dict(state, strict=True)
    return derived


def build_initialization_family(spec: ModelSpec, *, overwrite: bool = False) -> Path:
    target = INITIALIZATION_ROOT
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.with_name(f".{target.name}.lock").open("a") as lock:
        flock(lock, LOCK_EX)
        if target.exists() and not overwrite:
            return target
        with staged_directory(target) as staging:
            for seed in SEEDS:
                torch.manual_seed(seed)
                canonical = GPT2LMHeadModel(make_gpt2_config(spec, VOCAB_SIZES[-1]))
                for vocab_size in VOCAB_SIZES:
                    model = (
                        canonical
                        if vocab_size == VOCAB_SIZES[-1]
                        else derive_smaller_model(canonical, spec, vocab_size)
                    )
                    output = staging / f"vocab_{vocab_size}" / f"seed_{seed}"
                    model.save_pretrained(output, safe_serialization=True)
                    if model is not canonical:
                        del model
                del canonical
    return target


def load_training_model(
    path: str | Path, *, spec: ModelSpec, vocab_size: int, attention_backend: str = "sdpa"
) -> GPT2LMHeadModel:
    config = configure_model_attention(make_gpt2_config(spec, vocab_size), attention_backend)
    model = GPT2LMHeadModel.from_pretrained(
        path,
        config=config,
        local_files_only=True,
        attn_implementation=attention_backend,
    )
    require_model_backend(model, attention_backend)
    return model


def initialize_model_and_optimizers(
    path: str | Path, config: ProjectConfig, *, vocab_size: int, device: torch.device,
    total_steps: int, warmup_steps: int,
):
    training = config.training
    model = load_training_model(
        path, spec=config.model, vocab_size=vocab_size,
        attention_backend=config.attention_backend,
    ).to(device)
    actual_backend = model.config._attn_implementation
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=training.learning_rate,
        betas=(training.adam_beta1, training.adam_beta2),
        eps=training.adam_epsilon, weight_decay=training.weight_decay, fused=True,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
    )
    return model, optimizer, scheduler, actual_backend


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Build the shared controlled GPT-2 initialization family."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    target = build_initialization_family(config.model, overwrite=args.overwrite)
    print(f"Ready: shared initializations -> {target}", flush=True)


if __name__ == "__main__":
    main()
