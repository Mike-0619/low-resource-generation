"""Build GPT-2 and its optimization state for a Baseline run."""

import torch
from transformers import GPT2Config, GPT2LMHeadModel, get_cosine_schedule_with_warmup

from .attention import configure_model_attention, require_model_backend


def model_config(cfg):
    return GPT2Config(
        **cfg["model"], n_ctx=cfg["model"]["n_positions"],
        bos_token_id=1, eos_token_id=2, use_cache=True,
        reorder_and_upcast_attn=False, scale_attn_by_inverse_layer_idx=False,
        scale_attn_weights=True,
    )


def initialize_model_and_optimizers(cfg, *, backend, total_steps, warmup_steps):
    training = cfg["training"]
    config = configure_model_attention(model_config(cfg), backend)
    model = GPT2LMHeadModel(config).to("cuda")
    actual_backend = require_model_backend(model, backend)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=training["learning_rate"], weight_decay=training["weight_decay"],
        betas=(training["adam_beta1"], training["adam_beta2"]), eps=training["adam_epsilon"],
    )
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    return model, optimizer, scheduler, actual_backend
