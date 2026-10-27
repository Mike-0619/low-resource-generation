"""Single-GPU BabyLM 2026 recipe adapted to the project's existing environment."""

from pathlib import Path as _CachePath
import runpy as _cache_runpy
_configure_cache = _cache_runpy.run_path(str(_CachePath(__file__).resolve().parents[6] / "config/cache.py"))["configure"]
_configure_cache("low-resource-generation")

import argparse
import math
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, GPT2LMHeadModel, get_cosine_schedule_with_warmup

from .attention import configure_model_attention, require_model_backend, require_training_runtime
from .data_utils import BaselineDataset, collate_batch
from .models import model_config, save_checkpoint, export_final
from .utils import (PROJECT_ROOT, TEMPLATE_COMMIT, experiment_paths, load_experiment,
                    require_fresh_run, write_json_atomic, write_yaml_atomic)


def masked_loss(logits, targets, mask):
    # Preserve the official log-softmax/gather computation and token weighting.
    log_probs = F.log_softmax(logits, dim=2)
    selected = torch.gather(log_probs, 2, targets.unsqueeze(2)).squeeze(2)
    return -(selected * mask).sum() / mask.sum()


def schedule_steps(epoch_steps, epochs, warmup_fraction):
    total = epoch_steps * epochs
    return total, int(total * warmup_fraction)


def checkpoint_steps(epoch_steps, words_per_epoch, epochs):
    """Official zero-based milestone mapping, including the final-step clamp."""
    total_words = words_per_epoch * epochs
    milestones = list(range(1_000_000, min(10_000_001, total_words + 1), 1_000_000))
    milestones.extend(range(10_000_000, min(100_000_001, total_words + 1), 10_000_000))
    milestones.extend(range(100_000_000, min(1_000_000_001, total_words + 1), 100_000_000))
    result = {}
    for words in sorted(set(milestones)):
        step = min(int(words * (epoch_steps / words_per_epoch)), epoch_steps * epochs - 1)
        if step > 0:
            result[step] = f"{words // 1_000_000}M" if words < 1_000_000_000 else f"{words // 1_000_000_000}B"
    return result


def train(cfg, root):
    from functools import partial

    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError("Main baseline uses one GPU and global batch 16; do not launch with torchrun")
    backend = cfg.get("attention_backend", "sdpa")
    runtime = require_training_runtime(backend, cfg["model"]["attn_pdrop"])
    training = cfg["training"]
    seed = training["seed"]
    paths = experiment_paths(cfg, root)
    run = paths.results
    records = run / "records"
    require_fresh_run(run, paths.logs)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    tokenizer = AutoTokenizer.from_pretrained(paths.tokenizer, local_files_only=True)
    if len(tokenizer) != cfg["model"]["vocab_size"]:
        raise ValueError("Tokenizer vocabulary does not match this experiment")
    dataset = BaselineDataset(paths.cleaned, tokenizer, training["sequence_length"])
    loader = DataLoader(dataset, batch_size=training["batch_size"], shuffle=True,
                        collate_fn=partial(collate_batch, eos=tokenizer.eos_token_id))
    total_steps, warmup = schedule_steps(len(loader), training["epochs"], training["warmup_fraction"])
    config = configure_model_attention(model_config(cfg), backend)
    model = GPT2LMHeadModel(config).to("cuda")
    actual_backend = require_model_backend(model, backend)
    runtime["actual_attention_backend"] = actual_backend
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=training["learning_rate"], weight_decay=training["weight_decay"],
        betas=(training["adam_beta1"], training["adam_beta2"]), eps=training["adam_epsilon"],
    )
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup, total_steps)
    milestones = checkpoint_steps(len(loader), training["words_per_epoch"], training["epochs"])
    write_yaml_atomic(records / "config.yaml", {
        **cfg, "steps_per_epoch": len(loader), "total_steps": total_steps, "warmup_steps": warmup,
        "attention_backend": backend, "runtime": runtime,
        "template_commit": TEMPLATE_COMMIT,
    })
    history = []
    for epoch in range(training["epochs"]):
        torch.cuda.empty_cache()
        model.train()
        total_loss = total_tokens = 0
        for index, batch in enumerate(loader):
            inputs, targets, mask = (item.to("cuda") for item in batch)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(inputs).logits
                loss = masked_loss(logits, targets, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), training["gradient_clip_norm"])
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            count = int(mask.sum().item())
            total_loss += loss.item() * count
            total_tokens += count
            global_step = epoch * len(loader) + index
            if index % max(1, math.ceil(len(loader) / 20)) == 0:
                print(f"epoch={epoch + 1} step={index + 1}/{len(loader)} loss={loss.item():.6f}", flush=True)
            if global_step in milestones:
                save_checkpoint(run / "checkpoints" / f"epoch_{milestones[global_step]}", model, optimizer, scheduler)
        metrics = {"loss": total_loss / total_tokens}
        history.append({"epoch": epoch, **metrics})
        write_json_atomic(records / "metrics.json", history)
        save_checkpoint(run / "checkpoints" / f"epoch_{epoch}", model, optimizer, scheduler)
        print(f"Epoch {epoch}; train loss: {metrics['loss']}", flush=True)
    export_final(run / "final", model, tokenizer)
    write_json_atomic(records / "run_summary.json", {"epochs": training["epochs"], "optimizer_steps": total_steps,
        "final_model": str((run / "final").relative_to(PROJECT_ROOT)), "seed": seed,
        "attention_backend": actual_backend, "requested_attention_backend": backend,
        "attention_dropout": cfg["model"]["attn_pdrop"], "runtime": runtime,
        "protocol": cfg["protocol"]})


def main(argv=None):
    parser = argparse.ArgumentParser(description='Train a fresh single-GPU GPT-2 baseline experiment.')
    parser.add_argument("--config", required=True, help="Experiment config, relative to model/gpt2/baseline")
    args = parser.parse_args(argv)
    cfg, root = load_experiment(args.config)
    train(cfg, root)


if __name__ == "__main__":
    main()
