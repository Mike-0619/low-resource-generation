"""Single-GPU BabyLM 2026 recipe adapted to the project's existing environment."""

import math
import os
import random
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from ..data.data_utils import CachedBabyLMDataset, collate_batch, token_cache_directory
from ..models.attention import require_training_runtime
from ..models.initialization import initialize_model_and_optimizers
from .checkpoint import export_final, save_checkpoint
from .utils import (PROJECT_ROOT, TEMPLATE_COMMIT, experiment_paths, parse_experiment,
                    require_fresh_run, write_json_atomic, write_yaml_atomic)


def masked_loss(logits, targets, mask):
    # Preserve the official log-softmax/gather computation and token weighting.
    log_probs = F.log_softmax(logits, dim=2)
    selected = torch.gather(log_probs, 2, targets.unsqueeze(2)).squeeze(2)
    valid_tokens = mask.sum()
    return -(selected * mask).sum() / valid_tokens, valid_tokens


def schedule_steps(epoch_steps, epochs, warmup_fraction):
    total = epoch_steps * epochs
    return total, int(total * warmup_fraction)


def compute_checkpoint_steps(epoch_steps, words_per_epoch, epochs):
    """Official zero-based milestone mapping, including the final-step clamp."""
    total_words = words_per_epoch * epochs
    milestones = (
        *range(1_000_000, min(10_000_000, total_words + 1), 1_000_000),
        *range(10_000_000, min(100_000_000, total_words + 1), 10_000_000),
        *range(100_000_000, min(1_000_000_001, total_words + 1), 100_000_000),
    )
    result = {}
    last_step = epoch_steps * epochs - 1
    for words in milestones:
        step = min(int(words * (epoch_steps / words_per_epoch)), last_step)
        if step > 0:
            result[step] = f"{words // 1_000_000}M" if words < 1_000_000_000 else f"{words // 1_000_000_000}B"
    return result


def train_epoch(
    model, optimizer, scheduler, loader, *, epoch, steps_per_epoch, log_interval,
    gradient_clip_norm, milestones, checkpoint_dir,
):
    epoch_start = epoch * steps_per_epoch
    model.train()
    total_loss = total_tokens = 0
    for index, batch in enumerate(loader):
        inputs, targets, mask = (item.to("cuda") for item in batch)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(inputs).logits
            loss, valid_tokens = masked_loss(logits, targets, mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        count = int(valid_tokens.item())
        batch_loss = loss.item()
        total_loss += batch_loss * count
        total_tokens += count
        global_step = epoch_start + index
        if index % log_interval == 0:
            print(f"epoch={epoch + 1} step={index + 1}/{steps_per_epoch} loss={batch_loss:.6f}", flush=True)
        if global_step in milestones:
            save_checkpoint(checkpoint_dir / f"epoch_{milestones[global_step]}", model, optimizer, scheduler)
    return {"loss": total_loss / total_tokens}


def train(cfg, root):
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError("Main baseline uses one GPU and global batch 16; do not launch with torchrun")
    backend = cfg.get("attention_backend", "sdpa")
    runtime = require_training_runtime(backend, cfg["model"]["attn_pdrop"])
    training = cfg["training"]
    seed = training["seed"]
    paths = experiment_paths(cfg, root)
    run = paths.results
    records = run / "records"
    checkpoint_dir = run / "checkpoints"
    require_fresh_run(run, paths.logs)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    tokenizer = AutoTokenizer.from_pretrained(paths.tokenizer, local_files_only=True)
    if len(tokenizer) != cfg["model"]["vocab_size"]:
        raise ValueError("Tokenizer vocabulary does not match this experiment")
    cache = token_cache_directory(root, cfg["dataset"]["language"], cfg["dataset"]["size"])
    dataset = CachedBabyLMDataset(paths.cleaned, tokenizer, training["sequence_length"], cache)
    print(f"Baseline token cache {dataset.cache_id}: {len(dataset)} samples", flush=True)
    loader = DataLoader(dataset, batch_size=training["batch_size"], shuffle=True,
                        collate_fn=partial(collate_batch, eos=tokenizer.eos_token_id))
    steps_per_epoch = len(loader)
    epochs = training["epochs"]
    log_interval = max(1, math.ceil(steps_per_epoch / 20))
    total_steps, warmup = schedule_steps(steps_per_epoch, epochs, training["warmup_fraction"])
    model, optimizer, scheduler, actual_backend = initialize_model_and_optimizers(
        cfg, backend=backend, total_steps=total_steps, warmup_steps=warmup,
    )
    runtime["actual_attention_backend"] = actual_backend
    milestones = compute_checkpoint_steps(steps_per_epoch, training["words_per_epoch"], epochs)
    write_yaml_atomic(records / "config.yaml", {
        **cfg, "steps_per_epoch": steps_per_epoch, "total_steps": total_steps, "warmup_steps": warmup,
        "attention_backend": backend, "runtime": runtime,
        "template_commit": TEMPLATE_COMMIT,
        "token_cache": {"id": dataset.cache_id, "samples": len(dataset)},
    }, exclusive=True)
    history = []
    for epoch in range(epochs):
        torch.cuda.empty_cache()
        metrics = train_epoch(
            model, optimizer, scheduler, loader, epoch=epoch, steps_per_epoch=steps_per_epoch,
            log_interval=log_interval, gradient_clip_norm=training["gradient_clip_norm"],
            milestones=milestones, checkpoint_dir=checkpoint_dir,
        )
        history.append({"epoch": epoch, **metrics})
        write_json_atomic(records / "metrics.json", history)
        save_checkpoint(checkpoint_dir / f"epoch_{epoch}", model, optimizer, scheduler)
        print(f"Epoch {epoch}; train loss: {metrics['loss']}", flush=True)
    export_final(run / "final", model, tokenizer)
    write_json_atomic(records / "run_summary.json", {"epochs": epochs, "optimizer_steps": total_steps,
        "final_model": str((run / "final").relative_to(PROJECT_ROOT)), "seed": seed,
        "attention_backend": actual_backend, "requested_attention_backend": backend,
        "attention_dropout": cfg["model"]["attn_pdrop"], "runtime": runtime,
        "protocol": cfg["protocol"]})


def main(argv=None):
    cfg, root = parse_experiment('Train a fresh single-GPU GPT-2 baseline experiment.', argv)
    train(cfg, root)


if __name__ == "__main__":
    main()
