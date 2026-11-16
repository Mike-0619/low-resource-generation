"""Train one single-GPU GPT-2 vocabulary-ablation run."""

import argparse
import json
import math
import shutil
import time
from typing import Any

import torch
import torch.nn.functional as F
from transformers import set_seed

from ..models.attention import require_training_runtime
from ..data.train_tokenizer import load_tokenizer
from ..data.data_utils import IGNORE_INDEX, METADATA_FILE, CachedBabyLMDataset, make_dataloader
from ..models.initialization import initialize_model_and_optimizers
from .checkpoint import capture_rng_state, load_training_state, save_model_artifact
from .utils import (
    DEFAULT_CONFIG, PROJECT_ROOT, SEEDS, VOCAB_SIZES, condition_paths, load_config,
    require_fresh_run, run_paths, write_json_atomic, write_yaml_atomic,
)


def causal_lm_loss(
    logits: torch.Tensor, labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    target_count = int((labels != IGNORE_INDEX).sum().item())
    if not target_count:
        raise ValueError("Batch contains no target tokens")
    nll = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), labels.reshape(-1),
        ignore_index=IGNORE_INDEX, reduction="sum",
    )
    return nll / target_count, nll.detach(), target_count


def train_epoch(
    training, model, optimizer, scheduler, dataloader, *,
    epoch: int, global_step: int, device: torch.device,
    steps_per_epoch: int, total_steps: int, log_interval: int,
) -> dict[str, Any]:
    model.train()
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    total_nll = interval_nll = 0.0
    target_tokens = interval_targets = 0
    for epoch_step, batch in enumerate(dataloader, start=1):
        batch = {
            key: value.to(device, non_blocking=True)
            for key, value in batch.items()
        }
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(
                input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
            )
            loss, nll, valid_targets = causal_lm_loss(output.logits, batch["labels"])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), training.gradient_clip_norm, error_if_nonfinite=True,
        )
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        batch_nll = float(nll.item())
        total_nll += batch_nll
        target_tokens += valid_targets
        interval_nll += batch_nll
        interval_targets += valid_targets
        if epoch_step % log_interval == 0 or epoch_step == steps_per_epoch:
            print(
                f"epoch={epoch + 1}/{training.epochs} "
                f"epoch_step={epoch_step}/{steps_per_epoch} "
                f"global_step={global_step + epoch_step}/{total_steps} "
                f"loss={interval_nll / interval_targets:.6f} "
                f"lr={optimizer.param_groups[0]['lr']:.2e}",
                flush=True,
            )
            interval_nll, interval_targets = 0.0, 0
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    return {
        "nll_per_target_token": total_nll / target_tokens,
        "elapsed_seconds": elapsed,
        "target_tokens_per_second": target_tokens / elapsed,
    }


def run(args: argparse.Namespace, device: torch.device) -> None:
    config = load_config(args.config)
    training, condition = config.training, config.condition
    vocab_size, seed = args.vocab_size, args.seed
    paths = run_paths(condition, vocab_size, seed)
    snapshot = config.snapshot(vocab_size, seed)
    if args.resume and paths.final.exists():
        raise FileExistsError(f"Run is already complete: {paths.final}")
    if not args.resume:
        require_fresh_run(paths.run, paths.logs, paths.logs.with_name(f"seed_{seed}"))
    runtime = require_training_runtime(config.attention_backend, config.model.attn_pdrop)
    dataset = CachedBabyLMDataset(paths.token_cache)
    tokenizer = load_tokenizer(paths.tokenizer)
    balanced_steps = None
    if condition.variant == "balanced_batch":
        shared_metadata = json.loads(
            (condition_paths(condition).shared_cache(vocab_size) / METADATA_FILE)
            .read_text(encoding="utf-8")
        )
        balanced_steps = math.ceil(shared_metadata["samples"] / training.batch_size)
    dataloader = make_dataloader(
        dataset, batch_size=training.batch_size, shuffle=True, seed=seed,
        num_workers=training.num_workers, steps=balanced_steps,
    )
    sampler = dataloader.batch_sampler
    steps_per_epoch = len(dataloader)
    batch_sizes = sampler.global_batch_sizes
    minimum, maximum = min(batch_sizes), max(batch_sizes)
    batching = {
        "strategy": condition.variant,
        "batch_size_role": "reference" if balanced_steps else "fixed",
        "reference_global_batch_size": training.batch_size,
        "steps_per_epoch": steps_per_epoch,
        "global_batch_size_min": minimum,
        "global_batch_size_max": maximum,
    }
    print(
        f"variant={condition.variant} steps_per_epoch={steps_per_epoch} "
        f"global_batch_size={minimum}-{maximum}", flush=True,
    )
    total_steps = steps_per_epoch * training.epochs
    log_interval = max(1, math.ceil(steps_per_epoch / 20))
    warmup_steps = int(total_steps * training.warmup_fraction)
    signature = {
        **snapshot, "total_steps": total_steps, "warmup_steps": warmup_steps,
        "execution": "single_gpu_v1", "batching": batching,
        "cache": dataset.metadata, "runtime": runtime,
    }
    set_seed(seed)
    model, optimizer, scheduler, actual_backend = initialize_model_and_optimizers(
        paths.last if args.resume else paths.initialization, config,
        vocab_size=vocab_size, device=device, total_steps=total_steps, warmup_steps=warmup_steps,
    )
    runtime["actual_attention_backend"] = actual_backend
    start_epoch = global_step = 0
    if args.resume:
        start_epoch, global_step = load_training_state(
            checkpoint=paths.last, optimizer=optimizer, scheduler=scheduler,
            expected_signature=signature,
        )
    else:
        set_seed(seed)
        for path in (paths.run, paths.records):
            path.mkdir(parents=True, exist_ok=True)
    metrics_path = paths.records / "training_metrics.jsonl"
    records: list[dict[str, Any]] = []
    if args.resume:
        records = [
            record for line in metrics_path.read_text(encoding="utf-8").splitlines()
            if line.strip() and (record := json.loads(line))["epoch"] <= start_epoch
        ]
        metrics_path.write_text(
            "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
            encoding="utf-8",
        )
    else:
        write_yaml_atomic(paths.records / "config.yaml", {
            **snapshot, "execution": "single_gpu_v1", "batching": batching,
            "paths": paths.relative_paths(), "runtime": runtime,
        })
        metrics_path.touch(exist_ok=False)
    for epoch in range(start_epoch, training.epochs):
        sampler.seed = seed + epoch
        metrics = train_epoch(
            training, model, optimizer, scheduler, dataloader,
            epoch=epoch, global_step=global_step, device=device,
            steps_per_epoch=steps_per_epoch, total_steps=total_steps, log_interval=log_interval,
        )
        global_step += steps_per_epoch
        record = {"epoch": epoch + 1, "global_step": global_step, **metrics}
        records.append(record)
        with metrics_path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(record, sort_keys=True) + "\n")
        print(
            f"epoch={epoch + 1}/{training.epochs} complete "
            f"global_step={global_step} loss={metrics['nll_per_target_token']:.6f} "
            f"elapsed={metrics['elapsed_seconds']:.1f}s "
            f"tokens_per_second={metrics['target_tokens_per_second']:.2f}",
            flush=True,
        )
        save_model_artifact(paths.last, model, tokenizer, {
            "completed_epoch": epoch + 1, "global_step": global_step,
            "signature": signature, "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "format": "single_gpu_v1",
            "rng_state": capture_rng_state(device),
        })
    model.config.use_cache = True
    save_model_artifact(paths.final, model, tokenizer)
    training_seconds = sum(float(record["elapsed_seconds"]) for record in records)
    total_targets = int(dataset.metadata["target_tokens"]) * training.epochs
    summary = {
        "condition": condition.name, "variant": condition.variant,
        "vocab_size": vocab_size, "seed": seed,
        "attention_backend": actual_backend,
        "requested_attention_backend": config.attention_backend,
        "attention_dropout": config.model.attn_pdrop, "epochs": training.epochs,
        "optimizer_steps": total_steps, "steps_per_epoch": steps_per_epoch,
        "global_batch_size_min": minimum, "global_batch_size_max": maximum,
        "target_tokens_seen": total_targets, "training_seconds": training_seconds,
        "target_tokens_per_second": total_targets / training_seconds,
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "runtime": runtime, "final_model": str(paths.final.relative_to(PROJECT_ROOT)),
    }
    write_json_atomic(paths.records / "run_summary.json", summary)
    shutil.rmtree(paths.last)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--vocab-size", type=int, choices=VOCAB_SIZES, required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    run(args, torch.device("cuda"))


if __name__ == "__main__":
    main()

