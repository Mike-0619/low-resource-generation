"""Build token caches and dynamically padded causal-LM batches."""

import argparse
import json
import math
from collections.abc import Iterator
from dataclasses import dataclass
from itertools import batched
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, Sampler

from .train_tokenizer import load_tokenizer
from .clean_dataset import sources_for
from ..training.utils import (
    DEFAULT_CONFIG,
    SPECIAL_TOKEN_IDS,
    VOCAB_SIZES,
    ProjectConfig,
    condition_paths,
    load_config,
    staged_directory,
    write_json_atomic,
)


TOKEN_FILE = "tokens.i32"
OFFSETS_FILE = "offsets.i64"
METADATA_FILE = "metadata.json"
IGNORE_INDEX = -100
TOKENIZE_BATCH_SIZE = 1024


def _tokenized_records(tokenizer, path: Path) -> Iterator[tuple[str, list[int]]]:
    with path.open(encoding="utf-8") as source:
        for batch in batched(filter(None, map(str.strip, source)), TOKENIZE_BATCH_SIZE):
            texts = list(batch)
            yield from zip(texts, tokenizer(texts, add_special_tokens=False)["input_ids"])


def _canonical_spans(tokenizer, path: Path, payload_size: int) -> Iterator[str]:
    pending = ""
    pending_tokens = 0
    for text, ids in _tokenized_records(tokenizer, path):
        pending = f"{pending} {text}" if pending else text
        pending_tokens += len(ids)
        while pending_tokens > payload_size:
            encoded = tokenizer(
                pending, add_special_tokens=False, return_offsets_mapping=True
            )
            boundary = encoded["offset_mapping"][payload_size][0]
            cut = pending.rfind(" ", 0, boundary + 1)
            if cut <= 0:
                cut = boundary
                remainder = pending[cut:]
            else:
                remainder = pending[cut + 1:]
            yield pending[:cut]
            pending = remainder
            pending_tokens = len(
                tokenizer(pending, add_special_tokens=False)["input_ids"]
            )
    if pending:
        yield pending


def build_shared_samples(config: ProjectConfig, *, overwrite: bool = False) -> Path:
    paths = condition_paths(config.condition)
    target = paths.samples
    if target.exists() and not overwrite:
        return target
    tokenizer = load_tokenizer(paths.tokenizer(VOCAB_SIZES[0]))
    tokenizer.model_max_length = 10**30
    payload_size = config.training.sequence_length - 2
    source_stats = []
    with staged_directory(target) as staging:
        for source in sources_for(config.condition):
            count = 0
            with (staging / source.filename).open("w", encoding="utf-8") as output:
                for span in _canonical_spans(
                    tokenizer, paths.cleaned / source.filename, payload_size
                ):
                    output.write(span + "\n")
                    count += 1
            source_stats.append({
                "name": source.name,
                "filename": source.filename,
                "samples": count,
            })
        write_json_atomic(staging / METADATA_FILE, {
            "payload_size": payload_size,
            "samples": sum(source["samples"] for source in source_stats),
            "sources": source_stats,
        })
    return target


def _source_offsets(sources: list[dict], payload_size: int) -> list[int]:
    offsets = [0]
    for source in sources:
        first = len(offsets)
        stop = offsets[-1] + source["length"]
        offsets.extend(
            min(position + payload_size, stop)
            for position in range(offsets[-1], stop, payload_size)
        )
        source["samples"] = len(offsets) - first
    return offsets


def build_condition_cache(
    config: ProjectConfig,
    vocab_size: int,
    *,
    overwrite: bool = False,
    shared_samples: Path | None = None,
) -> Path:
    condition = config.condition
    paths = condition_paths(condition)
    target = paths.token_cache(vocab_size)
    mode = condition.variant
    if target.exists() and not overwrite:
        return target

    input_root = paths.cleaned
    if mode == "shared_spans":
        input_root = shared_samples if shared_samples is not None else build_shared_samples(config)
    tokenizer = load_tokenizer(paths.tokenizer(vocab_size))
    tokenizer.model_max_length = 10**30
    payload_size = config.training.sequence_length - 2
    source_stats: list[dict] = []
    offsets = [0]
    with staged_directory(target) as staging:
        with (staging / TOKEN_FILE).open("wb") as output:
            for source in sources_for(condition):
                length = 0
                first_sample = len(offsets)
                path = input_root / source.filename
                for _, ids in _tokenized_records(tokenizer, path):
                    array = np.asarray(ids, dtype="<i4")
                    if mode == "shared_spans" and len(array) > payload_size:
                        raise ValueError(f"Shared span exceeds {payload_size} tokens: {path}")
                    array.tofile(output)
                    length += len(array)
                    if mode == "shared_spans":
                        offsets.append(offsets[-1] + len(array))
                source_stats.append({
                    "name": source.name,
                    "filename": source.filename,
                    "length": length,
                    "samples": len(offsets) - first_sample,
                })
        if mode != "shared_spans":
            offsets = _source_offsets(source_stats, payload_size)
        np.asarray(offsets, dtype="<i8").tofile(staging / OFFSETS_FILE)
        samples = len(offsets) - 1
        write_json_atomic(staging / METADATA_FILE, {
            "payload_size": payload_size,
            "samples": samples,
            "target_tokens": sum(source["length"] for source in source_stats) + samples,
            "sources": source_stats,
        })
    return target


class CachedBabyLMDataset(Dataset[torch.Tensor]):
    def __init__(self, cache_path: str | Path) -> None:
        root = Path(cache_path)
        self.metadata = json.loads((root / METADATA_FILE).read_text(encoding="utf-8"))
        self.tokens = np.memmap(root / TOKEN_FILE, mode="r", dtype="<i4")
        self.offsets = np.memmap(root / OFFSETS_FILE, mode="r", dtype="<i8")

    def __len__(self) -> int:
        return len(self.offsets) - 1

    def __getitem__(self, index: int) -> torch.Tensor:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        start, stop = self.offsets[index : index + 2]
        item = np.empty(stop - start + 2, dtype=np.int64)
        item[0] = SPECIAL_TOKEN_IDS["<s>"]
        item[1:-1] = self.tokens[start:stop]
        item[-1] = SPECIAL_TOKEN_IDS["</s>"]
        return torch.from_numpy(item)


@dataclass
class BatchSampler(Sampler[list[int]]):
    dataset_size: int
    global_batch_size: int
    shuffle: bool
    seed: int
    steps: int | None = None

    def __post_init__(self):
        if self.dataset_size <= 0 or self.global_batch_size <= 0:
            raise ValueError("Dataset and batch size must be positive")
        if self.steps is not None and not 1 <= self.steps <= self.dataset_size:
            raise ValueError("Update count must allow at least one sample per batch")

    @property
    def global_batch_sizes(self) -> list[int]:
        if self.steps is not None:
            size, larger = divmod(self.dataset_size, self.steps)
            return [size + (step < larger) for step in range(self.steps)]
        sizes = [self.global_batch_size] * (len(self) - 1)
        return sizes + [self.dataset_size - sum(sizes)]

    def __iter__(self) -> Iterator[list[int]]:
        indices = (
            torch.randperm(
                self.dataset_size, generator=torch.Generator().manual_seed(self.seed)
            ).tolist()
            if self.shuffle
            else list(range(self.dataset_size))
        )
        start = 0
        for size in self.global_batch_sizes:
            yield indices[start : start + size]
            start += size

    def __len__(self) -> int:
        return self.steps or math.ceil(self.dataset_size / self.global_batch_size)


def collate_causal_lm(
    batch: list[torch.Tensor],
) -> dict[str, torch.Tensor]:
    framed = pad_sequence(
        batch, batch_first=True, padding_value=SPECIAL_TOKEN_IDS["<pad>"]
    )
    input_ids = framed[:, :-1].contiguous()
    labels = framed[:, 1:].clone()
    lengths = torch.tensor([sample.numel() - 1 for sample in batch]).unsqueeze(1)
    attention_mask = torch.arange(input_ids.shape[1]).unsqueeze(0) < lengths
    labels[~attention_mask] = IGNORE_INDEX
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def make_dataloader(
    dataset: Dataset,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
    steps: int | None = None,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_sampler=BatchSampler(
            len(dataset), batch_size, shuffle, seed, steps
        ),
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_causal_lm,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    shared_samples = None
    if config.condition.variant == "shared_spans":
        shared_samples = build_shared_samples(config, overwrite=args.overwrite)
    for vocab_size in VOCAB_SIZES:
        target = build_condition_cache(
            config, vocab_size, overwrite=args.overwrite, shared_samples=shared_samples
        )
        print(f"Ready: {config.condition.name}/vocab_{vocab_size} -> {target}", flush=True)


if __name__ == "__main__":
    main()
