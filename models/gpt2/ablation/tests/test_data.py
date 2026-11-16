import json
import re
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from src.data.data_utils import (
    METADATA_FILE,
    OFFSETS_FILE,
    TOKEN_FILE,
    BatchSampler,
    CachedBabyLMDataset,
    build_condition_cache,
    collate_causal_lm,
    make_dataloader,
)
from src.data.clean_dataset import sources_for
from src.training.utils import configs_for_condition


class TinyTokenizer:
    model_max_length = 1024

    def __call__(self, text, **_kwargs):
        texts = text if isinstance(text, list) else [text]
        matches = [list(re.finditer(r"\S+", item)) for item in texts]
        encoded = [[10] * len(items) for items in matches]
        result = {"input_ids": encoded if isinstance(text, list) else encoded[0]}
        if _kwargs.get("return_offsets_mapping"):
            offsets = [[item.span() for item in items] for items in matches]
            result["offset_mapping"] = offsets if isinstance(text, list) else offsets[0]
        return result


def _build_cache(config, root: Path, texts: list[str]) -> Path:
    cleaned, cache = root / "cleaned", root / "cache"
    cleaned.mkdir()
    for source, text in zip(sources_for(config.condition), texts):
        (cleaned / source.filename).write_text(text, encoding="utf-8")
    paths = SimpleNamespace(
        cleaned=cleaned,
        tokenizer=lambda _size: root / "tokenizer",
        token_cache=lambda _size: cache,
        samples=root / "samples",
    )
    with (
        patch("src.data.data_utils.condition_paths", return_value=paths),
        patch("src.data.data_utils.load_tokenizer", return_value=TinyTokenizer()),
    ):
        build_condition_cache(config, 512)
    return cache


class DataTest(unittest.TestCase):
    def test_cache_continuous_stream_offsets_and_dynamic_padding(self) -> None:
        config = configs_for_condition("10m_en_balanced")[0]
        sources = sources_for(config.condition)
        texts = [
            " ".join(["a"] * 300) + "\n" + " ".join(["b"] * 250) + "\n",
            "one two\n\nthree four\n",
            *("one two three\n" for _ in sources[2:]),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            cache = _build_cache(config, Path(temporary), texts)
            metadata = json.loads((cache / METADATA_FILE).read_text())
            tokens = np.fromfile(cache / TOKEN_FILE, dtype="<i4")
            offsets = np.fromfile(cache / OFFSETS_FILE, dtype="<i8")
            self.assertEqual(int(np.count_nonzero(tokens == 2)), 0)
            self.assertEqual(offsets.tolist(), [0, 510, 550, 554, 557, 560, 563, 566])
            self.assertEqual((metadata["samples"], metadata["target_tokens"]), (7, 573))
            self.assertEqual(tokens[:550].tolist(), [10] * 550)

            dataset = CachedBabyLMDataset(cache)
            self.assertEqual(dataset[2].tolist(), [1, 10, 10, 10, 10, 2])
            batch = collate_causal_lm([
                dataset[2],
                torch.tensor([1, 20, 2]),
            ])
            self.assertEqual(
                batch["labels"][1].tolist(), [20, 2, -100, -100, -100]
            )

        german = configs_for_condition("100m_de_balanced")[0]
        with tempfile.TemporaryDirectory() as temporary:
            sources = sources_for(german.condition)
            cache = _build_cache(
                german,
                Path(temporary),
                ["one two\n\nthree four\n"] * len(sources),
            )
            metadata = json.loads((cache / METADATA_FILE).read_text())
            tokens = np.fromfile(cache / TOKEN_FILE, dtype="<i4")
            offsets = np.fromfile(cache / OFFSETS_FILE, dtype="<i8")
            self.assertEqual(metadata["samples"], len(sources))
            self.assertEqual(np.diff(offsets).tolist(), [4] * len(sources))
            self.assertEqual(
                int(np.count_nonzero(tokens == 2)),
                0,
            )

        spans = configs_for_condition("10m_en_spans")[0]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sources = sources_for(spans.condition)
            cache = _build_cache(
                spans,
                root,
                [" ".join(f"word{i}" for i in range(600))]
                + ["one two\n"] * (len(sources) - 1),
            )
            sample_lines = (root / "samples" / sources[0].filename).read_text().splitlines()
            self.assertEqual([len(line.split()) for line in sample_lines], [510, 90])
            self.assertEqual(len(CachedBabyLMDataset(cache)), 2 + len(sources) - 1)
            self.assertFalse(bool((np.fromfile(cache / TOKEN_FILE, dtype="<i4") == 2).any()))

    def test_batches_cover_samples_once_with_deterministic_order(self):
        for size, steps, expected_sizes in (
            (1, None, [1]), (16, None, [16]), (17, None, [16, 1]),
            (32, None, [16, 16]), (17, 3, [6, 6, 5]),
        ):
            for shuffle in (False, True):
                sampler = BatchSampler(size, 16, shuffle, 42, steps)
                batches = list(sampler)
                self.assertEqual(sampler.global_batch_sizes, expected_sizes)
                self.assertEqual([len(batch) for batch in batches], expected_sizes)
                self.assertEqual(len(sampler),len(expected_sizes))
                order = [item for batch in batches for item in batch]
                expected = torch.randperm(size,generator=torch.Generator().manual_seed(42)).tolist() if shuffle else list(range(size))
                self.assertEqual(order,expected)
                self.assertEqual(sorted(order),list(range(size)))
                self.assertEqual(list(sampler),batches)
        for args in ((0,16,False,42,None),(1,16,False,42,3)):
            with self.assertRaises(ValueError):BatchSampler(*args)
        samples = [torch.tensor([1, index+10, 2]) for index in range(17)]
        batches = list(make_dataloader(samples,batch_size=16,shuffle=False,seed=42,num_workers=0))
        self.assertEqual([len(batch['input_ids']) for batch in batches],[16,1])
