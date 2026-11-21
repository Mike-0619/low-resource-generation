"""Source-local chunks, cache reuse, and causal-LM masks."""
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from transformers import PreTrainedTokenizerFast

from src.data import data_utils
from src.data.train_tokenizer import tokenizer_backend, tokenizer_trainer


class DataUtilsTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(dir=os.environ.get("XDG_CACHE_HOME"))
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.data = self.root / "cleaned"
        self.data.mkdir()
        self.cache = self.root / "tokens"
        backend = tokenizer_backend()
        trainer = tokenizer_trainer(512)
        backend.train_from_iterator(
            ["A test.\nGerman: Ä, Straße.\n*CHİ: yıl, şeker, gönül.\n A slash! \n / \r\n é."]
            * 2,
            trainer,
        )
        self.tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=backend, bos_token="<s>", eos_token="</s>", unk_token="<unk>",
            pad_token="<pad>",
        )

    def build(self, sequence_length=9):
        return data_utils.CachedBabyLMDataset(self.data, self.tokenizer, sequence_length, self.cache)

    def test_chunking_keeps_source_order_tails_and_special_tokens(self):
        (self.data / "b.train.txt").write_bytes(b"Two! \rNotes.\r\n\n*TITLE: trail\r\n")
        (self.data / "a.train.txt").write_text(
            (" A\u0301! \n  \n/ note <s> <pad>\n*CHİ: yıl.\nÄrger\n >hello\r\nA plain line.\n" * 3)
            + "END",
            encoding="utf-8",
        )
        (self.data / "empty.train.txt").touch()
        expected = []
        for path in sorted(self.data.glob("*.train.txt")):
            ids = self.tokenizer(path.read_text(encoding="utf-8"), add_special_tokens=False)["input_ids"]
            expected.extend([1, *ids[start:start + 7], 2] for start in range(0, len(ids), 7))
        with patch.object(data_utils, "CHUNK_BYTES", 46):
            actual = self.build()
        self.assertEqual([item.tolist() for item in actual], expected)
        self.assertEqual(actual[-1].tolist(), expected[-1])
        metadata = json.loads((self.cache / actual.cache_id / "metadata.json").read_text())
        self.assertEqual(metadata["samples"], len(expected))
        self.assertEqual(metadata["sources"][-1], {"file": "empty.train.txt", "tokens": 0, "samples": 0})

    def test_reuses_cache_and_invalidates_changed_inputs(self):
        path = self.data / "a.train.txt"
        path.write_text("First line.\nA second line.", encoding="utf-8")
        first = self.build()
        with patch.object(data_utils, "_write_cache", side_effect=AssertionError("cache was rebuilt")):
            cached = self.build()
        self.assertEqual(cached.cache_id, first.cache_id)
        self.assertEqual([item.tolist() for item in cached], [item.tolist() for item in first])
        self.assertNotEqual(self.build(8).cache_id, first.cache_id)
        path.write_text("First line.\nA new second line.", encoding="utf-8")
        updated = self.build()
        self.assertNotEqual(updated.cache_id, first.cache_id)
        backend = tokenizer_backend()
        trainer = tokenizer_trainer(256)
        backend.train_from_iterator(["First line. A new second line."] * 2, trainer)
        self.tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=backend, bos_token="<s>", eos_token="</s>", unk_token="<unk>",
            pad_token="<pad>",
        )
        self.assertNotEqual(self.build().cache_id, updated.cache_id)

    def test_interrupted_build_is_not_published(self):
        (self.data / "a.train.txt").write_text("First! \nNext.", encoding="utf-8")
        (self.data / "b.train.txt").write_text("Z" * 100, encoding="utf-8")
        with patch.object(data_utils, "CHUNK_BYTES", 32):
            with self.assertRaisesRegex(ValueError, "No safe tokenizer split"):
                self.build()
            self.assertFalse(any(entry.is_dir() for entry in self.cache.iterdir()))
            (self.data / "b.train.txt").write_text("Repaired.\nDone.", encoding="utf-8")
            self.assertGreater(len(self.build()), 0)

    def test_rejects_truncated_cache(self):
        (self.data / "a.train.txt").write_text("First line.\nA second line.", encoding="utf-8")
        cached = self.build()
        (self.cache / cached.cache_id / "tokens.i32").write_bytes(b"")
        with self.assertRaisesRegex(ValueError, "Incomplete|incomplete"):
            self.build()

    def test_workspace_cache_rejects_override_and_symlink_escape(self):
        workspace = self.root / "workspace"
        helper = workspace / ".agents/config/cache/env.sh"
        helper.parent.mkdir(parents=True)
        helper.touch()
        root = workspace / "Projects/low-resource-generation/models/gpt2/baseline"
        root.mkdir(parents=True)
        with patch.dict(os.environ, XDG_CACHE_HOME="", WORKSPACE_ROOT=str(self.root / "unused")):
            cache_path = data_utils.token_cache_directory(root, "English", "10M")
            self.assertTrue(cache_path.is_relative_to(workspace / ".cache"))
        with patch.dict(os.environ, XDG_CACHE_HOME=str(self.root / "elsewhere")):
            with self.assertRaisesRegex(ValueError, "must be inside"):
                data_utils.token_cache_directory(root, "English", "10M")
        (workspace / ".cache").mkdir()
        destination = self.root / "elsewhere"
        destination.mkdir()
        (workspace / ".cache/projects").symlink_to(destination)
        with patch.dict(os.environ, XDG_CACHE_HOME=""):
            with self.assertRaisesRegex(ValueError, "must be inside"):
                data_utils.token_cache_directory(root, "English", "10M")

    def test_padding_and_input_based_mask(self):
        batch = [torch.tensor([1, 5, 6, 2]), torch.tensor([1, 7, 2]), torch.tensor([1, 2, 8, 2])]
        inputs, targets, mask = data_utils.collate_batch(batch, 2)
        self.assertEqual(inputs.tolist(), [[1, 5, 6], [1, 7, 2], [1, 2, 8]])
        self.assertEqual(targets.tolist(), [[5, 6, 2], [7, 2, 2], [2, 8, 2]])
        self.assertEqual(mask.tolist(), [[True, True, True], [True, True, False], [True, False, True]])
