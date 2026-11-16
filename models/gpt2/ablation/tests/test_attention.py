"""Backend contracts without a GPU or an installed FA3 package."""

import contextlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import yaml
from transformers import GPT2Config, GPT2LMHeadModel

from src.models import attention
from src.training import train as training
from src.training.utils import configs_for_condition


class AttentionTest(unittest.TestCase):
    def runtime_mocks(self, capability=(8, 0)):
        stack = contextlib.ExitStack()
        for name, value in {
            "is_available": True, "is_bf16_supported": True,
            "get_device_capability": capability, "get_device_name": "mock GPU",
        }.items():
            stack.enter_context(patch(f"src.models.attention.torch.cuda.{name}", return_value=value))
        return stack

    def test_sdpa_cpu_forward_without_fa3_and_without_forcing_kernel(self):
        with patch("src.models.attention.require_local_fa3", side_effect=AssertionError("FA3 must not be queried")):
            config = GPT2Config(vocab_size=32, n_layer=1, n_head=2, n_embd=16,
                                n_positions=16, bos_token_id=1, eos_token_id=2,
                                attn_pdrop=.1, resid_pdrop=.1, embd_pdrop=.1)
            model = GPT2LMHeadModel(attention.configure_model_attention(config, "sdpa"))
            output = model(torch.tensor([[1, 2, 3]]), labels=torch.tensor([[1, 2, 3]]))
            output.loss.backward()
            self.assertTrue(torch.isfinite(output.loss))
            self.assertEqual(attention.require_model_backend(model, "sdpa"), "sdpa")
            self.assertEqual(model.transformer.h[0].attn.attn_dropout.p, .1)
            with self.runtime_mocks():
                runtime = attention.require_training_runtime("sdpa", .1)
            self.assertEqual(runtime["compute_capability"], "8.0")
            self.assertNotIn("fa3_version", runtime)
            self.assertEqual(runtime["attention_dropout"], .1)
            self.assertEqual(yaml.safe_load(yaml.safe_dump(runtime)), runtime)

    def test_invalid_config_precedes_data_and_model_loading(self):
        with patch("src.models.attention.torch.cuda.is_available") as cuda:
            for backend, dropout in [("eager", .1), ("flash_attention_3", .1)]:
                with self.assertRaises(ValueError):
                    attention.require_training_runtime(backend, dropout)
            cuda.assert_not_called()
        from dataclasses import replace
        config = configs_for_condition("10m_en_spans")[0]
        with self.assertRaisesRegex(ValueError, "dropout 0.0"):
            replace(config, attention_backend="flash_attention_3")
        # Even a configuration supplied directly to run must fail before data allocation.
        invalid = SimpleNamespace(**vars(config))
        invalid.attention_backend = "flash_attention_3"
        invalid.snapshot = config.snapshot
        args = SimpleNamespace(config="unused", vocab_size=512, seed=42, resume=False)
        with tempfile.TemporaryDirectory() as tmp:
            paths = SimpleNamespace(run=Path(tmp)/"results", logs=Path(tmp)/"logs")
            device = torch.device("cpu")
            with patch("src.training.train.load_config", return_value=invalid), patch("src.training.train.run_paths", return_value=paths), patch("src.training.train.CachedBabyLMDataset") as dataset, patch("src.training.train.initialize_model_and_optimizers") as model:
                with self.assertRaisesRegex(ValueError, "dropout 0.0"):
                    training.run(args, device)
                dataset.assert_not_called()
                model.assert_not_called()

    def test_fa3_checks_local_package_hardware_and_actual_backend(self):
        with self.runtime_mocks():
            with self.assertRaisesRegex(RuntimeError, "Hopper"):
                attention.require_training_runtime("flash_attention_3", 0.)
        with self.runtime_mocks((9, 0)), patch("src.models.attention.is_flash_attn_3_available", return_value=False) as available:
            with self.assertRaisesRegex(RuntimeError, "Hub fallback is disabled"):
                attention.require_training_runtime("flash_attention_3", 0.)
            available.assert_called_once_with(kernels_fallback_ok=False)
        with self.runtime_mocks((9, 0)), patch("src.models.attention.is_flash_attn_3_available", return_value=True), patch("src.models.attention.importlib.metadata.version", return_value="mock-version"):
            runtime = attention.require_training_runtime("flash_attention_3", 0.)
            self.assertEqual(runtime["fa3_version"], "mock-version")
            config = attention.configure_model_attention(GPT2Config(attn_pdrop=0.), "flash_attention_3")
            self.assertEqual(attention.require_model_backend(SimpleNamespace(config=config), "flash_attention_3"), "flash_attention_3")
        with self.assertRaisesRegex(RuntimeError, "fallback detected"):
            attention.require_model_backend(SimpleNamespace(config=GPT2Config(attn_implementation="sdpa")), "flash_attention_3")

    def test_training_requires_cuda_and_bf16_for_both_backends(self):
        for backend, dropout in [("sdpa", .1), ("flash_attention_3", 0.)]:
            with patch("src.models.attention.torch.cuda.is_available", return_value=False):
                with self.assertRaisesRegex(RuntimeError, "CUDA"):
                    attention.require_training_runtime(backend, dropout)
            with self.runtime_mocks(), patch("src.models.attention.torch.cuda.is_bf16_supported", return_value=False):
                with self.assertRaisesRegex(RuntimeError, "BF16"):
                    attention.require_training_runtime(backend, dropout)
