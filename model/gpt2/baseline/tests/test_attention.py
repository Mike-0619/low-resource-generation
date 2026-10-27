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


from src import attention, data_utils, models, training, utils

class AttentionTest(unittest.TestCase):
    def runtime_mocks(self, capability=(8, 0)):
        stack = contextlib.ExitStack()
        for name, value in {
            "is_available": True, "is_bf16_supported": True,
            "get_device_capability": capability, "get_device_name": "mock GPU",
        }.items():
            stack.enter_context(patch(f"src.attention.torch.cuda.{name}", return_value=value))
        return stack

    def test_sdpa_cpu_forward_without_fa3_and_without_forcing_kernel(self):
        with patch("src.attention.require_local_fa3", side_effect=AssertionError("FA3 must not be queried")):
            cfg, _ = utils.load_experiment('configs/English/10M.yaml')
            cfg['model'].update(n_layer=1, n_embd=32, n_head=2, n_positions=16, vocab_size=32)
            model = GPT2LMHeadModel(attention.configure_model_attention(models.model_config(cfg), 'sdpa'))
            inputs, targets, mask = data_utils.collate_batch(
                [torch.tensor([1, 5, 6, 2]), torch.tensor([1, 7, 2])], 2)
            output = model(inputs, labels=inputs)
            loss = training.masked_loss(output.logits, targets, mask)
            self.assertTrue(torch.isfinite(output.loss))
            self.assertTrue(torch.isfinite(loss))
            # Exercise both the built-in loss and the baseline masked loss.
            for objective in (output.loss, loss):
                model.zero_grad(set_to_none=True)
                objective.backward(retain_graph=True)
                self.assertTrue(all(torch.isfinite(p.grad).all()
                                    for p in model.parameters() if p.grad is not None))
            self.assertEqual(model.config._attn_implementation, 'sdpa')
            self.assertEqual(attention.require_model_backend(model, "sdpa"), "sdpa")
            self.assertEqual(model.transformer.h[0].attn.attn_dropout.p, .1)
            with self.runtime_mocks():
                runtime = attention.require_training_runtime("sdpa", .1)
            self.assertEqual(runtime["compute_capability"], "8.0")
            self.assertNotIn("fa3_version", runtime)
            self.assertEqual(runtime["attention_dropout"], .1)
            self.assertEqual(yaml.safe_load(yaml.safe_dump(runtime)), runtime)

    def test_invalid_config_precedes_data_and_model_loading(self):
        with patch("src.attention.torch.cuda.is_available") as cuda:
            for backend, dropout in [("eager", .1), ("flash_attention_3", .1)]:
                with self.assertRaises(ValueError):
                    attention.require_training_runtime(backend, dropout)
            cuda.assert_not_called()
        cfg, _ = utils.load_experiment("configs/English/10M.yaml")
        cfg["attention_backend"] = "flash_attention_3"
        with tempfile.TemporaryDirectory() as tmp, patch("src.training.AutoTokenizer.from_pretrained") as tokenizer, patch("src.training.GPT2LMHeadModel") as model:
            with self.assertRaisesRegex(ValueError, "dropout 0.0"):
                training.train(cfg, Path(tmp))
            tokenizer.assert_not_called()
            model.assert_not_called()

    def test_fa3_checks_local_package_hardware_and_actual_backend(self):
        with self.runtime_mocks():
            with self.assertRaisesRegex(RuntimeError, "Hopper"):
                attention.require_training_runtime("flash_attention_3", 0.)
        with self.runtime_mocks((9, 0)), patch("src.attention.is_flash_attn_3_available", return_value=False) as available:
            with self.assertRaisesRegex(RuntimeError, "Hub fallback is disabled"):
                attention.require_training_runtime("flash_attention_3", 0.)
            available.assert_called_once_with(kernels_fallback_ok=False)
        with self.runtime_mocks((9, 0)), patch("src.attention.is_flash_attn_3_available", return_value=True), patch("src.attention.importlib.metadata.version", return_value="mock-version"):
            runtime = attention.require_training_runtime("flash_attention_3", 0.)
            self.assertEqual(runtime["fa3_version"], "mock-version")
            config = attention.configure_model_attention(GPT2Config(attn_pdrop=0.), "flash_attention_3")
            self.assertEqual(attention.require_model_backend(SimpleNamespace(config=config), "flash_attention_3"), "flash_attention_3")
        with self.assertRaisesRegex(RuntimeError, "fallback detected"):
            attention.require_model_backend(SimpleNamespace(config=GPT2Config(attn_implementation="sdpa")), "flash_attention_3")

    def test_training_requires_cuda_and_bf16_for_both_backends(self):
        for backend, dropout in [("sdpa", .1), ("flash_attention_3", 0.)]:
            with patch("src.attention.torch.cuda.is_available", return_value=False):
                with self.assertRaisesRegex(RuntimeError, "CUDA"):
                    attention.require_training_runtime(backend, dropout)
            with self.runtime_mocks(), patch("src.attention.torch.cuda.is_bf16_supported", return_value=False):
                with self.assertRaisesRegex(RuntimeError, "BF16"):
                    attention.require_training_runtime(backend, dropout)
