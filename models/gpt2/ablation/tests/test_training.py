import contextlib
import json
import random
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import torch
import torch.nn.functional as F

from src.training.checkpoint import capture_rng_state, load_training_state, restore_rng_state, save_model_artifact
from src.training.train import causal_lm_loss, train_epoch, run as run_training
from src.training.utils import configs_for_condition
from src.data.data_utils import make_dataloader
from tests.support import DummyModel, DummyTokenizer


class TrainingTest(unittest.TestCase):
    def test_masked_loss_and_optimizer_update(self):
        torch.manual_seed(7)
        model = torch.nn.Linear(3, 5)
        inputs = torch.randn(2, 4, 3)
        labels = torch.tensor([[1, 2, -100, -100], [3, 4, 0, -100]])
        logits = model(inputs)
        loss, nll, count = causal_lm_loss(logits, labels)
        expected = F.cross_entropy(logits.reshape(-1, 5), labels.reshape(-1), ignore_index=-100)
        torch.testing.assert_close(loss, expected)
        self.assertEqual(count, 5)
        torch.testing.assert_close(nll, expected.detach() * 5)
        weight = model.weight.detach().clone()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
        loss.backward()
        optimizer.step()
        self.assertFalse(torch.equal(weight, model.weight))
        with self.assertRaisesRegex(ValueError, 'no target tokens'):
            causal_lm_loss(logits, torch.full_like(labels, -100))

    def test_epoch_updates_for_both_batching_routes(self):
        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(32, 8)
                self.head = torch.nn.Linear(8, 32)
            def forward(self, input_ids, attention_mask):
                return SimpleNamespace(logits=self.head(self.embedding(input_ids)))
        samples = [torch.tensor([1, 10+i, 2]) for i in range(17)]
        config = configs_for_condition('10m_en_spans')[0]
        for route, steps, expected_updates in [('shared_spans', None, 2), ('balanced_batch', 3, 3)]:
            with self.subTest(route=route):
                model = TinyModel()
                old = model.head.weight.detach().clone()
                optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
                scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
                with patch('src.training.train.torch.cuda.synchronize'), patch('src.training.train.torch.autocast', return_value=contextlib.nullcontext()), patch('builtins.print'):
                    loader = make_dataloader(samples, batch_size=16, shuffle=True, seed=42, num_workers=0, steps=steps)
                    metrics = train_epoch(
                        config.training, model, optimizer, scheduler, loader,
                        epoch=0, global_step=0, device=torch.device("cpu"),
                        steps_per_epoch=expected_updates,
                        total_steps=expected_updates * config.training.epochs, log_interval=1,
                    )
                self.assertEqual(scheduler.last_epoch, expected_updates)
                self.assertFalse(torch.equal(old,model.head.weight))
                self.assertTrue(np.isfinite(metrics['nll_per_target_token']))
                self.assertGreater(metrics['target_tokens_per_second'],0)

    def test_rng_restoration_and_legacy_rejection(self):
        random.seed(42); np.random.seed(42); torch.manual_seed(42)
        state = capture_rng_state(torch.device('cpu'))
        expected = (random.random(),np.random.rand(),torch.rand(3))
        restore_rng_state(state)
        actual = (random.random(),np.random.rand(),torch.rand(3))
        self.assertEqual(expected[:2],actual[:2])
        torch.testing.assert_close(expected[2],actual[2])
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            torch.save({'signature':{},'optimizer':{},'scheduler':{},'legacy_rng_states':[state,state]},root/'training_state.pt')
            optimizer=MagicMock();scheduler=MagicMock()
            with self.assertRaisesRegex(ValueError,'Unsupported legacy checkpoint'):
                load_training_state(checkpoint=root,optimizer=optimizer,scheduler=scheduler,expected_signature={})
            optimizer.load_state_dict.assert_not_called()
            scheduler.load_state_dict.assert_not_called()

    def test_checkpoint_rng_signature_and_resume_metrics(self) -> None:
        config = configs_for_condition("10m_en_spans")[0]
        signature = {"seed": 42, "attention_backend": "sdpa", "model": {"attn_pdrop": .1}}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parameters = torch.nn.Linear(2, 2)
            optimizer = torch.optim.AdamW(parameters.parameters(), lr=5e-5)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
            checkpoint = root / "last"
            save_model_artifact(checkpoint, DummyModel(), DummyTokenizer(), {
                "completed_epoch": 3,
                "global_step": 11,
                "signature": signature,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "format": "single_gpu_v1",
                "rng_state": capture_rng_state(torch.device("cpu")),
            })
            self.assertEqual(
                load_training_state(
                    checkpoint=checkpoint,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    expected_signature=signature,
                ),
                (3, 11),
            )
            with self.assertRaises(ValueError):
                load_training_state(
                    checkpoint=checkpoint,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    expected_signature={**signature, "seed": 43},
                )

            for mismatch in ({**signature, "attention_backend": "flash_attention_3"},
                             {**signature, "model": {"attn_pdrop": 0.0}}):
                with self.assertRaisesRegex(ValueError, "signature"):
                    load_training_state(checkpoint=checkpoint, optimizer=optimizer,
                                        scheduler=scheduler, expected_signature=mismatch)

            resume_config = replace(config, training=replace(config.training, epochs=4))
            run_root = root / "results/shared_spans/vocab_512/seed_42"
            paths = SimpleNamespace(
                initialization=root / "initialization",
                tokenizer=root / "tokenizer",
                token_cache=root / "cache",
                run=run_root,
                last=run_root / "checkpoints/last",
                final=run_root / "final",
                logs=root / "logs/shared_spans/train",
                records=run_root / "records",
            )
            paths.last.mkdir(parents=True)
            paths.records.mkdir(parents=True)
            metrics_path = paths.records / "training_metrics.jsonl"
            metrics_path.write_text("".join(
                json.dumps({
                    "epoch": epoch,
                    "global_step": 999 if epoch == 3 else epoch * 2,
                    "nll_per_target_token": 1.0,
                    "elapsed_seconds": 1.0,
                    "target_tokens_per_second": 5.0,
                }) + "\n"
                for epoch in (1, 2, 3)
            ))
            dataset = MagicMock(metadata={"target_tokens": 5})
            dataset.__len__.return_value = 17
            training_model = torch.nn.Linear(2, 2)
            training_model.config = SimpleNamespace(use_cache=False, _attn_implementation="sdpa")
            epoch_metrics = {
                "nll_per_target_token": 1.0,
                "elapsed_seconds": 1.0,
                "target_tokens_per_second": 5.0,
            }
            observed_loaders = []

            def record_epoch(*args, **kwargs):
                observed_loaders.append((id(args[4]), args[4].batch_sampler.seed))
                return epoch_metrics

            epoch_runner = MagicMock(side_effect=record_epoch)
            with (
                patch("src.training.train.load_config", return_value=resume_config),
                patch("src.training.train.run_paths", return_value=paths),
                patch("src.training.train.PROJECT_ROOT", root),
                patch("src.training.train.require_training_runtime", return_value={}),
                patch("src.training.train.CachedBabyLMDataset", return_value=dataset),
                patch("src.training.train.load_tokenizer", return_value=DummyTokenizer()),
                patch(
                    "src.training.train.initialize_model_and_optimizers",
                    return_value=(training_model, MagicMock(), MagicMock(), "sdpa"),
                ),
                patch("src.training.train.load_training_state", return_value=(2, 4)),
                patch("src.training.train.train_epoch", epoch_runner),
                patch("src.training.train.make_dataloader", wraps=make_dataloader) as loader_builder,
                patch("src.training.train.capture_rng_state", return_value={}),
                patch("src.training.train.save_model_artifact"),
                patch("builtins.print") as print_mock,
            ):
                run_training(
                    SimpleNamespace(config="unused", vocab_size=512, seed=42, resume=True),
                    torch.device("cpu"),
                )
            loader_builder.assert_called_once()
            self.assertEqual([seed for _, seed in observed_loaders], [44, 45])
            self.assertEqual(len({loader for loader, _ in observed_loaders}), 1)
            self.assertEqual(
                [(call.kwargs["epoch"], call.kwargs["global_step"])
                 for call in epoch_runner.call_args_list],
                [(2, 4), (3, 6)],
            )
            self.assertEqual(
                [
                    call.args[0].split()[0]
                    for call in print_mock.call_args_list
                    if call.args[0].startswith("epoch=")
                ],
                ["epoch=3/4", "epoch=4/4"],
            )
            records = [json.loads(line) for line in metrics_path.read_text().splitlines()]
            self.assertEqual(
                [(record["epoch"], record["global_step"]) for record in records],
                [(1, 2), (2, 4), (3, 6), (4, 8)],
            )
