"""Training recipe, checkpoints, artifact protection and training jobs."""
import json
import os
import shlex
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from contextlib import ExitStack, nullcontext
from unittest.mock import MagicMock, patch

import torch
import yaml
from transformers import AutoTokenizer, GPT2LMHeadModel, PreTrainedTokenizerFast

from src import attention, data_utils, models, train_tokenizer, training, utils

from src.utils import PROJECT_ROOT, require_fresh_run

CONDITIONS = (('English', '10M'), ('English', '100M'),
              ('German', '10M'), ('German', '108M'))


class TrainingTest(unittest.TestCase):
    def test_masked_loss_and_gradient(self):
        batch = [torch.tensor([1, 5, 6, 2]), torch.tensor([1, 7, 2]), torch.tensor([1, 2, 8, 2])]
        actual = data_utils.collate_batch(batch, 2)
        torch.manual_seed(42)
        logits = torch.randn(3, 3, 12, requires_grad=True)
        targets, mask = actual[1:]
        selected = torch.gather(torch.nn.functional.log_softmax(logits, dim=2), 2, targets.unsqueeze(2)).squeeze(2)
        expected_loss = -torch.sum(selected * mask) / torch.sum(mask)
        actual_loss = training.masked_loss(logits, targets, mask)
        torch.testing.assert_close(actual_loss, expected_loss)
        torch.testing.assert_close(torch.autograd.grad(actual_loss, logits, retain_graph=True)[0], torch.autograd.grad(expected_loss, logits)[0])

    def test_schedule_and_word_milestones(self):
        # Fixed examples cover rounding, colliding milestones and the last reachable step.
        cases = [
            ((1, 10_000_000, 1), {}),
            ((3, 10_000_000, 1), {1: '6M', 2: '10M'}),
            ((17, 10_000_000, 1), {1: '1M', 3: '2M', 5: '3M', 6: '4M', 8: '5M',
                                    10: '6M', 11: '7M', 13: '8M', 15: '9M', 16: '10M'}),
            ((10, 100_000_000, 1), {1: '10M', 2: '20M', 3: '30M', 4: '40M', 5: '50M',
                                     6: '60M', 7: '70M', 8: '80M', 9: '100M'}),
            ((1, 100_000_000, 10), {1: '100M', 2: '200M', 3: '300M', 4: '400M', 5: '500M',
                                     6: '600M', 7: '700M', 8: '800M', 9: '1B'}),
        ]
        for args, expected in cases:
            with self.subTest(checkpoints=args):
                self.assertEqual(training.checkpoint_steps(*args), expected)
        for steps, expected in [(1, (10, 0)), (17, (170, 1)), (1818, (18180, 181)), (30000, (300000, 3000))]:
            with self.subTest(schedule=steps):
                self.assertEqual(training.schedule_steps(steps, 10, .01), expected)

    def test_configs_and_independent_output_paths(self):
        all_paths = []
        for language, size in CONDITIONS:
            with self.subTest(language=language, size=size):
                cfg, root = utils.load_experiment(f'configs/{language}/{size}.yaml')
                model = models.model_config(cfg)
                self.assertEqual((model.n_layer, model.n_embd, model.n_head, model.n_positions, model.vocab_size), (12,768,12,1024,16384))
                self.assertEqual((model.attn_pdrop, model.resid_pdrop, model.embd_pdrop), (.1,.1,.1))
                self.assertEqual(cfg['training']['seed'], 42)
                expected_words = (cfg['dataset']['expected_whitespace_tokens']
                                  if (language, size) == ('German', '10M') else
                                  {'10M': 10_000_000, '100M': 100_000_000, '108M': 107_910_839}[size])
                self.assertEqual(cfg['training']['words_per_epoch'], expected_words)
                if (language, size) == ('German', '10M'):
                    self.assertEqual(cfg['dataset']['sampling'], 'category_word_ten_percent_prefix_v1')
                    self.assertEqual(expected_words, 10_791_084)
                self.assertEqual(cfg['attention_backend'], 'sdpa')
                self.assertNotIn('ablation', root.parts)
                self.assertEqual(root, PROJECT_ROOT)
                paths = utils.experiment_paths(cfg, root)
                self.assertEqual(paths.cleaned, root / 'data' / language / size / 'cleaned')
                self.assertEqual(paths.tokenizer, root / 'tokenizer' / language / size)
                self.assertEqual(paths.results, root / 'results' / language / size / 'seed_42')
                self.assertEqual(paths.logs, root / 'logs' / language / size / 'train')
                all_paths.append(paths)
        for field in ('cleaned', 'tokenizer', 'results', 'logs'):
            self.assertEqual(len({getattr(p, field) for p in all_paths}), 4)
        self.assertFalse((PROJECT_ROOT / 'configs/English/108M.yaml').exists())
        for language, size in [('English', '108M'), ('German', '100M')]:
            cfg['dataset'].update(size=size, language=language)
            with self.assertRaisesRegex(ValueError, 'Unsupported baseline language/size'):
                utils.experiment_paths(cfg, root)

    def test_config_location_does_not_determine_output_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            for language, size in CONDITIONS:
                cfg, root = utils.load_experiment(f'configs/{language}/{size}.yaml')
                path = Path(tmp) / 'custom-name.yaml'
                cfg['protocol'] = 'local_reproduction'
                path.write_text(yaml.safe_dump(cfg))
                with self.subTest(language=language, size=size):
                    loaded, actual_root = utils.load_experiment(path)
                    self.assertEqual(loaded, cfg)
                    self.assertEqual(actual_root, root)
                    self.assertEqual(utils.experiment_paths(loaded, actual_root),
                                     utils.experiment_paths(cfg, root))


    def test_training_uses_sibling_results_and_logs_before_loading_data(self):
        cfg, _ = utils.load_experiment('configs/English/10M.yaml')
        root = Path('/unused/experiment')
        with patch.dict(os.environ, WORLD_SIZE='1'), \
             patch('src.training.require_training_runtime', return_value={}), \
             patch('src.training.require_fresh_run', side_effect=FileExistsError('stop')) as guard, \
             patch('src.training.AutoTokenizer.from_pretrained') as tokenizer:
            with self.assertRaisesRegex(FileExistsError, 'stop'):
                training.train(cfg, root)
            guard.assert_called_once_with(root / 'results/English/10M/seed_42', root / 'logs/English/10M/train')
            tokenizer.assert_not_called()

    def test_checkpoint_roundtrip_and_overwrite_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = torch.nn.Linear(2, 2)
            optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
            target = Path(tmp) / 'epoch_0'
            models.save_checkpoint(target, model, optimizer, scheduler)
            self.assertEqual(set(p.name for p in target.iterdir()), {'latest_student.pt','latest_optimizer.pt','latest_scheduler.pt'})
            state = torch.load(target / 'latest_student.pt', weights_only=True)
            for name, value in model.state_dict().items():
                torch.testing.assert_close(state[name], value)
            with self.assertRaises(FileExistsError):
                models.save_checkpoint(target, model, optimizer, scheduler)

    def test_final_export_is_loadable_model_and_tokenizer(self):
        cfg, _ = utils.load_experiment('configs/English/10M.yaml')
        backend, trainer = train_tokenizer.tokenizer_backend(64)
        backend.train_from_iterator(['Hello world. Small CPU fixture!'] * 3, trainer)
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, bos_token='<s>',
            eos_token='</s>', unk_token='<unk>', pad_token='<pad>', model_max_length=1024)
        cfg['model'].update(vocab_size=len(tokenizer), n_layer=1, n_embd=16, n_head=2, n_positions=32)
        model = GPT2LMHeadModel(attention.configure_model_attention(models.model_config(cfg), 'sdpa')).eval()
        with tempfile.TemporaryDirectory() as tmp:
            final = Path(tmp) / 'final'
            final.mkdir()
            models.export_final(final, model, tokenizer)
            self.assertTrue((final / 'model.safetensors').is_file())
            loaded = GPT2LMHeadModel.from_pretrained(final, local_files_only=True, attn_implementation='sdpa').eval()
            loaded_tokenizer = AutoTokenizer.from_pretrained(final, local_files_only=True)
            self.assertEqual(loaded_tokenizer.get_vocab(), tokenizer.get_vocab())
            self.assertEqual((loaded.config.bos_token_id, loaded.config.eos_token_id), (1, 2))
            ids = tokenizer('Hello world', return_tensors='pt')['input_ids']
            with torch.no_grad():
                torch.testing.assert_close(loaded(ids).logits, model(ids).logits)
            with self.assertRaises(FileExistsError):
                models.export_final(final, model, tokenizer)

    def test_fresh_run_allows_scaffolding_but_never_training_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            results, logs = root / 'results', root / 'logs'
            require_fresh_run(results, logs)
            (results / 'checkpoints').mkdir(parents=True)
            (results / 'final').mkdir()
            (results / 'records').mkdir()
            logs.mkdir()
            for name in ('123.out', '123.err', '123_0.out', '123_0.err'):
                (logs / name).write_text('scheduler output')
            require_fresh_run(results, logs)
            for target in (logs / 'config.yaml', logs / 'training_metrics.jsonl',
                           logs / 'epoch_0_metrics.pth', logs / 'run_summary.json',
                           results / 'records/config.yaml', results / 'records/metrics.json',
                           results / 'records/run_summary.json',
                           results / 'final/model.safetensors', results / '123.out'):
                target.touch()
                with self.subTest(target=target), self.assertRaises(FileExistsError):
                    require_fresh_run(results, logs)
                target.unlink()


    def test_entrypoint_loads_config_and_dispatches(self):
        for language, sizes in [('English', ('10M', '100M')), ('German', ('10M', '108M'))]:
            for size in sizes:
                with self.subTest(language=language, size=size), patch.object(training, 'train') as run:
                    training.main(['--config', f'configs/{language}/{size}.yaml'])
                    run.assert_called_once()
                    cfg, root = run.call_args.args
                    self.assertEqual((cfg['dataset']['language'], cfg['dataset']['size']), (language, size))
                    self.assertEqual(root, PROJECT_ROOT)


    def test_training_records_accumulate_and_survive_interrupted_epoch(self):
        # Exercise the real training control flow with all GPU/model work mocked.
        for interrupted in (False, True):
            with self.subTest(interrupted=interrupted), tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
                root = Path(tmp)
                cfg, _ = utils.load_experiment('configs/English/10M.yaml')
                cfg['training']['epochs'] = 2
                tokenizer = MagicMock()
                tokenizer.__len__.return_value = cfg['model']['vocab_size']
                batch = [MagicMock() for _ in range(3)]
                batch[2].to.return_value.sum.return_value.item.return_value = 4
                loader = MagicMock()
                loader.__len__.return_value = 1
                loader.__iter__.side_effect = [iter([batch]), RuntimeError('interrupted epoch') if interrupted else iter([batch])]
                loss = MagicMock()
                loss.item.return_value = 2.5
                stack.enter_context(patch.dict(os.environ, WORLD_SIZE='1'))
                replacements = {
                    'PROJECT_ROOT': root,
                    'require_training_runtime': MagicMock(return_value={}),
                    'require_model_backend': MagicMock(return_value='sdpa'),
                    'AutoTokenizer.from_pretrained': MagicMock(return_value=tokenizer),
                    'BaselineDataset': MagicMock(), 'DataLoader': MagicMock(return_value=loader),
                    'GPT2LMHeadModel': MagicMock(), 'masked_loss': MagicMock(return_value=loss),
                    'torch.cuda.manual_seed_all': MagicMock(), 'torch.cuda.empty_cache': MagicMock(),
                    'torch.autocast': MagicMock(side_effect=lambda **kwargs: nullcontext()),
                    'torch.nn.utils.clip_grad_norm_': MagicMock(), 'torch.optim.AdamW': MagicMock(),
                    'get_cosine_schedule_with_warmup': MagicMock(),
                    'save_checkpoint': MagicMock(side_effect=lambda target, *args: target.mkdir(parents=True)),
                    'export_final': MagicMock(side_effect=lambda target, *args: target.mkdir(parents=True)),
                }
                for name, replacement in replacements.items():
                    stack.enter_context(patch('src.training.' + name, replacement))
                if interrupted:
                    with self.assertRaisesRegex(RuntimeError, 'interrupted epoch'):
                        training.train(cfg, root)
                else:
                    training.train(cfg, root)
                run = root / 'results/English/10M/seed_42'
                records = run / 'records'
                expected = [{'epoch': epoch, 'loss': 2.5} for epoch in range(1 if interrupted else 2)]
                self.assertEqual(json.loads((records / 'metrics.json').read_text()), expected)
                saved_cfg = yaml.safe_load((records / 'config.yaml').read_text())
                self.assertEqual(saved_cfg['training'], cfg['training'])
                self.assertEqual((records / 'run_summary.json').exists(), not interrupted)
                if not interrupted:
                    summary = json.loads((records / 'run_summary.json').read_text())
                    self.assertEqual(summary['epochs'], 2)
                    self.assertEqual(summary['final_model'], 'results/English/10M/seed_42/final')
                self.assertTrue(all(path.is_dir() for path in run.iterdir()))
                self.assertFalse((root / 'logs').exists())
                self.assertFalse(list(run.rglob('*.pth')))

    def test_failed_atomic_metrics_update_preserves_previous_epochs(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'records/metrics.json'
            previous = [{'epoch': 0, 'loss': 2.5}]
            utils.write_json_atomic(path, previous)
            with patch.object(Path, 'replace', side_effect=OSError('write interrupted')):
                with self.assertRaisesRegex(OSError, 'write interrupted'):
                    utils.write_json_atomic(path, previous + [{'epoch': 1, 'loss': 2.0}])
            self.assertEqual(json.loads(path.read_text()), previous)
