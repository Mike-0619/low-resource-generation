import unittest
import tempfile

from pathlib import Path

from src.training.utils import CONFIG_PATHS, SEEDS, VOCAB_SIZES, condition_paths, configs_for_condition, run_paths, require_fresh_run


class ConfigPathTest(unittest.TestCase):
    def test_configs_paths_and_shared_artifacts(self) -> None:
        configs = sum(
            (configs_for_condition(group) for group in (
                "shared_spans", "balanced_batch"
            )),
            (),
        )
        self.assertEqual(set(CONFIG_PATHS), {config.condition.name for config in configs})
        paths = [
            run_paths(config.condition, vocab, seed)
            for config in configs
            for seed in SEEDS
            for vocab in VOCAB_SIZES
        ]
        self.assertEqual(
            tuple(len({getattr(path, field) for path in paths})
                  for field in ("run", "records", "initialization")),
            (108, 108, 18),
        )

        by_family = {}
        for config in configs:
            key = (config.condition.size, config.condition.language)
            path = run_paths(config.condition, 512, 42)
            suffix = Path(config.condition.variant) / 'vocab_512/seed_42'
            family = Path(config.condition.language) / config.condition.size
            from src.training.utils import PROJECT_ROOT
            self.assertEqual(path.run, PROJECT_ROOT / 'results' / family / suffix)
            self.assertEqual(path.logs, PROJECT_ROOT / 'logs' / family / config.condition.variant / 'train')
            self.assertEqual(path.relative_paths()['logs'], str(path.logs.relative_to(PROJECT_ROOT)))
            self.assertEqual(path.records, path.run / 'records')
            self.assertEqual(path.last, path.run / 'checkpoints/last')
            self.assertEqual(path.final, path.run / 'final')
            if key in by_family:
                self.assertEqual(path.tokenizer, by_family[key].tokenizer)
                self.assertEqual(path.initialization, by_family[key].initialization)
            else:
                by_family[key] = path

        spans = condition_paths(configs_for_condition("10m_en_spans")[0].condition)
        balanced = condition_paths(configs_for_condition("10m_en_balanced")[0].condition)
        self.assertNotEqual(balanced.token_cache(512), spans.token_cache(512))
        self.assertNotEqual(balanced.cleaned, spans.cleaned)
        self.assertEqual(balanced.cleaned.parent.name, "balanced_batch")
        self.assertEqual(spans.cleaned.parent.name, "shared_spans")
        self.assertEqual(balanced.shared_cache(512), spans.token_cache(512))

    def test_expected_raw_corpora_are_shared_between_protocols(self):
        expected = {
            ("English", "10M"): (
                "babylm-2026/English/10M", "BabyLM-community/BabyLM-2026-Strict-Small",
                "c92ab16b4f08858304b0815706065b3354d8fc0a",
            ),
            ("English", "100M"): (
                "babylm-2026/English/100M", "BabyLM-community/BabyLM-2026-Strict",
                "9e57baaaa91ac3c638746be14d1d5fa6c789f4cf",
            ),
            ("German", "100M"): (
                "babylm-2026/German/108M", "BabyLM-community/babylm-deu",
                "69ff474c64a7e0df3a473b1b5d968a0fc18478ac",
            ),
        }
        for protocol in ("shared_spans", "balanced_batch"):
            for config in configs_for_condition(protocol):
                condition = config.condition
                with self.subTest(condition=condition.name):
                    self.assertEqual(
                        (condition.corpus_path, condition.dataset_repo_id,
                         condition.dataset_revision),
                        expected[condition.language, condition.size],
                    )

    def test_fresh_run_allows_scaffolding_but_never_training_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            results, logs = root / 'results', root / 'logs/train'
            legacy_logs = logs.with_name('seed_42')
            require_fresh_run(results, logs, legacy_logs)
            (results / 'checkpoints').mkdir(parents=True)
            (results / 'final').mkdir()
            for path in (logs, legacy_logs):
                path.mkdir(parents=True)
                for name in ('123.out', '123.err', '123_0.out', '123_0.err'):
                    (path / name).write_text('scheduler output')
            require_fresh_run(results, logs, legacy_logs)
            for target in (logs / 'config.yaml', logs / 'training_metrics.jsonl',
                           logs / 'epoch_0_metrics.pth', legacy_logs / 'run_summary.json',
                           results / 'final/model.safetensors', results / '123.out'):
                target.touch()
                with self.subTest(target=target), self.assertRaises(FileExistsError):
                    require_fresh_run(results, logs, legacy_logs)
                target.unlink()

    def test_removed_route_rejected(self):
        from dataclasses import replace
        from src.training.utils import load_config
        with self.assertRaises(ValueError):
            configs_for_condition("invalid_route")
        with self.assertRaises(ValueError):
            replace(load_config().condition, variant="invalid_route")
