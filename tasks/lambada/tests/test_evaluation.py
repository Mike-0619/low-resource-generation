"""CLI, fixed datasets, evaluation preflight and protected result exports."""
import io
import json
import os
import subprocess
import sys
import textwrap
import tempfile
import unittest
from contextlib import contextmanager, redirect_stdout, redirect_stderr
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import call, patch

import torch
from tasks.lambada import src as lambada
from tasks.lambada.src import greedy as word, data as dataset, local, api as backend_api, scoring
from tasks.lambada.src.data import DATASET_SPECS
from tasks.lambada.src.data import _load_lambada_dataset
from tasks.lambada.src.data import load_lambada_examples
from tasks.lambada.src.greedy import evaluate
from tasks.lambada.src.scoring import summarize_greedy
from tasks.lambada.src.greedy import write_greedy_results
from test_local import ByteTokenizer, FakeModel, local_runtime
from test_api import api, response, ResponseStream


def data(count=2):
    return ([f"Context{i} Straße" for i in range(count)], {
        "dataset": "fixture", "dataset_revision": "fixed", "dataset_examples": count,
        "eligible_examples": count, "skipped_examples": [],
    })


@contextmanager
def local_evaluation(tokenizer, model, examples):
    metadata = {'dataset_examples': len(examples), 'skipped_examples': []}
    with (
        patch.object(local.LocalModel, 'load', return_value=local_runtime(tokenizer, model)),
        patch.object(dataset, 'load_lambada_examples', return_value=(examples, metadata)),
    ):
        yield


class EvaluationTests(unittest.TestCase):
    def test_invalid_programmatic_options_fail_before_model_data_or_api(self):
        local_options = {'model': '/model', 'language': 'en', 'device': 'cpu'}
        api_options = {'model': 'any/model', 'language': 'en', 'backend': 'api'}
        invalid = [
            {**local_options, 'backend': 'unknown'}, {**local_options, 'language': 'fr'},
            {**local_options, 'limit': 0}, {**local_options, 'model': ''},
            {**local_options, 'batch_size': 0}, {**local_options, 'device': 'invalid'},
            {**local_options, 'limit': None}, {**local_options, 'reasoning_effort': 'none'},
            {**local_options, 'concurrency': 4}, {**local_options, 'context_limit': 0},
            {**api_options, 'batch_size': 8}, {**api_options, 'device': 'cpu'},
            {**api_options, 'bos': 'auto'}, {**api_options, 'dtype': 'auto'},
            {**api_options, 'context_limit': 512}, {**api_options, 'concurrency': 0},
        ]
        for options in invalid:
            with (self.subTest(options=options),
                  patch.object(local.LocalModel, 'load') as model,
                  patch.object(dataset, 'load_lambada_examples') as data_load,
                  patch.object(backend_api, 'generate') as api_generate,
                  self.assertRaises(ValueError)):
                word.evaluate(**options)
            model.assert_not_called()
            data_load.assert_not_called()
            api_generate.assert_not_called()


    def test_turkish_placeholder_fails_before_evaluation(self):
        message = 'Turkish LAMBADA data is pending'
        for backend in ('local', 'api'):
            stderr = io.StringIO()
            with (self.subTest(backend=backend), TemporaryDirectory() as directory,
                  redirect_stderr(stderr), patch.object(local, 'resolve_device') as resolve_device,
                  patch.object(word, '_evaluate') as run):
                output = Path(directory) / 'results'
                with self.assertRaises(SystemExit) as error:
                    word.main(['--backend', backend, '--model', 'fixture/model', '--language',
                               'tr', '--output_dir', str(output)])
                self.assertEqual(error.exception.code, 2)
                self.assertIn(message, stderr.getvalue())
                with self.assertRaisesRegex(ValueError, message):
                    word.evaluate('fixture/model', 'tr', backend=backend)
                resolve_device.assert_not_called()
                run.assert_not_called()
                self.assertFalse(output.exists())

    def test_turkish_placeholder_fails_before_loading_dataset(self):
        with (patch.object(dataset, '_load_lambada_dataset') as load,
              self.assertRaisesRegex(ValueError, 'Turkish LAMBADA data is pending')):
            dataset.load_lambada_examples('tr')
        load.assert_not_called()

    def test_invalid_cli_options_preserve_existing_outputs(self):
        cases = [
            ['--backend', 'local', '--model', '/model', '--batch_size', '0'],
            ['--backend', 'local', '--model', '/model', '--device', 'cpu'],
            ['--backend', 'local', '--model', '/model', '--concurrency', '4'],
            ['--backend', 'api', '--model', 'fixture/model', '--dtype', 'auto'],
            ['--backend', 'api','--model','fixture/model', '--concurrency', '0'],
        ]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            previous = {'metrics.json': 'previous metrics', 'samples.json': 'previous samples'}
            for name, text in previous.items():
                (output / name).write_text(text)
            for options in cases:
                with (self.subTest(options=options), redirect_stderr(io.StringIO()),
                      patch.object(word, '_evaluate') as run,
                      self.assertRaises(SystemExit)):
                    word.main(['--language', 'en', '--output_dir', directory, *options])
                run.assert_not_called()
                self.assertEqual({p.name: p.read_text() for p in output.iterdir()}, previous)

    def test_cli_defaults_selection_and_invalid_combinations(self):
        base = ['--model', '/model', '--language', 'en', '--output_dir', '/unused']
        command = [*base, '--backend', 'local']
        args = word.parse_args(command)
        self.assertEqual((args.backend, args.limit, args.device, args.batch_size, args.bos, args.dtype),
                         ('local', None, 'cuda', 8, 'auto', 'auto'))
        args = word.parse_args([*command, '--device', 'cpu', '--batch_size', '1', '--limit', '3'])
        self.assertEqual((args.limit, args.batch_size, args.device), (3, 1, 'cpu'))
        args = word.parse_args([*base, '--backend', 'api'])
        self.assertEqual((args.model, args.concurrency, args.reasoning_effort), ('/model', 4, None))
        invalid = [base, [*command, '--all'], [*command, '--overwrite'], [*command, '--device', 'cpu'],
                   [*command, '--batch_size', '0'], [*command, '--concurrency', '4'],
                   [*command, '--reasoning_effort', 'none'], [*command, '--context_limit', '0'],
                   [*command, '--model_path', '/legacy'], [*command, '--gpt_model', 'legacy'],
                   [*command, '--backend', 'gpt2']]
        for option, value in [('--device', 'cpu'), ('--batch_size', '8'), ('--dtype', 'auto'),
                              ('--bos', 'auto'), ('--context_limit', '512'), ('--concurrency', '0'), ('--limit', '0')]:
            invalid.append([*base, '--backend', 'api', option, value])
        for args in invalid:
            with self.subTest(args=args), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                word.parse_args(args)


    def test_cli_and_programmatic_paths_share_defaults_and_validate_once(self):
        for backend in ('local', 'api'):
            outputs = []
            for cli in (False, True):
                tok = ByteTokenizer()
                model = FakeModel([tok.encode('Straße!')])
                args = ['--backend', backend, '--model', '/model', '--language', 'en']
                kwargs = {'backend': backend}
                if backend == 'local':
                    args += ['--device', 'cpu', '--limit', '1']
                    kwargs.update(device='cpu', limit=1)
                with (self.subTest(backend=backend, cli=cli),
                      tempfile.TemporaryDirectory() as directory,
                      patch.object(dataset, 'load_lambada_examples', return_value=data(1)),
                      patch.object(local.LocalModel, 'load', return_value=local_runtime(tok, model)),
                      patch.object(word, 'evaluation_options', wraps=word.evaluation_options) as validate,
                      api([response('Straße!')]), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO())):
                    if cli:
                        word.main([*args, '--output_dir', directory])
                        outputs.append((json.loads((Path(directory)/'metrics.json').read_text()),
                                        json.loads((Path(directory)/'samples.json').read_text())))
                    else:
                        outputs.append(word.evaluate('/model', 'en', **kwargs))
                    validate.assert_called_once()
            self.assertEqual(outputs[0], outputs[1])

    def test_model_preflight_precedes_dataset_loading(self):
        with (patch.object(local.LocalModel, 'load', side_effect=ValueError('incompatible model')),
              patch.object(dataset, 'load_lambada_examples') as load,
              self.assertRaisesRegex(ValueError, 'incompatible model')):
            word.evaluate('/model', 'en', device='cpu', limit=20)
        load.assert_not_called()

    def test_no_cuda_fails_before_loading_or_output_changes(self):
        with tempfile.TemporaryDirectory() as directory, patch('torch.cuda.is_available', return_value=False), \
                patch.object(local.LocalModel, 'load') as load:
            output = Path(directory)/'new'
            with self.assertRaisesRegex(RuntimeError, 'CUDA is required'):
                word.main(['--backend','local','--model','/model','--language','en','--output_dir',str(output)])
            self.assertFalse(output.exists())
            load.assert_not_called()

    def test_preflight_rejects_entire_invalid_local_selection(self):
        tok = ByteTokenizer()
        model = FakeModel([])
        for invalid in ('x'*511 + ' target', ' target'):
            with local_evaluation(tok, model, ['valid target', invalid]):
                with self.assertRaisesRegex(ValueError, "dataset index 1"):
                    word.evaluate('/model','en',device='cpu', limit=20)
                self.assertEqual(model.calls, [])

    def test_programmatic_defaults_metadata_and_cpu_full_rejection(self):
        tok = ByteTokenizer()
        examples = ['context target'] * 25
        model = FakeModel([tok.encode('target ')] * 25)
        with local_evaluation(tok, model, examples):
            metrics, rows = word.evaluate('/model', 'en')
        self.assertEqual(len(rows), 25)
        self.assertEqual(metrics['sample_selection'], {'mode': 'all', 'limit': None})
        with self.assertRaisesRegex(ValueError, 'full evaluation requires CUDA'):
            word.evaluate('/model', 'en', device='cpu')
        with local_evaluation(tok, FakeModel([tok.encode('target ')] * 25), examples):
            limited, rows = word.evaluate('/model', 'en', limit=3, device='cpu')
        self.assertEqual(len(rows), 3)
        self.assertEqual(limited['sample_selection'], {'mode': 'limit', 'limit': 3})

    def test_contexts_are_encoded_once_and_prepared_inputs_stay_private(self):
        tok = ByteTokenizer()
        contexts = ['first', 'a longer context', 'last']
        examples = [context + ' target' for context in contexts]
        model = FakeModel([tok.encode('target ')] * len(examples))
        with (
            local_evaluation(tok, model, examples),
            patch.object(tok, 'encode', wraps=tok.encode) as encode,
        ):
            metrics, samples = word.evaluate('/model', 'en', device='cpu', limit=20)
        self.assertEqual(encode.call_args_list, [call(context, add_special_tokens=True) for context in contexts])
        self.assertEqual(metrics['correct'], len(examples))
        expected_fields = {
            'sample_id', 'dataset_index', 'context', 'target_word',
            'generated_ids', 'generated_text', 'generation_budget',
            'prediction', 'completed', 'stop_reason',
            'normalized_prediction', 'normalized_target_word', 'exact_match',
        }
        for index, sample in enumerate(samples):
            self.assertEqual(set(sample), expected_fields)
            self.assertEqual(sample['dataset_index'], index)

    def test_selected_english_samples_are_checked_before_generation(self):
        tok = ByteTokenizer()
        rows = [{'text': 'context target'}, {'text': 'invalid'}] + [{'text': 'context target'}] * 5151
        model = FakeModel([tok.encode('target ')])
        with patch.object(local, 'FirstWord', wraps=local.FirstWord) as create:
            runner = local_runtime(tok, model)
            with (patch.object(dataset, '_load_lambada_dataset', return_value=rows),
                  patch.object(local.LocalModel, 'load', return_value=runner)):
                metrics, samples = word.evaluate('/model', 'en', limit=1, device='cpu')
        self.assertEqual(metrics['correct'], 1)
        self.assertEqual(samples[0]['dataset_index'], 0)
        create.assert_called_once()
        model = FakeModel([])
        with (patch.object(dataset, '_load_lambada_dataset', return_value=rows),
              patch.object(local.LocalModel, 'load', return_value=local_runtime(tok, model)),
              self.assertRaisesRegex(ValueError, 'dataset index 1')):
            word.evaluate('/model', 'en', limit=2, device='cpu')
        self.assertEqual(model.calls, [])

    def test_run_outputs_and_partial_failure(self):
        for failure in (False, True):
            with (self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory,
                  patch.object(dataset, 'load_lambada_examples', return_value=data()),
                  patch.object(local.LocalModel, 'load') as local_model,
                  redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO())):
                output = Path(directory) / 'result'
                output.mkdir()
                previous = {'metrics.json': '{"status":"complete"}',
                            'samples.json': 'previous samples'}
                for name, contents in previous.items():
                    (output / name).write_text(contents)
                args = ['--backend', 'api', '--model', 'fixture/model', '--language', 'en',
                        '--concurrency', '1', '--output_dir', str(output)]
                ending = ResponseStream([OSError('broken')]) if failure else response('Strasse')
                with api([response('Straße'), ending]) as client:
                    if failure:
                        with self.assertRaises(OSError):
                            word.main(args)
                    else:
                        word.main(args)
                    self.assertTrue(client.closed)
                local_model.assert_not_called()
                if failure:
                    self.assertEqual({p.name: p.read_text() for p in output.iterdir()}, previous)
                else:
                    self.assertEqual({p.name for p in output.iterdir()},
                                     {'metrics.json', 'samples.json'})
                    rows = json.loads((output / 'samples.json').read_text())
                    self.assertEqual(len(rows), 2)
                    self.assertTrue(all(row['exact_match'] for row in rows))
                    self.assertEqual(json.loads((output / 'metrics.json').read_text())['accuracy'], 1)


    def test_export_errors_close_api_and_preserve_previous_results(self):
        for filename in ('samples.json', 'metrics.json'):
            with (self.subTest(filename=filename),
                  tempfile.TemporaryDirectory() as directory,
                  patch.object(dataset, 'load_lambada_examples', return_value=data()),
                  api([response('Straße'), response('Strasse')]) as client,
                  redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO())):
                output = Path(directory) / 'result'
                output.mkdir()
                previous = {'metrics.json': '{"status":"complete"}',
                            'samples.json': 'previous samples'}
                for name, contents in previous.items():
                    (output / name).write_text(contents)
                original_write = Path.write_text

                def write(path, *args, **kwargs):
                    if path.name == filename and path.parent.name == '.result.staging':
                        original_write(path, 'partial')
                        raise OSError('disk full')
                    return original_write(path, *args, **kwargs)

                with patch.object(Path, 'write_text', write), self.assertRaises(OSError):
                    word.main(['--backend', 'api', '--model', 'fixture/model',
                               '--language', 'en', '--concurrency', '1',
                               '--output_dir', str(output)])
                self.assertTrue(client.closed)
                self.assertEqual({p.name: p.read_text() for p in output.iterdir()}, previous)
                self.assertEqual(client.responses.create.call_count, 2)
                self.assertFalse(output.with_name('.result.staging').exists())

    def test_reference_independence_and_all_record_exports(self):
        tokenizer = ByteTokenizer()
        context = 'Context, with "quotes"\nand a newline'
        targets = ["Shane", "LongerReference", "Shane", "Shane", "Shane", "Shane", "é"]
        outputs = ["Shane ", "Shane ", "Shane, ", "shane ", "and it ", "a" * 128, "e\u0301 "]
        model = FakeModel([tokenizer.encode(text) for text in outputs])
        examples = [context + ' ' + target for target in targets]
        with local_evaluation(tokenizer, model, examples):
            metrics, samples = evaluate('/model', 'en', batch_size=1, device='cpu', limit=20)
        self.assertEqual((metrics['total'], metrics['correct'], metrics['completed'], metrics['failed']), (7, 2, 6, 1))
        self.assertEqual(metrics['accuracy'], 2 / 7)
        self.assertEqual(metrics['protocol'], 'case_sensitive_punctuation_boundary_sharp_s_v2')
        self.assertEqual(samples[2]['prediction'], 'Shane,')
        self.assertEqual(samples[2]['normalized_prediction'], 'Shane')
        self.assertEqual(summarize_greedy(samples)['correct'], metrics['correct'])
        self.assertEqual([item['dataset_index'] for item in samples], list(range(7)))
        self.assertEqual([item['sample_id'] for item in samples], [f'en:{i}' for i in range(7)])
        self.assertEqual(samples[0]['generated_ids'], samples[1]['generated_ids'])
        self.assertTrue(torch.equal(model.calls[0]['input_ids'], model.calls[1]['input_ids']))
        for call in model.calls:
            config = call['generation_config']
            self.assertFalse(config.do_sample)
            self.assertEqual((config.num_beams, config.max_new_tokens), (1, 128))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            write_greedy_results(output, metrics, samples)
            stored = json.loads((output / 'samples.json').read_text())
            self.assertEqual(stored, samples)
            self.assertEqual(json.loads((output / 'metrics.json').read_text()), metrics)
            self.assertEqual({path.name for path in output.iterdir()},
                             {'metrics.json', 'samples.json'})

    def test_online_load_uses_language_dataset_spec(self):
        expected_specs = {
            "en": (
                "EleutherAI/lambada_openai",
                "default",
                "900124bf3b8235c6daf21033af9948b3f07346c4",
            ),
            "de": (
                "Boldt/lambada_openai_de",
                "default",
                "74d444e7ba3e598d4ae48455f8a4937c5295cce4",
            ),
        }
        for language, expected_spec in expected_specs.items():
            expected = object()
            with patch("datasets.load_dataset", return_value=expected) as load:
                actual = _load_lambada_dataset(language)
            self.assertIs(actual, expected)
            name, config, revision = expected_spec
            from datasets import config as datasets_config
            cache = Path(datasets_config.HF_DATASETS_CACHE)
            load.assert_called_once_with(name, config, split="test", revision=revision, cache_dir=str(cache))

    def test_online_and_offline_cache_resolution_without_shell_setup(self):
        probe = textwrap.dedent("""
            import sys
            from pathlib import Path
            from unittest.mock import patch
            from datasets import config
            from tasks.lambada.src.data import DATASET_SPECS, _load_lambada_dataset

            cache = Path(sys.argv[1])
            assert Path(config.HF_DATASETS_CACHE) == cache
            for language, spec in DATASET_SPECS.items():
                arrow = (cache / spec.name.replace("/", "___") / spec.config / "0.0.0"
                         / spec.revision / f"{spec.name.rsplit('/', 1)[-1]}-test.arrow")
                arrow.parent.mkdir(parents=True)
                arrow.touch()
                expected = object()
                with patch("datasets.load_dataset", return_value=expected) as load:
                    assert _load_lambada_dataset(language) is expected
                    assert load.call_args.kwargs["cache_dir"] == str(cache)
                with (patch("datasets.load_dataset", side_effect=OSError("offline")) as load,
                      patch("datasets.Dataset.from_file", return_value=expected) as from_file):
                    assert _load_lambada_dataset(language) is expected
                    assert load.call_args.kwargs["cache_dir"] == str(cache)
                    from_file.assert_called_once_with(str(arrow))
        """)
        for mode in ("default", "xdg", "hf_home", "explicit"):
            with self.subTest(mode=mode), TemporaryDirectory(dir=os.environ.get("XDG_CACHE_HOME")) as directory:
                root = Path(directory)
                env = {key: value for key, value in os.environ.items()
                       if key not in {"HF_HOME", "HF_DATASETS_CACHE", "XDG_CACHE_HOME",
                                      "WORKSPACE_ROOT", "PYTHONPATH"}}
                env["HOME"] = str(root)
                cache = root / ".cache/huggingface/datasets"
                if mode == "xdg":
                    env["XDG_CACHE_HOME"] = str(root / "xdg")
                    cache = root / "xdg/huggingface/datasets"
                elif mode in {"hf_home", "explicit"}:
                    env["HF_HOME"] = str(root / "hf")
                    cache = root / "hf/datasets"
                    if mode == "explicit":
                        env["HF_DATASETS_CACHE"] = str(root / "explicit")
                        cache = root / "explicit"
                result = subprocess.run(
                    [sys.executable, "-B", "-c", probe, str(cache)],
                    cwd=lambada.PROJECT_ROOT, env=env, capture_output=True, text=True, timeout=90,
                )
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_offline_load_uses_language_specific_cache(self):
        for language in ("en", "de"):
            spec = DATASET_SPECS[language]
            with TemporaryDirectory(dir=os.environ.get("XDG_CACHE_HOME")) as temporary_directory:
                cache_namespace = spec.name.replace("/", "___")
                cache_filename = (
                    f"{spec.name.rsplit('/', 1)[-1]}-test.arrow"
                )
                arrow = (
                    Path(temporary_directory)
                    / cache_namespace
                    / spec.config
                    / "0.0.0"
                    / spec.revision
                    / cache_filename
                )
                arrow.parent.mkdir(parents=True)
                arrow.touch()
                expected = object()
                with (
                    patch("datasets.config.HF_DATASETS_CACHE", Path(temporary_directory)),
                    patch("datasets.load_dataset", side_effect=OSError("offline")),
                    patch("datasets.Dataset.from_file", return_value=expected) as from_file,
                ):
                    actual = _load_lambada_dataset(language)

            self.assertIs(actual, expected)
            from_file.assert_called_once_with(str(arrow))

    def test_dataset_size_mismatch_is_rejected(self):
        with patch(
            "tasks.lambada.src.data._load_lambada_dataset",
            return_value=[{"text": "Kontext Ziel"}],
        ), self.assertRaisesRegex(ValueError, "has 1 examples; expected 5150"):
            load_lambada_examples("de")

    def test_english_metadata_and_text_are_unchanged(self):
        dataset = [{"text": "hello world"}] * 5153
        with patch(
            "tasks.lambada.src.data._load_lambada_dataset",
            return_value=dataset,
        ):
            examples, metadata = load_lambada_examples("en")

        self.assertEqual(examples, ["hello world"] * 5153)
        self.assertEqual(
            metadata,
            {
                "dataset": "EleutherAI/lambada_openai (test)",
                "dataset_revision": (
                    "900124bf3b8235c6daf21033af9948b3f07346c4"
                ),
                "dataset_examples": 5153,
                "eligible_examples": 5153,
                "skipped_examples": [],
            },
        )

    def test_successful_export_replaces_previous_results(self):
        with TemporaryDirectory(dir=os.environ.get("XDG_CACHE_HOME")) as directory:
            output = Path(directory) / "evaluation"
            write_greedy_results(output, {"metric": "old"}, [{"prediction": "old"}])
            write_greedy_results(output, {"metric": "new"}, [{"prediction": "new"}])
            self.assertEqual(json.loads((output / "metrics.json").read_text()), {"metric": "new"})
            self.assertEqual(json.loads((output / "samples.json").read_text()), [{"prediction": "new"}])
            self.assertEqual({path.name for path in output.iterdir()}, {'metrics.json', 'samples.json'})


if __name__ == '__main__':
    unittest.main()
