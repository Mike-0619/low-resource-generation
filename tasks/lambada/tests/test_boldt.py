"""Boldt cleanup, occurrence alignment and unchanged generation/scoring."""
import io
import unittest
from contextlib import redirect_stderr
from dataclasses import replace
from pathlib import Path
from unittest.mock import call, patch

from tasks.lambada.src import greedy as word, data as dataset, local, api as backend_api, scoring
from test_local import ByteTokenizer, FakeModel, local_runtime


class BoldtTests(unittest.TestCase):
    def test_cleanup_preserves_source_ids_and_matches_occurrences(self):
        english = ['English First', 'Repeated Name', 'Repeated Name',
                   'Missing Context', 'Other Target']
        rows = [
            {'text': 'Kontext Ziel \n', 'text_orig': english[4], 'text_cont': ' NEVER APPEND'},
            {'text': 'Kontext Name', 'text_orig': english[1], 'text_cont': ''},
            {'text': 'Selbst', 'text_orig': english[3], 'text_cont': ' not context'},
            {'text': 'Anderer Name', 'text_orig': english[1], 'text_cont': ''},
            {'text': 'Extra Name', 'text_orig': english[1], 'text_cont': ''},
            {'text': 'Letzter Anfang', 'text_orig': english[0], 'text_cont': ''},
        ]
        examples, metadata = dataset.prepare_boldt_examples(rows, english)
        self.assertEqual(examples, ['Kontext Ziel', 'Kontext Name', 'Anderer Name', 'Letzter Anfang'])
        self.assertEqual(metadata['sample_mapping'], [
            {'dataset_index': 0, 'english_dataset_index': 4},
            {'dataset_index': 1, 'english_dataset_index': 1},
            {'dataset_index': 3, 'english_dataset_index': 2},
            {'dataset_index': 5, 'english_dataset_index': 0},
        ])
        self.assertEqual(metadata['trimmed_examples'], [{'dataset_index': 0, 'reason': 'trailing_whitespace'}])
        self.assertEqual(metadata['skipped_examples'], [
            {'dataset_index': 2, 'english_dataset_index': 3, 'reason': 'missing_context_or_target'},
            {'dataset_index': 4, 'reason': 'excess_original_occurrence'},
        ])
        self.assertEqual(metadata['preparation_version'], 'boldt_minimal_cleanup_v1')
        self.assertEqual(rows[0]['text'], 'Kontext Ziel \n')

    def test_unknown_original_is_an_error_not_an_arbitrary_pair(self):
        with self.assertRaisesRegex(ValueError, 'Unknown English original'):
            dataset.prepare_boldt_examples([{'text': 'Kontext Ziel', 'text_orig': 'unknown'}], ['Known Source'])

    def test_loader_uses_pinned_sources_and_reports_raw_and_eligible_counts(self):
        rows = [{'text': 'Kontext Ziel ', 'text_orig': 'English Target', 'text_cont': ' ignore'},
                {'text': 'Extra Ziel', 'text_orig': 'English Target', 'text_cont': ''}]
        english = [{'text': 'English Target'}]
        with (patch.dict(dataset.DATASET_SPECS, {'de': replace(dataset.DATASET_SPECS['de'], expected_examples=2)}),
              patch.dict(dataset.DATASET_SPECS, {'en': replace(dataset.DATASET_SPECS['en'], expected_examples=1)}),
              patch.object(dataset, 'BOLDT_ELIGIBLE_EXAMPLES', 1),
              patch.object(dataset, '_load_lambada_dataset', side_effect=[rows, english]) as load):
            examples, metadata = dataset.load_lambada_examples('de')
        self.assertEqual(load.call_args_list, [call('de'), call('en')])
        self.assertEqual(examples, ['Kontext Ziel'])
        self.assertEqual((metadata['dataset_examples'], metadata['eligible_examples']), (2, 1))
        self.assertEqual(metadata['dataset_revision'], '74d444e7ba3e598d4ae48455f8a4937c5295cce4')
        self.assertEqual(metadata['reference_dataset']['revision'], '900124bf3b8235c6daf21033af9948b3f07346c4')

    def test_german_defaults_to_boldt_and_dataset_switch_is_removed(self):
        for backend in ('local', 'api'):
            args = ['--backend', backend, '--language', 'de', '--model', 'any/model', '--output_dir', '/unused']
            self.assertEqual(word.parse_args(args).language, 'de')
            self.assertEqual(dataset.DATASET_SPECS['de'].name, 'Boldt/lambada_openai_de')
            for value in ('default', 'boldt'):
                with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    word.parse_args(args + ['--dataset', value])


    def test_evaluation_keeps_boldt_ids_and_uses_selected_count(self):
        tok = ByteTokenizer()
        examples = ['first target', 'second target']
        model = FakeModel([tok.encode('target '), tok.encode('target ')])
        metadata = {'dataset': 'Boldt/lambada_openai_de (test)', 'dataset_variant': 'boldt',
                    'dataset_examples': 5, 'eligible_examples': 2,
                    'skipped_examples': [{'dataset_index': 1, 'reason': 'fixture'}],
                    'sample_mapping': [{'dataset_index': 2, 'english_dataset_index': 9},
                                       {'dataset_index': 4, 'english_dataset_index': 3}],
                    'preparation_version': 'boldt_minimal_cleanup_v1'}
        with (patch.object(local.LocalModel, 'load', return_value=local_runtime(tok, model)),
              patch.object(dataset, 'load_lambada_examples', return_value=(examples, metadata)) as load):
            metrics, rows = word.evaluate('/model', 'de', device='cpu', limit=20)
        load.assert_called_once_with('de')
        self.assertEqual([r['sample_id'] for r in rows], ['de:boldt:2', 'de:boldt:4'])
        self.assertEqual([r['english_dataset_index'] for r in rows], [9, 3])
        self.assertEqual((metrics['correct'], metrics['total'], metrics['accuracy']), (2, 2, 1.0))
        self.assertEqual(metrics['dataset_examples'], 5)
        self.assertNotIn('sample_mapping', metrics)
        self.assertIn('sample_mapping', metadata)
        with (patch.object(local.LocalModel, 'load', return_value=local_runtime(tok, FakeModel([tok.encode('target ')]))),
              patch.object(dataset, 'load_lambada_examples', return_value=(examples, metadata))):
            limited, rows = word.evaluate('/model', 'de', device='cpu', limit=1)
        self.assertEqual(limited['total'], 1)
        self.assertEqual(rows[0]['dataset_index'], 2)


if __name__ == '__main__':
    unittest.main()
