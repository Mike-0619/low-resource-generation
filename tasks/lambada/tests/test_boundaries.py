"""Shared punctuation boundaries across byte tokens, batches and API chunks."""
import unittest
from pathlib import Path
from unittest.mock import patch

from tasks.lambada.src import greedy as word, data as dataset, local, api as backend_api, scoring
MODELS = ("openai/gpt-5.6-luna", "openai/gpt-5.6-sol")
from test_api import api, event, ResponseStream
from test_local import ByteTokenizer, FakeModel, generate_word, local_runtime


class BoundaryTests(unittest.TestCase):
    def test_punctuation_boundaries_and_leading_quotes(self):
        for text in ('Wolf“.“…', '„Wolf!“', ' \t« „  Wolf»later', 'Wolf!later',
                     'Wolf?later', 'Wolf;later', 'Wolf…later', 'Wolf...later'):
            with self.subTest(text=text):
                self.assertEqual(scoring.inspect_text(text),
                                 {'prediction': 'Wolf', 'completed': True, 'stop_reason': 'punctuation'})
        for quote in scoring.LEADING_PUNCTUATION:
            with self.subTest(quote=quote):
                self.assertEqual(scoring.inspect_text(quote + ' Wolf' + quote)['prediction'], 'Wolf')

    def test_internal_punctuation_is_preserved_until_a_real_boundary(self):
        for text in ('3.14', '1,000', '12:30', 'U.S.', "don't", "dogs'", 'mother-in-law',
                     'word.', 'word..', "'Wolf'", '.5'):
            with self.subTest(text=text):
                self.assertIsNone(scoring.inspect_text(text))
                self.assertEqual(scoring.inspect_text(text + ' ')['prediction'], text)
                self.assertEqual(scoring.inspect_text(text, final=True)['prediction'], text)
        self.assertEqual(scoring.normalize_word('U.S.'), 'U.S')
        self.assertFalse(scoring.score_word('Wolf', 'Wolf!Suffix', True)['exact_match'])

    def test_pure_punctuation_fails_without_seeking_a_later_word(self):
        for text, reason in (
            ('!Wolf', 'empty_punctuation'), ('...Wolf', 'empty_punctuation'),
            (',;Wolf', 'empty_punctuation'), ('--- Wolf', 'empty_whitespace'),
            ("'' Wolf", 'empty_whitespace'), ('($) Wolf', 'empty_whitespace'),
            ('。。 Wolf', 'empty_whitespace'), ('„ “ ', 'empty_eos'),
            ('...', 'empty_punctuation'), ('', 'empty_eos'),
        ):
            with self.subTest(text=text):
                self.assertEqual(scoring.inspect_text(text, final=True),
                                 {'prediction': '', 'completed': False, 'stop_reason': reason})

    def test_utf8_and_special_tokens_before_and_after_boundaries(self):
        tok = ByteTokenizer()
        inspect = local.FirstWord(tok).inspect
        self.assertIsNone(inspect([value + 5 for value in b'k\xc3']))
        self.assertEqual(inspect(tok.encode('kühl!'))['prediction'], 'kühl')
        for prefix in (b'Wolf\xc3', b'Wolf\xff', b'\xffWolf'):
            for suffix in (b'!', b' ', b'...'):
                with self.subTest(prefix=prefix, suffix=suffix):
                    result = inspect([value + 5 for value in prefix + suffix])
                    self.assertEqual(result['stop_reason'], 'invalid_utf8')
        for boundary, reason in (('!', 'punctuation'), ('“', 'punctuation'), ('...', 'punctuation'), (' ', 'whitespace')):
            with self.subTest(boundary=boundary):
                ids = tok.encode('Wolf' + boundary) + [0xff + 5, 4]
                self.assertEqual(inspect(ids), {'prediction': 'Wolf', 'completed': True, 'stop_reason': reason})
        self.assertEqual(inspect(tok.encode('Wolf') + [4])['stop_reason'], 'special_token')
        self.assertEqual(inspect(tok.encode('Wolf') + [2])['stop_reason'], 'eos')
        self.assertEqual(inspect([0xc3 + 5, 2])['stop_reason'], 'invalid_utf8')

    def test_token_chunking_and_boundary_with_garbage_in_one_token(self):
        for text in ('„Wolf!“', 'kühl…later', 'Wolf...later', '« Straße »later'):
            raw = text.encode('utf-8')
            expected = scoring.inspect_text(text)
            for width in (1, 2, len(raw)):
                tok = ByteTokenizer()
                ids = []
                for at in range(0, len(raw), width):
                    piece = ''.join(tok.pieces[value + 5] for value in raw[at:at + width])
                    # Real tokenizers assign one ID per piece, even for repeated chunks.
                    token_id = next((token for token, value in tok.pieces.items() if value == piece), None)
                    if token_id is None:
                        token_id = max(tok.pieces) + 1
                        tok.pieces[token_id] = piece
                    ids.append(token_id)
                model = FakeModel([ids])
                model.config.vocab_size = len(tok)
                result = generate_word('context', tok, model, 'cpu')
                with self.subTest(text=text, width=width):
                    self.assertEqual({key: result[key] for key in expected}, expected)
        tok = ByteTokenizer()
        tok.pieces[301] = ''.join(tok.pieces[value + 5] for value in b'Wolf!\xff')
        self.assertEqual(local.FirstWord(tok).inspect([301, 4])['prediction'], 'Wolf')

    def test_ellipsis_waits_for_third_period(self):
        tok = ByteTokenizer()
        inspect = local.FirstWord(tok).inspect
        for text in ('Wolf.', 'Wolf..'):
            self.assertIsNone(inspect(tok.encode(text)))
        self.assertEqual(inspect(tok.encode('Wolf...'))['prediction'], 'Wolf')

    def test_api_splits_use_the_same_rule_and_close_at_first_boundary(self):
        for text in ('„Wolf!“', 'kühl…later', 'Wolf...later', '« Straße »later', '!Wolf',
                     'Wolf!\udcff', 'Wo\udcfflf!', '--- Wolf'):
            expected = scoring.inspect_text(text, final=True, terminal_reason='completed')
            for width in (1, 2, 100):
                stream = ResponseStream([
                    *[event('response.output_text.delta', delta=text[at:at + width])
                      for at in range(0, len(text), width)], event('response.completed'),
                ])
                with self.subTest(text=repr(text), width=width), api([stream]) as client:
                    actual = backend_api.request(client, {'context': 'context'}, MODELS[0])
                    self.assertEqual({key: actual[key] for key in expected}, expected)
                    self.assertTrue(stream.closed)
                    self.assertLess(stream.read_count, len(stream.events))

    def test_mixed_batches_equal_singles_including_caps_and_empty_answers(self):
        tok = ByteTokenizer()
        rows = [tok.encode(text) for text in ('„Wolf!“', 'kühl…', '3.14 ', 'U.S. ',
                                             '!later', '--- later', 'x' * 128, 'Wolf..')]
        rows[-1].append(tok.eos_token_id)
        prepared = [local_runtime(tok, FakeModel([]), context_limit=512).prepare('x' * (index + 1)) for index in range(len(rows))]
        batched = list(local_runtime(tok, FakeModel(rows)).generate(prepared, 8))
        singles = list(local_runtime(tok, FakeModel(rows)).generate(prepared, 1))
        self.assertEqual(batched, singles)
        self.assertEqual([row['stop_reason'] for row in batched],
                         ['punctuation', 'punctuation', 'whitespace', 'whitespace',
                          'empty_punctuation', 'empty_whitespace', 'token_limit', 'eos'])

    def test_new_protocol_is_reported_by_local_and_api_evaluations(self):
        tok = ByteTokenizer()
        fixture_dataset = (['Context Wolf!Suffix'], {'dataset_examples': 1, 'skipped_examples': []})
        for backend in ('local', 'api'):
            with patch.object(dataset, 'load_lambada_examples', return_value=fixture_dataset):
                if backend == 'local':
                    with patch.object(local.LocalModel, 'load',
                                      return_value=local_runtime(tok, FakeModel([tok.encode('Wolf!')]))):
                        metrics, rows = word.evaluate('/model', 'en', device='cpu')
                else:
                    with api([ResponseStream([event('response.output_text.delta', delta='Wolf!')])]):
                        metrics, rows = word.evaluate('fixture/model', 'en', backend='api')
            self.assertEqual(metrics['protocol'], 'case_sensitive_punctuation_boundary_sharp_s_v2')
            self.assertEqual(metrics['scoring_version'], metrics['protocol'])
            self.assertEqual(metrics['word_boundary'], 'punctuation_first_word_v1')
            self.assertFalse(metrics['oracle_target_token_length'])
            self.assertEqual(rows[0]['normalized_target_word'], 'Wolf!Suffix')
            self.assertEqual(metrics['correct'], 0)


if __name__ == '__main__':
    unittest.main()
