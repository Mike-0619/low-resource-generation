"""Generic model selection, tokenizer decoders and actual generation controls."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from tokenizers import decoders
from tasks.lambada.src import api as backend_api, greedy, local
from test_api import api, response
from test_local import ByteTokenizer, FakeModel, validation_model, local_runtime


class PieceTokenizer(ByteTokenizer):
    def __init__(self, decoder):
        super().__init__()
        self.backend_tokenizer = SimpleNamespace(decoder=decoder)
        self.pieces = {0: '<unk>', 1: '<s>', 2: '</s>', 3: '<pad>', 4: '<mask>',
                       5: '▁k', 6: '<0xC3>', 7: '<0xBC>', 8: 'hl', 9: '▁next',
                       10: '!', 11: '<0xFF>', 12: '▁Wolf', 13: '“', 14: '.',
                       15: '<0xE2>', 16: '<0x80>', 17: '<0xA6>', 18: '▁Wolf▁next'}

    def encode(self, text, **kwargs):
        return ([1] if kwargs.get('add_special_tokens', False) else []) + [12]

    def decode(self, ids, **kwargs):
        return self.backend_tokenizer.decoder.decode(self.convert_ids_to_tokens(ids))


def sentencepiece():
    return decoders.Sequence([decoders.Replace('▁', ' '), decoders.ByteFallback(),
                              decoders.Fuse(), decoders.Strip(' ', 1, 0)])


class GenericTests(unittest.TestCase):
    def test_arbitrary_local_and_api_models_and_both_languages(self):
        for backend in ('local', 'api'):
            for language in ('en', 'de'):
                args = greedy.parse_args(['--backend', backend, '--model', 'new-family/model-123',
                                          '--language', language, '--output_dir', '/unused'])
                self.assertEqual(args.model, 'new-family/model-123')

    def test_reasoning_is_omitted_unless_explicit_and_errors_are_not_retried(self):
        for effort in (None, 'none', 'low'):
            with api([response('Wolf!')]) as client:
                backend_api.request(client, {'context': 'context'}, 'new/model', effort)
                args = client.responses.create.call_args.kwargs
                self.assertEqual(args['temperature'], 0)
                self.assertFalse(args['store'])
                if effort is None:
                    self.assertNotIn('reasoning', args)
                else:
                    self.assertEqual(args['reasoning'], {'effort': effort})
        with api([ValueError('unsupported parameter')]) as client:
            with self.assertRaisesRegex(ValueError, 'unsupported parameter'):
                backend_api.request(client, {'context': 'context'}, 'new/model', 'none')
            self.assertEqual(client.responses.create.call_count, 1)

    def test_api_german_preserves_boldt_ids(self):
        metadata = {'sample_mapping': [{'dataset_index': 1147, 'english_dataset_index': 1154}],
                    'dataset_examples': 5150, 'eligible_examples': 5147, 'skipped_examples': []}
        with patch.object(greedy.data, 'load_lambada_examples', return_value=(['Kontext Wolf'], metadata)), api([response('Wolf!')]):
            metrics, rows = greedy.evaluate('arbitrary/model', 'de', backend='api')
        self.assertEqual(rows[0]['sample_id'], 'de:boldt:1147')
        self.assertEqual(rows[0]['english_dataset_index'], 1154)
        self.assertEqual(metrics['correct'], 1)
        self.assertEqual(metrics['api_protocol'], 'responses')

    def test_supported_decoder_bytes_match_native_valid_text(self):
        for decoder in (sentencepiece(), decoders.Metaspace(), decoders.Metaspace(prepend_scheme='never'),
                        decoders.Sequence([decoders.ByteFallback(), decoders.Fuse(), decoders.Metaspace()])):
            tok = PieceTokenizer(decoder)
            adapter = local.TokenBytes(tok)
            for pieces in (['▁Wolf', '!'], ['▁Wolf▁next'], ['▁k', '<0xC3>', '<0xBC>', 'hl'], ['<0xC3>', '<0xBC>']):
                with self.subTest(decoder=adapter.name, pieces=pieces):
                    self.assertEqual(adapter.decode(pieces).decode('utf-8'), decoder.decode(pieces))

    def test_sentencepiece_pending_invalid_and_post_boundary_bytes(self):
        tok = PieceTokenizer(sentencepiece())
        inspect = local.FirstWord(tok).inspect
        self.assertIsNone(inspect([5, 6]))
        self.assertEqual(inspect([5, 6, 7, 8, 9])['prediction'], 'kühl')
        self.assertEqual(inspect([5, 11, 10])['stop_reason'], 'invalid_utf8')
        self.assertEqual(inspect([12, 13, 11])['prediction'], 'Wolf')
        self.assertIsNone(inspect([12, 15, 16]))
        self.assertEqual(inspect([12, 15, 16, 17])['stop_reason'], 'punctuation')

    def test_bytelevel_with_existing_gpt2_cleanup_retains_strict_utf8(self):
        tok = ByteTokenizer()
        decoder = decoders.Sequence([decoders.ByteLevel(), decoders.Strip(' ', 1, 0),
                                     decoders.Replace('\n ', '\n')])
        tok.backend_tokenizer = SimpleNamespace(decoder=decoder)
        adapter = local.TokenBytes(tok)
        for text in (' Wolf!', '  Wolf', '\n Wolf', 'Wolf\n next'):
            pieces = tok.convert_ids_to_tokens(tok.encode(text))
            self.assertEqual(adapter.decode(pieces).decode('utf-8'), decoder.decode(pieces))
            expected = local.inspect_text(text, final=True)
            self.assertEqual(local.FirstWord(tok).inspect(tok.encode(text) + [2]), expected)
        self.assertEqual(local.FirstWord(tok).inspect([value + 5 for value in b'Wo\xfflf!'])['stop_reason'], 'invalid_utf8')

    def test_sentencepiece_batches_equal_singles(self):
        tok = PieceTokenizer(sentencepiece())
        rows = [[5, 6, 7, 8, 9], [12, 10], [5, 11, 10], [12, 2]]
        runner = local_runtime(tok, FakeModel(rows))
        prepared = [runner.prepare('context') for _ in rows]
        batch = list(runner.generate(prepared, 8))
        single = list(local_runtime(tok, FakeModel(rows)).generate(prepared, 1))
        self.assertEqual(batch, single)
        self.assertEqual([row['stop_reason'] for row in batch], ['whitespace', 'punctuation', 'invalid_utf8', 'eos'])

    def test_unknown_decoder_and_slow_tokenizers_are_rejected(self):
        tok = ByteTokenizer()
        tok.backend_tokenizer = SimpleNamespace(decoder=decoders.WordPiece())
        with self.assertRaisesRegex(ValueError, 'Unsupported tokenizer decoder'):
            local.TokenBytes(tok)
        tok.is_fast = False
        with self.assertRaisesRegex(ValueError, 'fast tokenizer'):
            local.TokenBytes(tok)

    def test_multiple_eos_and_padding_fallback_use_model_config(self):
        tok = ByteTokenizer()
        tok.pad_token_id = None
        rows = [tok.encode('Wolf') + [4], tok.encode('longer ')]
        model = FakeModel(rows)
        model.config.eos_token_id, model.config.pad_token_id = [2, 4], None
        model.generation_config = SimpleNamespace(eos_token_id=7, pad_token_id=8)
        runner = local_runtime(tok, model)
        prepared = [runner.prepare(text) for text in ('x', 'long context')]
        results = list(runner.generate(prepared, 8))
        self.assertEqual(results[0]['stop_reason'], 'eos')
        config = model.calls[0]['generation_config']
        self.assertEqual(config.eos_token_id, [2, 4])
        self.assertEqual(config.pad_token_id, 2)
        self.assertEqual(model.calls[0]['input_ids'][0, 0], 2)

    def test_padded_vocabulary_and_unmapped_predictions(self):
        tok = ByteTokenizer()
        model = validation_model()
        model.config.vocab_size = 320
        local_runtime(tok, model)
        inspect = local.FirstWord(tok).inspect
        self.assertEqual(inspect([310])['stop_reason'], 'invalid_token')
        self.assertEqual(inspect(tok.encode('Wolf!') + [310])['prediction'], 'Wolf')

    def test_context_limits_inference_override_and_unknown_window(self):
        tok = ByteTokenizer()
        model = validation_model()
        model.config.n_positions = None
        model.config.max_position_embeddings = 1024
        self.assertEqual(local_runtime(tok, model).context_limit, 1024)
        tok.model_max_length = 256
        self.assertEqual(local_runtime(tok, model).context_limit, 256)
        self.assertEqual(local_runtime(tok, model, context_limit=128).context_limit, 128)
        with self.assertRaisesRegex(ValueError, 'within'):
            local_runtime(tok, model, context_limit=512)
        tok.model_max_length = 10**30
        model.config.max_position_embeddings = None
        with self.assertRaisesRegex(ValueError, 'Cannot infer'):
            local_runtime(tok, model)
        self.assertEqual(local_runtime(tok, model, context_limit=512).context_limit, 512)

    def test_bos_modes_and_missing_bos(self):
        tok = ByteTokenizer()
        model = FakeModel([])
        for policy in ('auto', 'always'):
            self.assertEqual(local_runtime(tok, model, bos=policy).prepare('context')['input_ids'], [1, *tok.encode('context')])
        self.assertEqual(local_runtime(tok, model, bos='never').prepare('context')['input_ids'], tok.encode('context'))
        tok.bos_token_id = model.config.bos_token_id = None
        with self.assertRaisesRegex(ValueError, 'requires a BOS'):
            local_runtime(tok, model, bos='always').prepare('context')
        self.assertEqual(local_runtime(tok, model, bos='auto').prepare('context')['input_ids'], tok.encode('context'))

    def test_batches_share_inspection_but_keep_independent_stop_state(self):
        tok = ByteTokenizer()
        rows = [tok.encode('Wolf!'), tok.encode('U.S. '), tok.encode('Wolf') + [4],
                tok.encode('x' * 128), tok.encode('Straße...'), tok.encode('last ')]
        model = FakeModel(rows)
        model.config.eos_token_id = [2, 4]
        with patch.object(local, 'FirstWord', wraps=local.FirstWord) as create:
            runner = local_runtime(tok, model)
            prepared = [runner.prepare('x' * size) for size in (1, 2, 3, 383, 4, 5)]
            batch = list(runner.generate(prepared, 2))
        create.assert_called_once()
        singles_model = FakeModel(rows)
        singles_model.config.eos_token_id = [2, 4]
        singles = list(local_runtime(tok, singles_model).generate(prepared, 1))
        self.assertEqual(batch, singles)
        self.assertEqual([item['stop_reason'] for item in batch],
                         ['punctuation', 'whitespace', 'eos', 'context_limit', 'punctuation', 'whitespace'])
        criteria = [call['stopping_criteria'][0] for call in model.calls]
        self.assertTrue(all(item.inspector is criteria[0].inspector for item in criteria))
        self.assertEqual(len({id(item.stops) for item in criteria}), len(criteria))

    def test_loader_rejects_unknown_decoder_before_loading_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in ('config.json', 'model.safetensors'):
                (Path(directory) / name).touch()
            tok = PieceTokenizer(decoders.WordPiece())
            config = validation_model().config
            config.is_encoder_decoder = False
            with (patch.object(local, 'resolve_device', return_value=torch.device('cpu')),
                  patch('transformers.AutoConfig.from_pretrained', return_value=config),
                  patch('transformers.AutoTokenizer.from_pretrained', return_value=tok),
                  patch('transformers.AutoModelForCausalLM.from_pretrained') as weights,
                  self.assertRaisesRegex(ValueError, 'Unsupported tokenizer decoder')):
                local.LocalModel.load(directory, device='cpu')
            weights.assert_not_called()

    def test_local_loader_precision_and_offline_flags_without_tokenizer_json(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in ('config.json', 'model.safetensors'):
                (Path(directory) / name).touch()
            for device, dtype, expected in (('cpu', 'auto', torch.float32), ('cpu', 'float16', torch.float16),
                                             ('cuda', 'auto', 'auto'), ('cuda', 'bfloat16', torch.bfloat16)):
                tok = ByteTokenizer()
                model = validation_model()
                model.config.is_encoder_decoder = False
                model.to = Mock(return_value=model)
                model.eval = Mock()
                with (self.subTest(device=device, dtype=dtype),
                      patch.object(local, 'resolve_device', return_value=torch.device(device)),
                      patch('transformers.AutoConfig.from_pretrained', return_value=model.config),
                      patch('transformers.AutoTokenizer.from_pretrained', return_value=tok) as tokenizer_load,
                      patch('transformers.AutoModelForCausalLM.from_pretrained', return_value=model) as weights_load):
                    local.LocalModel.load(directory, device=device, dtype=dtype)
                    self.assertEqual(weights_load.call_args.kwargs['dtype'], expected)
                    self.assertTrue(weights_load.call_args.kwargs['local_files_only'])
                    self.assertFalse(weights_load.call_args.kwargs['trust_remote_code'])
                    self.assertTrue(tokenizer_load.call_args.kwargs['use_fast'])


if __name__ == '__main__':
    unittest.main()
