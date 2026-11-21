"""Local model validation, token boundaries and greedy batch correctness."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from tokenizers import decoders

from tasks.lambada.src import greedy as word, data as dataset, local, api as backend_api, scoring
from tasks.lambada.src.local import BYTE_DECODER
from tasks.lambada.src.local import FirstWord


def local_runtime(tokenizer, model, device='cpu', context_limit=None, bos='auto'):
    return local.LocalModel('/model', tokenizer, model, device, context_limit=context_limit, bos=bos)


def generate_word(context, tokenizer, model, device, context_limit=512):
    runner = local_runtime(tokenizer, model, device, context_limit)
    return list(runner.generate([runner.prepare(context)], 1))[0]


class ByteTokenizer:
    unk_token_id, bos_token_id, eos_token_id, pad_token_id = 0, 1, 2, 3
    all_special_ids = [0, 1, 2, 3, 4]
    is_fast = True
    model_max_length = 10**30
    backend_tokenizer = SimpleNamespace(decoder=decoders.ByteLevel())

    def __init__(self):
        self.pieces = {value + 5: char for char, value in BYTE_DECODER.items()}
        self.pieces.update({i: name for i, name in enumerate(("<unk>", "<s>", "</s>", "<pad>", "<mask>"))})
        self.pieces[300] = "".join(self.pieces[value + 5] for value in b" and it")

    def __len__(self):
        return max(self.pieces) + 1

    def get_vocab(self):
        return {piece: index for index, piece in self.pieces.items()}

    def encode(self, text, **kwargs):
        return ([self.bos_token_id] if kwargs.get("add_special_tokens", False) and self.bos_token_id is not None else []) + [value + 5 for value in text.encode("utf-8")]

    def convert_ids_to_tokens(self, ids):
        return [self.pieces[index] for index in ids]

    def decode(self, ids, **kwargs):
        return decoders.ByteLevel().decode(self.convert_ids_to_tokens(ids))



class FakeModel:
    """Emit supplied token rows, honoring per-row EOS and stopping criteria."""
    def __init__(self, rows):
        self.rows = iter(rows)
        self.calls = []
        self.config = SimpleNamespace(model_type='gpt2', vocab_size=301, n_positions=512, bos_token_id=1, eos_token_id=2, pad_token_id=3, is_encoder_decoder=False)
        self.dtype = 'unknown'

    def get_input_embeddings(self):
        return SimpleNamespace(num_embeddings=self.config.vocab_size)

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        ids = kwargs['input_ids'].clone()
        rows = [next(self.rows) for _ in ids]
        config = kwargs['generation_config']
        eos_ids = config.eos_token_id
        eos_ids = eos_ids if isinstance(eos_ids, list) else [eos_ids]
        done = torch.zeros(len(rows), dtype=torch.bool)
        for step in range(config.max_new_tokens):
            tokens = [config.pad_token_id if done[i] else row[step] for i, row in enumerate(rows)]
            ids = torch.cat((ids, torch.tensor(tokens).unsqueeze(1)), dim=1)
            done |= kwargs['stopping_criteria'](ids, None)
            done |= torch.tensor([token in eos_ids for token in tokens])
            if done.all():
                break
        return ids


def validation_model():
    model = FakeModel([])
    model.config.n_positions = 1024
    return model


def tiny_models(tokenizer, window):
    """Random CPU models only; no pretrained files or downloads."""
    from transformers import GPT2Config, GPT2LMHeadModel, LlamaConfig, LlamaForCausalLM
    for token in range(261, 300):
        tokenizer.pieces[token] = f'unused{token}'
    special = dict(vocab_size=301, bos_token_id=1, eos_token_id=2, pad_token_id=3)
    yield GPT2LMHeadModel(GPT2Config(
        **special, n_positions=window, n_embd=16, n_layer=1, n_head=2,
    )).eval()
    yield LlamaForCausalLM(LlamaConfig(
        **special, max_position_embeddings=window, hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
    )).eval()


class LocalTests(unittest.TestCase):
    def test_model_path_requires_a_directory_not_a_legacy_bin_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ('config.json', 'tokenizer.json', 'pytorch_model.bin'):
                (root / name).touch()
            with (patch('transformers.AutoTokenizer.from_pretrained') as tokenizer,
                  patch('transformers.AutoModelForCausalLM.from_pretrained') as model,
                  self.assertRaisesRegex(FileNotFoundError, 'not ready')):
                local.LocalModel.load(str(root / 'pytorch_model.bin'), device='cpu')
            tokenizer.assert_not_called()
            model.assert_not_called()

    def test_word_boundaries_multitoken_utf8_and_strict_surface(self):
        tokenizer = ByteTokenizer()
        for text, expected in (
            ("  Shane next", "Shane"),
            ("\t\nkühl\u2003next", "kühl"),
            ("Shane, next", "Shane,"),
            ("shane next", "shane"),
            ("mother-in-law next", "mother-in-law"),
            ("can't next", "can't"),
            ("\ufffd next", "\ufffd"),
        ):
            with self.subTest(text=text):
                result = generate_word("context", tokenizer, FakeModel([tokenizer.encode(text)]), "cpu")
                self.assertEqual(result["prediction"], expected)
                self.assertTrue(result["completed"])
                self.assertEqual(result["stop_reason"], "whitespace")
        result = generate_word("context", tokenizer, FakeModel([[300]]), "cpu")
        self.assertEqual((result["prediction"], result["generated_text"]), ("and", " and it"))
        criterion = FirstWord(tokenizer)
        self.assertIsNone(criterion.inspect([value + 5 for value in b" k\xc3"]))
        self.assertEqual(criterion.inspect(tokenizer.encode(" kühl "))["prediction"], "kühl")

    def test_eos_invalid_bytes_special_tokens_and_limits(self):
        tokenizer = ByteTokenizer()
        rows = (
            (tokenizer.encode("Shane") + [2], "eos", "Shane"),
            (tokenizer.encode(" \t") + [2], "empty_eos", ""),
            (tokenizer.encode("Sh") + [4], "special_token", ""),
            ([value + 5 for value in b"Sh\xc3 "], "invalid_utf8", ""),
            ([value + 5 for value in b"Sh\xc3"] + [2], "invalid_utf8", ""),
            (tokenizer.encode("a" * 128), "token_limit", ""),
        )
        for ids, reason, expected in rows:
            with self.subTest(reason=reason):
                result = generate_word("context", tokenizer, FakeModel([ids]), "cpu")
                self.assertEqual(result["stop_reason"], reason)
                self.assertEqual(result["prediction"], expected)
                self.assertEqual(result["completed"], bool(expected))
        model = FakeModel([tokenizer.encode("abcd ")])
        result = generate_word("x" * 508, tokenizer, model, "cpu")
        self.assertEqual((result["generation_budget"], result["stop_reason"]), (3, "context_limit"))
        self.assertEqual(len(result["generated_ids"]) + 509, 512)
        with self.assertRaisesRegex(ValueError, "no generation space"):
            generate_word("x" * 511, tokenizer, FakeModel([]), "cpu")

    def test_empty_final_rejected_before_tokenizer_or_weights_load(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch("transformers.AutoTokenizer.from_pretrained") as tokenizer, \
                patch("transformers.AutoModelForCausalLM.from_pretrained") as model:
            with self.assertRaisesRegex(FileNotFoundError, "not ready"):
                local.LocalModel.load(directory)
            tokenizer.assert_not_called()
            model.assert_not_called()

    def test_loader_rejects_missing_cuda_before_loading_tokenizer_or_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in ("config.json", "tokenizer.json", "model.safetensors"):
                (Path(directory) / name).touch()
            with (
                patch("torch.cuda.is_available", return_value=False),
                patch("transformers.AutoTokenizer.from_pretrained") as tokenizer,
                patch("transformers.AutoModelForCausalLM.from_pretrained") as model,
                self.assertRaisesRegex(RuntimeError, "CUDA is required"),
            ):
                local.LocalModel.load(directory)
            tokenizer.assert_not_called()
            model.assert_not_called()

    def test_llama_uses_own_ids_one_bos_and_own_context_window(self):
        tokenizer = ByteTokenizer()
        tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id = 400, 401, 401
        tokenizer.unk_token_id = None
        tokenizer.all_special_ids = [0, 1, 2, 3, 4, 400, 401]
        tokenizer.pieces.update({400: "<bos>", 401: "<eos>"})
        model = FakeModel([tokenizer.encode("Straße ")])
        model.config = SimpleNamespace(model_type="llama", vocab_size=402, max_position_embeddings=8192,
                                       bos_token_id=400, eos_token_id=401, pad_token_id=None, is_encoder_decoder=False)
        llama_config = model.config
        model.generation_config = SimpleNamespace(bos_token_id=400, eos_token_id=401, pad_token_id=None,
                                                do_sample=True, temperature=0.6)
        local_runtime(tokenizer, model)
        result = generate_word("x" * 600, tokenizer, model, "cpu", context_limit=8192)
        self.assertTrue(result["completed"])
        call = model.calls[0]
        self.assertEqual(call['input_ids'][0].tolist(), [400, *tokenizer.encode('x' * 600)])
        self.assertEqual(call['generation_config'].max_new_tokens, 128)
        self.assertFalse(call['generation_config'].do_sample)
        self.assertEqual(call['generation_config'].eos_token_id, 401)
        model = FakeModel([tokenizer.encode('abcd ')])
        model.config = llama_config
        result = generate_word('x' * 8188, tokenizer, model, 'cpu', context_limit=8192)
        self.assertEqual((result['generation_budget'], result['stop_reason']), (3, 'context_limit'))
        with self.assertRaisesRegex(ValueError, 'no generation space'):
            generate_word('x' * 8191, tokenizer, model, 'cpu', context_limit=8192)

    def test_byte_mapping_rejects_non_byte_tokenizers(self):
        tokenizer = ByteTokenizer()
        local.TokenBytes(tokenizer)
        tokenizer.pieces[301] = '▁word'
        with self.assertRaisesRegex(ValueError, 'non-byte-level'):
            local.TokenBytes(tokenizer)

    def test_model_and_tokenizer_validation_covers_shared_contract(self):
        tokenizer = ByteTokenizer()
        model = validation_model()
        model.config.pad_token_id = None
        local_runtime(tokenizer, model)
        model.config.vocab_size = 31
        with self.assertRaisesRegex(ValueError, "vocab size"):
            local_runtime(tokenizer, model)

    def test_saved_generation_config_is_ignored_but_model_config_is_checked(self):
        tokenizer = ByteTokenizer()
        for backend, model_type in (("gpt2", "gpt2"), ("llama3", "llama")):
            model = validation_model()
            model.config.model_type = model_type
            model.config.max_position_embeddings = 8192
            model.generation_config = SimpleNamespace(
                bos_token_id=7, eos_token_id=[8, 9], pad_token_id=10,
                do_sample=True, forced_bos_token_id=11,
            )
            with self.subTest(backend=backend):
                local_runtime(tokenizer, model)
                for name in ("bos_token_id", "eos_token_id", "pad_token_id"):
                    for mismatch in (7, [7]):
                        with self.subTest(field=name, value=mismatch), \
                                patch.object(model.config, name, mismatch), \
                                self.assertRaisesRegex(ValueError, f"Model {name}"):
                            local_runtime(tokenizer, model)
                with patch.object(model.config, "is_encoder_decoder", True, create=True), \
                        self.assertRaisesRegex(ValueError, "decoder-only"):
                    local_runtime(tokenizer, model)

    def test_tiny_real_models_match_manual_greedy_decoding(self):
        tokenizer = ByteTokenizer()
        with torch.random.fork_rng(devices=[]), torch.inference_mode():
            torch.manual_seed(42)
            for model in tiny_models(tokenizer, 32):
                with self.subTest(model=type(model).__name__):
                    baseline = generate_word('x', tokenizer, model, 'cpu', context_limit=32)
                    # Deliberately conflict with the protocol to detect inherited settings.
                    model.generation_config.do_sample = True
                    model.generation_config.forced_bos_token_id = 4
                    model.generation_config.bos_token_id = 7
                    model.generation_config.eos_token_id = [8, 9]
                    model.generation_config.pad_token_id = 10
                    actual = generate_word('x', tokenizer, model, 'cpu', context_limit=32)
                    self.assertEqual(actual, baseline)
                    ids = torch.tensor([[1, *tokenizer.encode('x')]])
                    expected = []
                    stopping = local.FirstWord(tokenizer)
                    for _ in range(30):
                        token = int(model(input_ids=ids).logits[0, -1].argmax())
                        expected.append(token)
                        ids = torch.cat((ids, torch.tensor([[token]])), dim=1)
                        if stopping.inspect(expected) is not None:
                            break
                    self.assertEqual(actual['generated_ids'], expected)

    def test_mixed_stops_utf8_caps_and_padding_match_singles(self):
        tok = ByteTokenizer()
        sequences = [tok.encode('Straße '), tok.encode('hello') + [2], [4],
                     [value + 5 for value in b'k\xc3\xbchl '],
                     [value + 5 for value in b'k\xc3 '], tok.encode('x' * 128),
                     tok.encode('and it'), [2], tok.encode('last ') ]
        examples = [{'context':'x' * (i + 1)} for i in range(len(sequences))]
        prepared = [local_runtime(tok, FakeModel([]), context_limit=512).prepare(row["context"]) for row in examples]
        batch_model = FakeModel(sequences)
        batched = list(local_runtime(tok, batch_model).generate(prepared, 8))
        single_model = FakeModel(sequences)
        singles = list(local_runtime(tok, single_model).generate(prepared, 1))
        self.assertEqual(batched, singles)
        self.assertEqual([r['stop_reason'] for r in batched],
                         ['whitespace','eos','special_token','whitespace','invalid_utf8',
                          'token_limit','whitespace','empty_eos','whitespace'])
        self.assertEqual([len(c['input_ids']) for c in batch_model.calls], [8, 1])
        first = batch_model.calls[0]
        self.assertEqual(first['input_ids'][0].tolist(), [3] * 7 + [1, *tok.encode('x')])
        self.assertEqual(first['attention_mask'][0].tolist(), [0] * 7 + [1, 1])
        self.assertEqual(batched[2]['generated_ids'], [4])
        self.assertEqual(batched[7]['generated_ids'], [2])

    def test_budget_groups_preserve_order_and_individual_limits(self):
        tok = ByteTokenizer()
        for window in (512, 8192):
            lengths = [10, 20, window-3, window-3, window-2, 5]
            rows = [tok.encode('ok '), tok.encode('ok '), tok.encode('xxx'),
                    tok.encode('x '), tok.encode('xx'), tok.encode('ok ')]
            examples = [{'context':'x'*n} for n in lengths]
            model = FakeModel(rows)
            model.config.n_positions = window
            runner = local_runtime(tok, model)
            prepared = [runner.prepare(row["context"]) for row in examples]
            results = list(runner.generate(prepared, 8))
            self.assertEqual([r['generation_budget'] for r in results], [128,128,2,2,1,128])
            self.assertEqual([r['stop_reason'] for r in results],
                             ['whitespace','whitespace','context_limit','whitespace','context_limit','whitespace'])
            self.assertEqual([len(c['input_ids']) for c in model.calls], [2,2,1,1])
            for call in model.calls:
                self.assertLessEqual(call['input_ids'].shape[1] + call['generation_config'].max_new_tokens, window)

    def test_tiny_gpt2_and_llama_batches_match_singles(self):
        tok = ByteTokenizer()
        examples = [{'context':s} for s in ['x', 'different length', 'two']]
        prepared = [local_runtime(tok, FakeModel([]), context_limit=160).prepare(row["context"]) for row in examples]
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42)
            for model in tiny_models(tok, 160):
                with self.subTest(model=type(model).__name__):
                    single = list(local_runtime(tok, model).generate(prepared, 1))
                    batch = list(local_runtime(tok, model).generate(prepared, 8))
                    self.assertEqual(batch, single)

    def test_oom_is_propagated_without_retry(self):
        from unittest.mock import Mock
        model = Mock()
        model.config = validation_model().config
        model.get_input_embeddings.return_value = SimpleNamespace(num_embeddings=301)
        model.generate.side_effect = torch.OutOfMemoryError('fixture OOM')
        tok = ByteTokenizer()
        prepared = [local_runtime(tok, FakeModel([]), context_limit=512).prepare('context')]
        with self.assertRaises(torch.OutOfMemoryError):
            list(local_runtime(tok, model).generate(prepared, 8))
        self.assertEqual(model.generate.call_count, 1)


if __name__ == '__main__':
    unittest.main()
