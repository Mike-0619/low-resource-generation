import tempfile
import unittest
from pathlib import Path

from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

from src.data.train_tokenizer import _new_backend, as_fast_tokenizer, truncate_backend
from src.training.utils import SPECIAL_TOKEN_IDS


class TokenizerTest(unittest.TestCase):
    def test_nested_byte_tokenizer(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            corpus = Path(temporary) / "corpus.txt"
            corpus.write_text(
                "hello world äöüß Ελληνικά 中文 😀 café\n" * 200,
                encoding="utf-8",
            )
            backend = _new_backend()
            backend.train([str(corpus)], BpeTrainer(
                vocab_size=512,
                min_frequency=2,
                initial_alphabet=ByteLevel.alphabet(),
                special_tokens=list(SPECIAL_TOKEN_IDS),
            ))
            full = as_fast_tokenizer(backend)
            small = as_fast_tokenizer(truncate_backend(backend.to_str(), min(270, len(full) - 1)))

            self.assertEqual(
                {token: small.convert_tokens_to_ids(token) for token in SPECIAL_TOKEN_IDS},
                SPECIAL_TOKEN_IDS,
            )
            self.assertFalse(set(ByteLevel.alphabet()) - set(small.get_vocab()))
            self.assertTrue(all(
                full.get_vocab()[token] == token_id
                for token, token_id in small.get_vocab().items()
            ))
            sample = "Deutsch äöüß Ελληνικά 中文 😀 café\n"
            ids = small(sample, add_special_tokens=False)["input_ids"]
            self.assertNotIn(small.unk_token_id, ids)
            self.assertEqual(small.decode(ids, clean_up_tokenization_spaces=False), sample)
