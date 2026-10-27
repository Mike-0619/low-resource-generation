"""Tokenizer rules, fixed examples, output protection and CPU jobs."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from transformers import AutoTokenizer, PreTrainedTokenizerFast

from src import train_tokenizer, utils

from src.utils import PROJECT_ROOT


class TrainTokenizerTest(unittest.TestCase):
    def test_tokenizer_rules_and_saved_examples(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, text in [('b.train.txt', 'Grüße Welt!\nNoch eine Zeile.\n'),
                               ('a.train.txt', 'Hello world!\nHello again.\n')]:
                (root / name).write_text(text * 3, encoding='utf-8')
            backend, trainer = train_tokenizer.tokenizer_backend(128)
            backend.train([str(p) for p in sorted(root.glob('*.train.txt'))], trainer)
            for token, expected in {'<unk>': 0, '<s>': 1, '</s>': 2, '<pad>': 3, '<mask>': 4}.items():
                self.assertEqual(backend.token_to_id(token), expected, token)
            for text, expected in [('Ａ\u00a0Ｂ\tC \n D', ' A B\tC\n  D'),
                                   ('Hello  world!', ' Hello  world!'), ('Grüße\nWelt!', ' Grüße\n Welt!')]:
                with self.subTest(normalization=text):
                    self.assertEqual(backend.normalizer.normalize_str(text), expected)
            for text, expected in [('abc123! \nxyz', ['Ġabc', '1', '2', '3', '!Ċ', 'Ġxyz']),
                                   ('a' * 25, ['Ġ' + 'a' * 23, 'aa'])]:
                normalized = backend.normalizer.normalize_str(text)
                self.assertEqual([token for token, _ in backend.pre_tokenizer.pre_tokenize_str(normalized)], expected)
            tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, bos_token='<s>', eos_token='</s>',
                                               unk_token='<unk>', pad_token='<pad>', model_max_length=1024)
            tokenizer.save_pretrained(root / 'saved')
            loaded = AutoTokenizer.from_pretrained(root / 'saved', local_files_only=True)
            self.assertEqual(loaded.get_vocab(), tokenizer.get_vocab())
            # Small examples verified against the pinned official recipe, not a full vocabulary snapshot.
            examples = [('Hello again.', [1, 37, 67, 6, 2], 'Hello again.'),
                        ('Grüße\nWelt!', [1, 63, 27, 65, 5, 2], 'Grüße\nWelt!'),
                        ('new punctuation?!', [1, 28, 47, 24, 28, 0, 0, 20, 13, 23, 0, 12, 23, 18, 21, 20, 0, 5, 2], 'new nctation!')]
            for text, ids, decoded in examples:
                with self.subTest(text=text):
                    self.assertEqual(backend.encode(text).ids, ids)
                    self.assertEqual(loaded.encode(text), ids)
                    self.assertEqual(loaded.decode(ids, skip_special_tokens=True), decoded)
            self.assertEqual(loaded.encode('Hello again.', text_pair='Grüße\nWelt!'),
                             [1, 37, 67, 6, 2, 1, 63, 27, 65, 5, 2])

    def test_tokenizer_uses_condition_paths_and_refuses_existing_output(self):
        cfg, _ = utils.load_experiment('configs/English/10M.yaml')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = utils.experiment_paths(cfg, root)
            paths.cleaned.mkdir(parents=True)
            (paths.cleaned / 'toy.train.txt').write_text('A tiny test corpus.')
            paths.tokenizer.mkdir(parents=True)
            saved = paths.tokenizer / 'tokenizer.json'
            saved.write_text('existing tokenizer')
            with patch('src.train_tokenizer.tokenizer_backend') as backend:
                with self.assertRaises(FileExistsError):
                    train_tokenizer.train_tokenizer(cfg, root)
                backend.assert_not_called()
            self.assertEqual(saved.read_text(), 'existing tokenizer')


    def test_entrypoint_loads_config_and_dispatches(self):
        for language, sizes in [('English', ('10M', '100M')), ('German', ('10M', '108M'))]:
            for size in sizes:
                with self.subTest(language=language, size=size), patch.object(train_tokenizer, 'train_tokenizer') as run:
                    train_tokenizer.main(['--config', f'configs/{language}/{size}.yaml'])
                    run.assert_called_once()
                    cfg, root = run.call_args.args
                    self.assertEqual((cfg['dataset']['language'], cfg['dataset']['size']), (language, size))
                    self.assertEqual(root, PROJECT_ROOT)
