"""Explicit source chunks, special tokens, padding and masks."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, call

import torch

from src import data_utils


class DataUtilsTest(unittest.TestCase):
    def test_chunking_keeps_source_order_tails_and_special_tokens(self):
        tokenizer = Mock(bos_token_id=1, eos_token_id=2, side_effect=[
            {'input_ids': [[3, 4, 5, 6]]}, {'input_ids': [[7, 8]]},
        ])
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp)
            (data / 'b.train.txt').write_text('second')
            (data / 'a.train.txt').write_text('first')
            actual = data_utils.BaselineDataset(data, tokenizer, 5)
            self.assertEqual(actual.data, [[3, 4, 5], [6], [7, 8]])
            self.assertEqual([item.tolist() for item in actual],
                             [[1, 3, 4, 5, 2], [1, 6, 2], [1, 7, 8, 2]])
            self.assertEqual(tokenizer.call_args_list, [
                call(text=['first'], add_special_tokens=False),
                call(text=['second'], add_special_tokens=False),
            ])

    def test_padding_and_input_based_mask(self):
        batch = [torch.tensor([1, 5, 6, 2]), torch.tensor([1, 7, 2]), torch.tensor([1, 2, 8, 2])]
        inputs, targets, mask = data_utils.collate_batch(batch, 2)
        self.assertEqual(inputs.tolist(), [[1, 5, 6], [1, 7, 2], [1, 2, 8]])
        self.assertEqual(targets.tolist(), [[5, 6, 2], [7, 2, 2], [2, 8, 2]])
        self.assertEqual(mask.tolist(), [[True, True, True], [True, True, False], [True, False, True]])
