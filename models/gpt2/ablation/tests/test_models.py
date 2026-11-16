import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from transformers import GPT2LMHeadModel

from src.models.initialization import derive_smaller_model, load_training_model, make_gpt2_config
from src.training.checkpoint import save_model_artifact
from dataclasses import replace
from tests.support import DummyTokenizer, tiny_spec


class ModelTest(unittest.TestCase):
    def test_initialization_backend_loading_and_portable_final(self) -> None:
        spec = tiny_spec()
        torch.manual_seed(42)
        canonical = GPT2LMHeadModel(make_gpt2_config(spec, 512))
        smaller = derive_smaller_model(canonical, spec, 300)
        self.assertTrue(torch.equal(
            canonical.transformer.h[0].attn.c_attn.weight,
            smaller.transformer.h[0].attn.c_attn.weight,
        ))
        self.assertTrue(torch.equal(
            canonical.get_input_embeddings().weight[:300],
            smaller.get_input_embeddings().weight,
        ))
        self.assertEqual(
            smaller.get_input_embeddings().weight.data_ptr(),
            smaller.lm_head.weight.data_ptr(),
        )
        with patch("src.models.initialization.GPT2LMHeadModel.from_pretrained", return_value=smaller) as load:
            self.assertIs(load_training_model("init", spec=spec, vocab_size=300), smaller)
        self.assertEqual(load.call_args.kwargs["attn_implementation"], "sdpa")

        with tempfile.TemporaryDirectory() as temporary:
            final = Path(temporary) / "final"
            smaller.config.use_cache = True
            save_model_artifact(final, smaller, DummyTokenizer())
            config = json.loads((final / "config.json").read_text())
            self.assertTrue(config["use_cache"])
            self.assertFalse(any("attn_implementation" in key for key in config))
            portable = GPT2LMHeadModel.from_pretrained(
                final, local_files_only=True, attn_implementation="sdpa"
            )
            self.assertEqual(portable.config._attn_implementation, "sdpa")
            original_files = {p.name: p.read_bytes() for p in final.iterdir() if p.is_file()}
            changed = load_training_model(final, spec=replace(spec, attn_pdrop=.1, resid_pdrop=.1, embd_pdrop=.1), vocab_size=300)
            self.assertEqual(changed.transformer.h[0].attn.attn_dropout.p, .1)
            self.assertEqual(changed.transformer.h[0].attn.resid_dropout.p, .1)
            self.assertEqual(changed.transformer.drop.p, .1)
            self.assertTrue(torch.equal(changed.get_input_embeddings().weight, smaller.get_input_embeddings().weight))
            self.assertEqual(original_files, {p.name:p.read_bytes() for p in final.iterdir() if p.is_file()})

