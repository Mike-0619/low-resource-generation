"""Fast tests for evaluation invariants; no model download is required."""

import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path


from tasks.paraphrase.src.evaluation.aggregate import run_build_database
from tasks.paraphrase.src.data.formatter import build_chat_prompt, get_format_fn
from tasks.paraphrase.src.evaluation.metrics import evaluate_bleu


class FakeBleu:
    def compute(self, **kwargs):
        return {"score": float(len(kwargs["predictions"][0]))}


class FakeTokenizer:
    eos_token = "<eos>"
    eos_token_id = 99
    pad_token_id = 0

    def __call__(self, text, add_special_tokens=False):
        length = 300 if text.startswith(" ") else 400
        return {"input_ids": list(range(length))}


class PipelineInvariantTests(unittest.TestCase):
    def test_bleu_normalization_is_shared(self):
        first = evaluate_bleu(
            ["Text ,  here ."], ["Source ."], ["Text, here."], FakeBleu()
        )
        second = evaluate_bleu(
            ["Text, here."], ["Source."], ["Text , here ."], FakeBleu()
        )
        self.assertEqual(first["predictions"], second["predictions"])
        self.assertEqual(first["sources"], second["sources"])
        self.assertEqual(first["references"], second["references"])
        self.assertEqual(first["config"], second["config"])

    def test_formatter_preserves_supervised_target(self):
        formatted = get_format_fn(FakeTokenizer(), 256)(
            {"sentence1": ["source"], "sentence2": ["target"]}
        )
        self.assertEqual(len(formatted["input_ids"][0]), 256)
        self.assertTrue(any(label != -100 for label in formatted["labels"][0]))

    def test_few_shot_example_preserves_facts(self):
        prompt = str(build_chat_prompt("Test sentence.", "few_shot"))
        self.assertIn("Nasdaq Composite Index", prompt)
        self.assertIn("1,496.90", prompt)
        self.assertNotIn("Standard & Poor", prompt)
        self.assertIn("Instruction: Paraphrase the following sentence.", prompt)
        self.assertNotIn("Rewrite", prompt)
        self.assertNotIn("rewrite", prompt)

    def test_central_table_aligns_by_sample_id(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            details = [
                {"sample_id": "mrpc-test-1", "src": "A", "ref": "B", "pred": "C"},
                {"sample_id": "mrpc-test-2", "src": "D", "ref": "E", "pred": "F"},
            ]
            first = directory / "first.json"
            second = directory / "second.json"
            first.write_text(json.dumps({"details": details}), encoding="utf-8")
            second.write_text(
                json.dumps({"details": [dict(row, pred=row["pred"] + "2") for row in details]}),
                encoding="utf-8",
            )
            output_json = directory / "central.json"
            run_build_database(
                Namespace(
                    reports=[str(first), str(second)],
                    names=["First", "Second"],
                    output_json=str(output_json),
                    output_csv=str(directory / "central.csv"),
                )
            )
            rows = json.loads(output_json.read_text(encoding="utf-8"))
            self.assertEqual(rows[0]["sample_id"], "mrpc-test-1")
            self.assertEqual(rows[0]["First"], "C")
            self.assertEqual(rows[0]["Second"], "C2")


if __name__ == "__main__":
    unittest.main()
