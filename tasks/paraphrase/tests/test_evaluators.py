"""Dependency-light tests for the independent metric evaluators."""

import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch


from tasks.paraphrase.src.evaluation.metrics import evaluate_bleu
from tasks.paraphrase.src.evaluation.parapluie import evaluate_parapluie
from tasks.paraphrase.src.evaluation.parapluie import run_parapluie_evaluate


class FakeBleu:
    def compute(self, **kwargs):
        return {"score": float(len(kwargs["predictions"][0]))}


class FakeParapluie:
    scorer = None

    def __init__(self):
        self.init_kwargs = None

    def init(self, **kwargs):
        self.init_kwargs = kwargs

    def compute(self, **kwargs):
        return {"scores": [-1.0, 1.0, 2.0]}


class EvaluatorTests(unittest.TestCase):
    def test_bleu_evaluator_normalizes_and_scores(self):
        result = evaluate_bleu(
            ["First .", "Second", "Third"],
            ["A .", "B", "C"],
            ["D .", "E", "F"],
            metric=FakeBleu(),
        )
        self.assertEqual(result["predictions"][0], "First.")
        self.assertEqual(result["bleu_ref"], 6.0)
        self.assertEqual(result["bleu_src"], 6.0)

    def test_parapluie_evaluator_does_not_call_bleu(self):
        metric = FakeParapluie()
        result = evaluate_parapluie(
            ["First .", "Second", "Third"],
            ["A .", "B", "C"],
            judge_model_id="judge/model",
            metric=metric,
            device="cpu",
        )
        self.assertAlmostEqual(result["parapluie"], 2.0 / 3.0)
        self.assertEqual(result["parapluie_median"], 1.0)
        self.assertAlmostEqual(result["parapluie_positive_ratio"], 2.0 / 3.0)
        self.assertEqual(result["predictions"][0], "First.")
        self.assertEqual(metric.init_kwargs["model"], "judge/model")

    def test_report_pipeline_preserves_bleu_and_merges_parapluie(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "base.json"
            output_path = Path(temp_dir) / "with_parapluie.json"
            input_path.write_text(
                json.dumps(
                    {
                        "config": {"bleu": {"tokenize": "intl"}},
                        "metrics": {"bleu_ref": 10.5, "bleu_src": 20.5},
                        "details": [
                            {
                                "sample_id": "mrpc-test-1",
                                "src": "Source .",
                                "pred": "Prediction .",
                                "ref": "Reference .",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            fake_result = {
                "parapluie": 0.75,
                "parapluie_median": 0.75,
                "parapluie_positive_ratio": 1.0,
                "scores": [0.75],
                "predictions": ["Prediction."],
                "sources": ["Source."],
                "config": {"model": "judge/model", "template": "FS-DIRECT"},
            }
            with patch(
                "tasks.paraphrase.src.evaluation.parapluie.evaluate_parapluie",
                return_value=fake_result,
            ) as evaluator:
                run_parapluie_evaluate(
                    Namespace(
                        from_report=str(input_path),
                        out_report=str(output_path),
                        report_path=None,
                        parapluie_report_path=None,
                        judge_model_id="judge/model",
                    )
                )

            report = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(report["metrics"]["bleu_ref"], 10.5)
            self.assertEqual(report["metrics"]["bleu_src"], 20.5)
            self.assertEqual(report["metrics"]["parapluie"], 0.75)
            self.assertEqual(report["details"][0]["sample_id"], "mrpc-test-1")
            evaluator.assert_called_once()

    def test_report_pipeline_requires_precomputed_bleu(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "base.json"
            input_path.write_text(
                json.dumps(
                    {
                        "metrics": {},
                        "details": [
                            {"src": "Source", "pred": "Prediction", "ref": "Reference"}
                        ],
                    }
                ),
                encoding="utf-8",
            )
            with patch("tasks.paraphrase.src.evaluation.parapluie.evaluate_parapluie") as evaluator:
                with self.assertRaisesRegex(ValueError, "missing.*BLEU"):
                    run_parapluie_evaluate(
                        Namespace(
                            from_report=str(input_path),
                            out_report=str(Path(temp_dir) / "output.json"),
                            report_path=None,
                            parapluie_report_path=None,
                            judge_model_id="judge/model",
                        )
                    )
            evaluator.assert_not_called()


if __name__ == "__main__":
    unittest.main()
