"""ParaPLUIE scoring for predictions and existing BLEU reports."""

import json
import statistics
from datetime import datetime
from pathlib import Path

from .metrics import compute_copy_ratio, compute_generation_diagnostics, normalize_pairs, normalize_triplets
from .reports import require_new_file, write_report


PARAPLUIE_TEMPLATE = "FS-DIRECT"
# The upstream Space's main branch no longer contains the Evaluate metric.
PARAPLUIE_REVISION = "3567093d8eb6f5fdf72b636e215de4072d99c654"


def _patch_apply_chat_template_to_tensor(metric_obj):
    """Work around ParaPLUIE tokenizers returning lists for tensor requests."""
    scorer = getattr(metric_obj, "scorer", None)
    tokenizer = getattr(scorer, "tokenizer", None)
    if tokenizer is None or getattr(tokenizer, "_parapluie_tensor_patch", False):
        return False

    import torch

    original = tokenizer.apply_chat_template

    def patched_apply_chat_template(*args, **kwargs):
        output = original(*args, **kwargs)
        if kwargs.get("return_tensors") is None or isinstance(output, torch.Tensor):
            return output

        try:
            candidate = output["input_ids"]
        except Exception:
            candidate = output

        if isinstance(candidate, torch.Tensor):
            return candidate
        if isinstance(candidate, list):
            if candidate and isinstance(candidate[0], list):
                return torch.tensor(candidate, dtype=torch.long)
            return torch.tensor([candidate], dtype=torch.long)
        return output

    tokenizer.apply_chat_template = patched_apply_chat_template
    tokenizer._parapluie_tensor_patch = True
    return True


def evaluate_parapluie(
    predictions,
    sources,
    judge_model_id,
    metric=None,
    device=None,
):
    """Evaluate ParaPLUIE independently from BLEU and return aggregate scores."""
    predictions, sources = normalize_pairs(predictions, sources)

    if device is None:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
    if metric is None:
        import evaluate

        metric = evaluate.load("qlemesle/parapluie", revision=PARAPLUIE_REVISION)

    print(f"Attempting ParaPLUIE evaluation using {judge_model_id} on {device}...")
    try:
        for attempt, use_chat_template in enumerate([True, False, None]):
            try:
                init_kwargs = {
                    "model": judge_model_id,
                    "device": device,
                    "template": PARAPLUIE_TEMPLATE,
                    "half_mode": device == "cuda",
                    "n_right_specials_tokens": 1,
                }
                if use_chat_template is not None:
                    init_kwargs["use_chat_template"] = use_chat_template

                metric.init(**init_kwargs)
                _patch_apply_chat_template_to_tensor(metric)
                result = metric.compute(sources=sources, hypotheses=predictions)
                scores = result.get("scores", []) if result else []
                if len(scores) != len(predictions) or not scores:
                    raise RuntimeError(
                        f"ParaPLUIE returned {len(scores)} scores for "
                        f"{len(predictions)} predictions"
                    )
                scores = [float(score) for score in scores]
                print(f"ParaPLUIE succeeded on attempt {attempt + 1}")
                return {
                    "parapluie": float(sum(scores) / len(scores)),
                    "parapluie_median": float(statistics.median(scores)),
                    "parapluie_positive_ratio": float(
                        sum(score > 0 for score in scores) / len(scores)
                    ),
                    "scores": scores,
                    "predictions": predictions,
                    "sources": sources,
                    "config": {
                        "model": judge_model_id,
                        "template": PARAPLUIE_TEMPLATE,
                        "device": device,
                        "normalizer": "src.evaluation.metrics.normalize_text",
                    },
                }
            except TypeError as exc:
                if "list indices must be integers" in str(exc) and attempt < 2:
                    continue
                raise
    except Exception as exc:
        raise RuntimeError(
            f"ParaPLUIE failed: {type(exc).__name__}: {str(exc)[:200]}"
        ) from exc


def load_predictions_from_report(report_path):
    with open(report_path, encoding="utf-8") as f:
        report = json.load(f)

    details = report.get("details", [])
    sources = [d.get("src", "") for d in details]
    predictions = [d.get("pred", "") for d in details]
    references = [d.get("ref", "") for d in details]

    if not sources or not predictions:
        raise ValueError("Input report has no usable details/src/pred entries")
    if not (len(sources) == len(predictions) == len(references)):
        raise ValueError("Input report has mismatched src/pred/ref counts")
    if any(not isinstance(value, str) for value in sources + predictions + references):
        raise ValueError("All src/pred/ref values must be strings")

    return report, sources, predictions, references


def run_parapluie_evaluate(args):
    """Score an existing BLEU report with ParaPLUIE."""
    from_report = getattr(args, "from_report", None) or getattr(args, "report_path", None)
    out_report = getattr(args, "out_report", None) or getattr(args, "parapluie_report_path", None)

    if not from_report or not out_report:
        raise ValueError("from_report/report_path and out_report/parapluie_report_path must be provided.")

    require_new_file(out_report)

    report_path = from_report
    if not Path(report_path).is_file():
        raise FileNotFoundError(f"Report not found: {report_path}")

    print(f"Loading prediction report: {report_path}")
    base_report, sources, predictions, references = load_predictions_from_report(report_path)

    base_metrics = base_report.get("metrics", {})
    missing_bleu = [
        name for name in ("bleu_ref", "bleu_src") if name not in base_metrics
    ]
    if missing_bleu:
        raise ValueError(
            "Input report is missing independently computed BLEU metrics: "
            + ", ".join(missing_bleu)
        )
    predictions, sources, references = normalize_triplets(
        predictions, sources, references
    )

    print("\n" + "="*35)
    print(f"BLEU vs Reference (from report): {base_metrics['bleu_ref']}")
    print(f"BLEU vs Source (from report):    {base_metrics['bleu_src']}")
    print("="*35)

    parapluie_results = evaluate_parapluie(
        predictions, sources, args.judge_model_id
    )
    predictions = parapluie_results["predictions"]
    sources = parapluie_results["sources"]
    print(f"✓ ParaPLUIE mean: {parapluie_results['parapluie']:.4f}")
    print(f"✓ ParaPLUIE median: {parapluie_results['parapluie_median']:.4f}")
    print(
        "✓ ParaPLUIE > 0 ratio: "
        f"{parapluie_results['parapluie_positive_ratio']:.4f}"
    )

    gen_diag = compute_generation_diagnostics(predictions)
    copy_ratio = compute_copy_ratio(sources, predictions)

    final_metrics = dict(base_metrics)
    final_metrics["exact_copy_ratio"] = copy_ratio
    final_metrics["parapluie"] = parapluie_results["parapluie"]
    final_metrics["parapluie_median"] = parapluie_results["parapluie_median"]
    final_metrics["parapluie_positive_ratio"] = parapluie_results[
        "parapluie_positive_ratio"
    ]

    report = {
        "timestamp": datetime.now().isoformat(),
        "source_report": report_path,
        "config": {
            **base_report.get("config", {}),
            "parapluie": parapluie_results["config"],
            "parapluie_model": parapluie_results["config"]["model"],
            "parapluie_template": parapluie_results["config"]["template"],
        },
        "metrics": final_metrics,
        "diagnostics": gen_diag,
        "details": [
            {**detail, "src": source, "pred": prediction, "ref": reference}
            for detail, source, prediction, reference in zip(
                base_report["details"], sources, predictions, references
            )
        ],
    }

    write_report(out_report, report)
    print(f"\nReport saved to {out_report}")
