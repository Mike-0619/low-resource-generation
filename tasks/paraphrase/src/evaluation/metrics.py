"""Text normalization, SacreBLEU and generation diagnostics."""

import re
from collections import Counter


BLEU_LOWERCASE = False
BLEU_TOKENIZE = "intl"


def normalize_text(text):
    """Normalize spacing without changing case or lexical content."""
    if not isinstance(text, str):
        raise TypeError(f"Expected text to be str, got {type(text).__name__}")
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([.,!?;:%)])", r"\1", text)
    return re.sub(r"([\(\[])\s+", r"\1", text)


def normalize_pairs(predictions, sources):
    if len(predictions) != len(sources):
        raise ValueError(
            "Mismatched evaluation lengths: "
            f"predictions={len(predictions)}, sources={len(sources)}"
        )
    if not predictions:
        raise ValueError("Cannot evaluate an empty prediction set")
    return (
        [normalize_text(text) for text in predictions],
        [normalize_text(text) for text in sources],
    )


def normalize_triplets(predictions, sources, references):
    if not (len(predictions) == len(sources) == len(references)):
        raise ValueError(
            "Mismatched evaluation lengths: "
            f"predictions={len(predictions)}, sources={len(sources)}, "
            f"references={len(references)}"
        )
    if not predictions:
        raise ValueError("Cannot evaluate an empty prediction set")
    return (
        [normalize_text(text) for text in predictions],
        [normalize_text(text) for text in sources],
        [normalize_text(text) for text in references],
    )


def evaluate_bleu(predictions, sources, references, metric=None):
    """Evaluate reference/source BLEU with one canonical configuration."""
    predictions, sources, references = normalize_triplets(
        predictions, sources, references
    )
    if metric is None:
        import evaluate

        metric = evaluate.load("sacrebleu")

    bleu_ref = metric.compute(
        predictions=predictions,
        references=[[reference] for reference in references],
        lowercase=BLEU_LOWERCASE,
        tokenize=BLEU_TOKENIZE,
    )["score"]
    bleu_src = metric.compute(
        predictions=predictions,
        references=[[source] for source in sources],
        lowercase=BLEU_LOWERCASE,
        tokenize=BLEU_TOKENIZE,
    )["score"]
    return {
        "bleu_ref": float(bleu_ref),
        "bleu_src": float(bleu_src),
        "predictions": predictions,
        "sources": sources,
        "references": references,
        "config": {
            "lowercase": BLEU_LOWERCASE,
            "tokenize": BLEU_TOKENIZE,
            "normalizer": "src.evaluation.metrics.normalize_text",
        },
    }


def compute_generation_diagnostics(predictions, *, include_repeats=True):
    count = len(predictions)
    stats = {
        "unique_ratio": round(len(set(predictions)) / count, 4) if count else 0.0,
        "avg_pred_tokens": round(sum(len(text.split()) for text in predictions) / count, 2) if count else 0.0,
        "empty_ratio": round(sum(not text.strip() for text in predictions) / count, 4) if count else 0.0,
    }
    if include_repeats:
        stats["top_repeated_predictions"] = Counter(predictions).most_common(5)
    return stats


def compute_copy_ratio(sources, predictions):
    if not sources:
        return 0.0
    copied = sum(1 for source, prediction in zip(sources, predictions) if source.strip() == prediction.strip())
    return round(copied / len(sources), 4)
