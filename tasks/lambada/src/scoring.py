"""Shared punctuation boundaries and case-sensitive normalized scoring."""
import string
import unicodedata
from collections import Counter

SCORING_VERSION = "case_sensitive_punctuation_boundary_sharp_s_v2"
LEADING_PUNCTUATION = '\"“”„‟«»'
TRAILING_PUNCTUATION = LEADING_PUNCTUATION + '.,;:!?…'


def normalize_word(text):
    start, end = 0, len(text)
    while start < end and (text[start].isspace() or text[start] in LEADING_PUNCTUATION):
        start += 1
    while end > start and (text[end - 1].isspace() or text[end - 1] in TRAILING_PUNCTUATION):
        end -= 1
    return text[start:end].replace("ß", "ss").replace("ẞ", "SS")


def score_word(prediction, target_word, completed):
    prediction, target_word = normalize_word(prediction), normalize_word(target_word)
    return {
        "normalized_prediction": prediction,
        "normalized_target_word": target_word,
        "exact_match": bool(completed and prediction and prediction == target_word),
    }


def summarize_greedy(samples):
    total = len(samples)
    correct = sum(item["exact_match"] for item in samples)
    completed = sum(item["completed"] for item in samples)
    return {
        "metric": "lambada_first_word_normalized_exact_match",
        "protocol": SCORING_VERSION,
        "scoring_version": SCORING_VERSION,
        "matching": "case_sensitive_normalized_string_equality",
        "reference_normalization": {
            "leading_punctuation": LEADING_PUNCTUATION,
            "trailing_punctuation": TRAILING_PUNCTUATION,
            "strip_edge_whitespace": True,
            "unicode_normalization": None,
            "character_equivalences": {"ß": "ss", "ẞ": "SS"},
        },
        "correct": correct,
        "total": total,
        "accuracy": correct / total if total else None,
        "completed": completed,
        "failed": total - completed,
        "stop_reasons": dict(Counter(item["stop_reason"] for item in samples)),
    }


def inspect_text(text, final=False, terminal_reason="eos"):
    """Find the first word using the same boundaries for tokens and API chunks."""
    start = 0
    while start < len(text) and (text[start].isspace() or text[start] in LEADING_PUNCTUATION):
        start += 1
    end = start
    reason = None
    while end < len(text):
        char = text[end]
        if char.isspace():
            reason = "whitespace"
            break
        if char in LEADING_PUNCTUATION or char in "!?;…" or text.startswith("...", end):
            reason = "punctuation"
            break
        end += 1
    if reason is None:
        if not final:
            return None
        reason = terminal_reason
    word = text[start:end]
    if any(0xD800 <= ord(char) <= 0xDFFF for char in word):
        return {"prediction": "", "completed": False, "stop_reason": "invalid_utf8"}
    if not word or all(unicodedata.category(char).startswith("P") or char in string.punctuation
                       for char in word):
        return {"prediction": "", "completed": False, "stop_reason": f"empty_{reason}"}
    return {"prediction": word, "completed": True, "stop_reason": reason}
