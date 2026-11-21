"""Pinned English and minimally cleaned German Boldt LAMBADA data."""
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    config: str
    revision: str
    expected_examples: int


DATASET_SPLIT = "test"
LANGUAGES = ("en", "de", "tr")
DATASET_SPECS = {
    "en": DatasetSpec(name="EleutherAI/lambada_openai", config="default",
                      revision="900124bf3b8235c6daf21033af9948b3f07346c4", expected_examples=5153),
    "de": DatasetSpec(name="Boldt/lambada_openai_de", config="default",
                      revision="74d444e7ba3e598d4ae48455f8a4937c5295cce4", expected_examples=5150),
}
BOLDT_ELIGIBLE_EXAMPLES = 5147
BOLDT_PREPARATION_VERSION = "boldt_minimal_cleanup_v1"


def get_dataset_spec(language: str) -> DatasetSpec:
    if language in DATASET_SPECS:
        return DATASET_SPECS[language]
    if language == "tr":
        raise ValueError("Turkish LAMBADA data is pending. Configure the dataset before evaluating with --language tr.")
    raise ValueError(f"Unsupported language: {language}")


def _load_lambada_dataset(language: str):
    from datasets import Dataset, config, load_dataset

    spec = DATASET_SPECS[language]
    cache_root = Path(config.HF_DATASETS_CACHE)
    try:
        return load_dataset(spec.name, spec.config, split=DATASET_SPLIT,
                            revision=spec.revision, cache_dir=str(cache_root))
    except (ConnectionError, OSError):
        cache_namespace = spec.name.replace("/", "___")
        cache_filename = f"{spec.name.rsplit('/', 1)[-1]}-{DATASET_SPLIT}.arrow"
        pattern = f"{cache_namespace}/{spec.config}/*/{spec.revision}/{cache_filename}"
        candidates = sorted(cache_root.glob(pattern))
        if not candidates:
            raise
        return Dataset.from_file(str(candidates[-1]))


def split_example(text, index):
    parts = text.rsplit(" ", 1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError(f"Invalid context/final-word split at dataset index {index}")
    return parts


def prepare_boldt_examples(rows, english_examples):
    """Retain source order and match repeated originals by occurrence."""
    originals = defaultdict(deque)
    for index, text in enumerate(english_examples):
        originals[text].append(index)
    examples, mapping, skipped, trimmed = [], [], [], []
    for index, row in enumerate(rows):
        occurrences = originals.get(row["text_orig"])
        if occurrences is None:
            raise ValueError(f"Unknown English original at Boldt index {index}")
        if not occurrences:
            skipped.append({"dataset_index": index, "reason": "excess_original_occurrence"})
            continue
        english_index = occurrences.popleft()
        text = row["text"].rstrip()
        if text != row["text"]:
            trimmed.append({"dataset_index": index, "reason": "trailing_whitespace"})
        try:
            split_example(text, index)
        except ValueError:
            skipped.append({"dataset_index": index, "english_dataset_index": english_index,
                            "reason": "missing_context_or_target"})
            continue
        examples.append(text)
        mapping.append({"dataset_index": index, "english_dataset_index": english_index})
    return examples, {
        "preparation_version": BOLDT_PREPARATION_VERSION,
        "sample_mapping": mapping,
        "skipped_examples": skipped,
        "trimmed_examples": trimmed,
        "reference_dataset": {field: getattr(DATASET_SPECS["en"], field)
                              for field in ("name", "config", "revision")},
    }


def load_lambada_examples(language: str):
    spec = get_dataset_spec(language)
    rows = _load_lambada_dataset(language)
    dataset_examples = len(rows)
    description = f"{spec.name} ({DATASET_SPLIT})"
    if dataset_examples != spec.expected_examples:
        raise ValueError(
            f"{description} has {dataset_examples} examples; "
            f"expected {spec.expected_examples}"
        )

    metadata = {
        "dataset": description, "dataset_revision": spec.revision,
        "dataset_examples": dataset_examples, "skipped_examples": [],
    }
    if language == "de":
        english_examples, _ = load_lambada_examples("en")
        examples, preparation = prepare_boldt_examples(rows, english_examples)
        if len(examples) != BOLDT_ELIGIBLE_EXAMPLES:
            raise ValueError(f"Boldt preparation has {len(examples)} eligible examples; "
                             f"expected {BOLDT_ELIGIBLE_EXAMPLES}")
        metadata.update(dataset_variant="boldt", **preparation)
    else:
        examples = [row["text"] for row in rows]
    metadata["eligible_examples"] = len(examples)
    return examples, metadata
