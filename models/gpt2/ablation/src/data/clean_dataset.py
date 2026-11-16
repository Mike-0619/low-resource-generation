"""Prepare deterministic cleaned corpora for the two training routes."""

import argparse
import os
import re
from contextlib import ExitStack
from dataclasses import dataclass
from itertools import batched
from pathlib import Path
from typing import Any

from ..training.utils import (
    CONDITION_GROUPS, CONFIG_PATHS, ConditionSpec, condition_paths,
    configs_for_condition, staged_directory, write_json_atomic,
)

@dataclass(frozen=True)
class SourceSpec:
    name: str
    filename: str
    cleaning: str = "plain"
    short: bool = False


SOURCE_SPECS = {
    "English": (
        SourceSpec("bnc_spoken", "bnc_spoken.train.txt", short=True),
        SourceSpec("childes", "childes.train.txt", short=True),
        SourceSpec("gutenberg", "gutenberg.train.txt", "html"),
        SourceSpec("open_subtitles", "open_subtitles.train.txt", "subtitle", True),
        SourceSpec("simple_wiki", "simple_wiki.train.txt", "html"),
        SourceSpec("switchboard", "switchboard.train.txt", short=True),
    ),
    "German": (
        SourceSpec("child-available-speech", "child_available_speech.train.txt", short=True),
        SourceSpec("child-books", "child_books.train.txt", "html"),
        SourceSpec("child-directed-speech", "child_directed_speech.train.txt", short=True),
        SourceSpec("child-news", "child_news.train.txt"),
        SourceSpec("child-wiki", "child_wiki.train.txt", "html"),
        SourceSpec("educational", "educational.train.txt", "html"),
        SourceSpec("padding-opensubtitles", "padding_open_subtitles.train.txt", "subtitle", True),
    ),
}
GERMAN_PARQUET_FILES = ("train-00000-of-00002.parquet", "train-00001-of-00002.parquet")
GERMAN_EXPECTED_DOCUMENTS = 36_550
GERMAN_EXPECTED_WHITESPACE_TOKENS = 107_910_839
SHORT_CHUNK_WORDS = 128


def sources_for(condition: ConditionSpec) -> tuple[SourceSpec, ...]:
    return SOURCE_SPECS[condition.language]


def _clean_source(source: SourceSpec, text: str, language: str) -> str:
    if source.cleaning == "html":
        text = re.sub(r"<[^>]+>", "", text)
    elif source.cleaning == "subtitle":
        pattern = r"subtitle|\u5b57\u5e55|ondertitel" + ("|untertitel" if language == "German" else "")
        text = re.sub(rf"^.*(?:{pattern}).*$\n?", "", text, flags=re.MULTILINE | re.IGNORECASE)
    text = re.sub(r"[ \t\u00a0]+", " ", text)
    return re.sub(r"[ \t\u00a0]([.,;!?\u3002\uff0c\uff1b\uff01\uff1f])", r"\1", text)


def _pack_text(text: str) -> tuple[str, tuple[int, int, int]]:
    chunks: list[list[str]] = []
    current: list[str] = []
    for words in filter(None, map(str.split, text.splitlines())):
        if len(words) > SHORT_CHUNK_WORDS:
            if current:
                chunks.append(current)
                current = []
            chunks.extend(map(list, batched(words, SHORT_CHUNK_WORDS)))
        elif current and len(current) + len(words) > SHORT_CHUNK_WORDS:
            chunks.append(current)
            current = words
        else:
            current.extend(words)
    if current:
        chunks.append(current)
    sizes = [len(chunk) for chunk in chunks]
    return "\n".join(map(" ".join, chunks)), (len(chunks), sum(sizes), max(sizes, default=0))


def _clean_and_pack(
    source: SourceSpec, text: str, language: str, stats: dict[str, Any]
) -> str:
    cleaned = _clean_source(source, text, language)
    if not source.short:
        return cleaned
    cleaned, (count, words, maximum) = _pack_text(cleaned)
    packing = stats.setdefault(
        "packing", {"logical_chunks": 0, "max_chunk_words": 0, "_words": 0}
    )
    packing["logical_chunks"] += count
    packing["max_chunk_words"] = max(packing["max_chunk_words"], maximum)
    packing["_words"] += words
    return cleaned


def _metadata(condition: ConditionSpec, sources: dict[str, Any]) -> dict[str, Any]:
    for stats in sources.values():
        if packing := stats.get("packing"):
            packing["mean_chunk_words"] = packing.pop("_words") / packing["logical_chunks"]
    language = {"English": "en", "German": "de"}[condition.language]
    return {
        "dataset": {
            "corpus_path": condition.corpus_path,
            "repo_id": condition.dataset_repo_id,
            "revision": condition.dataset_revision,
        },
        "cleaning_protocol": f"{language}_short",
        "source_order": list(sources),
        "sources": sources,
        "packing_protocol": "record_aware_128", "chunk_word_limit": 128,
    }


def _prepare_english(condition: ConditionSpec, raw: Path, staging: Path) -> dict[str, Any]:
    sources: dict[str, Any] = {}
    for source in sources_for(condition):
        text = (raw / source.filename).read_text(encoding="utf-8")
        stats: dict[str, Any] = {"filename": source.filename}
        cleaned = _clean_and_pack(source, text, condition.language, stats)
        (staging / source.filename).write_text(cleaned, encoding="utf-8")
        sources[source.name] = stats
    return _metadata(condition, sources)


def _prepare_german(condition: ConditionSpec, raw: Path, staging: Path) -> dict[str, Any]:
    import pyarrow.parquet as pq

    specs = {source.name: source for source in sources_for(condition)}
    sources = {
        source.name: {
            "filename": source.filename,
            "documents": 0,
            "official_tokens": 0,
        }
        for source in specs.values()
    }
    with ExitStack() as stack:
        handles = {
            name: stack.enter_context((staging / source.filename).open("w", encoding="utf-8"))
            for name, source in specs.items()
        }
        for filename in GERMAN_PARQUET_FILES:
            parquet = pq.ParquetFile(raw / filename)
            for batch in parquet.iter_batches(
                columns=["category", "language", "num-tokens", "text"], batch_size=256
            ):
                for row in batch.to_pylist():
                    category = row["category"]
                    text = row["text"]
                    if row["language"] != "deu":
                        raise ValueError(f"Unexpected German language: {row['language']}")
                    state = sources[category]
                    output_text = _clean_and_pack(specs[category], text, condition.language, state)
                    if state["documents"]:
                        handles[category].write("\n\n")
                    handles[category].write(output_text)
                    state["documents"] += 1
                    state["official_tokens"] += int(row["num-tokens"])
    actual = tuple(
        sum(source[key] for source in sources.values())
        for key in ("documents", "official_tokens")
    )
    expected = (GERMAN_EXPECTED_DOCUMENTS, GERMAN_EXPECTED_WHITESPACE_TOKENS)
    if actual != expected:
        raise ValueError(f"Expected German documents/tokens {expected}, got {actual}")

    return _metadata(condition, sources)


def prepare_condition(condition: ConditionSpec, *, overwrite: bool = False) -> Path:
    target = condition_paths(condition).cleaned
    if target.exists() and not overwrite:
        return target
    raw = Path(os.environ["CORPUS_ROOT"]).expanduser() / condition.corpus_path
    with staged_directory(target) as staging:
        stats = {"English": _prepare_english, "German": _prepare_german}[condition.language](condition, raw, staging)
        write_json_atomic(staging / "cleaning_stats.json", stats)
    return target


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=(*CONDITION_GROUPS, *CONFIG_PATHS), default="shared_spans")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    for config in configs_for_condition(args.condition):
        target = prepare_condition(config.condition, overwrite=args.overwrite)
        print(f"Ready: {config.condition.name} -> {target}", flush=True)


if __name__ == "__main__":
    main()
