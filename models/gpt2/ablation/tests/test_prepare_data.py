import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq

from src.data.clean_dataset import (
    GERMAN_PARQUET_FILES,
    SHORT_CHUNK_WORDS,
    _prepare_english,
    _prepare_german,
    sources_for,
)
from src.training.utils import configs_for_condition


class PrepareDataTest(unittest.TestCase):
    def test_english_cleaning_and_record_packing(self) -> None:
        spans = configs_for_condition("10m_en_spans")[0]
        short = configs_for_condition("10m_en_balanced")[0]
        records = [
            " ".join(f"a{i}" for i in range(70)),
            " ".join(f"b{i}" for i in range(70)),
            " ".join(f"c{i}" for i in range(130)),
            " ".join(f"d{i}" for i in range(20)),
        ]
        text = "\n".join(records)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw, spans_out, short_out = (root / name for name in ("raw", "spans", "short"))
            for directory in (raw, spans_out, short_out):
                directory.mkdir()
            sources = sources_for(spans.condition)
            for source in sources:
                source_text = (
                    f"<p>{text}</p>" if source.name == "gutenberg"
                    else f"Subtitle by Alice\n\u5b57\u5e55 by Bob\n{text}"
                    if source.name == "open_subtitles"
                    else text
                )
                (raw / source.filename).write_text(source_text, encoding="utf-8")

            _prepare_english(spans.condition, raw, spans_out)
            stats = _prepare_english(short.condition, raw, short_out)
            self.assertEqual(stats["packing_protocol"], "record_aware_128")
            self.assertEqual(stats["dataset"], {
                "corpus_path": short.condition.corpus_path,
                "repo_id": short.condition.dataset_repo_id,
                "revision": short.condition.dataset_revision,
            })
            for source in sources:
                original = (spans_out / source.filename).read_text()
                packed = (short_out / source.filename).read_text()
                self.assertEqual(original.split(), packed.split())
                if source.short:
                    self.assertEqual(
                        [len(line.split()) for line in packed.splitlines()],
                        [70, 70, SHORT_CHUNK_WORDS, 2, 20],
                    )
                else:
                    self.assertEqual(original, packed)
            self.assertNotIn("<p>", (spans_out / "gutenberg.train.txt").read_text())
            subtitles = (spans_out / "open_subtitles.train.txt").read_text()
            self.assertNotIn("Subtitle by Alice", subtitles)
            self.assertNotIn("\u5b57\u5e55 by Bob", subtitles)

    def test_german_cleaning_stats_and_document_boundaries(self) -> None:
        config = configs_for_condition("108m_de_balanced")[0]
        sources = sources_for(config.condition)
        rows = [{
            "text": (
                "Untertitel von Alice\n" + " ".join(f"first_{i}" for i in range(20))
                if source.name == "padding-opensubtitles"
                else f"<p>{' '.join(f'{source.name}_{i}' for i in range(20))}</p>"
                if source.name == "child-books"
                else " ".join(f"{source.name}_{i}" for i in range(20))
            ),
            "category": source.name,
            "language": "deu",
            "num-tokens": 20,
        } for source in sources]
        subtitle = next(row for row in rows if row["category"] == "padding-opensubtitles")
        subtitle["text"] = "\n".join([
            "Untertitel von Alice",
            " ".join(f"first_a_{i}" for i in range(70)),
            " ".join(f"first_b_{i}" for i in range(70)),
        ])
        subtitle["num-tokens"] = 140
        rows.append({
            "text": " ".join(f"second_{i}" for i in range(140)),
            "category": "padding-opensubtitles",
            "language": "deu",
            "num-tokens": 140,
        })

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw, output = root / "raw", root / "output"
            raw.mkdir()
            output.mkdir()
            midpoint = len(rows) // 2
            for filename, subset in zip(GERMAN_PARQUET_FILES, (rows[:midpoint], rows[midpoint:])):
                pq.write_table(pa.Table.from_pylist(subset), raw / filename)
            with (
                patch("src.data.clean_dataset.GERMAN_EXPECTED_DOCUMENTS", len(rows)),
                patch(
                    "src.data.clean_dataset.GERMAN_EXPECTED_WHITESPACE_TOKENS",
                    sum(row["num-tokens"] for row in rows),
                ),
            ):
                stats = _prepare_german(config.condition, raw, output)

            self.assertEqual(stats["dataset"], {
                "corpus_path": config.condition.corpus_path,
                "repo_id": config.condition.dataset_repo_id,
                "revision": config.condition.dataset_revision,
            })
            documents = (output / "padding_open_subtitles.train.txt").read_text().split("\n\n")
            self.assertEqual(
                [[len(line.split()) for line in document.splitlines()] for document in documents],
                [[70, 70], [SHORT_CHUNK_WORDS, 12]],
            )
            self.assertNotIn("second_", documents[0])
            self.assertEqual(
                sum(source["documents"] for source in stats["sources"].values()), len(rows)
            )
            self.assertEqual(
                sum(source["official_tokens"] for source in stats["sources"].values()),
                sum(row["num-tokens"] for row in rows),
            )
            self.assertEqual(
                stats["sources"]["padding-opensubtitles"]["packing"]["logical_chunks"],
                4,
            )
            self.assertNotIn("<p>", (output / "child_books.train.txt").read_text())
