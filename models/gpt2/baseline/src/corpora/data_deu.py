"""Take the original-order prefix of 10% of each German category's word count.

Read the pinned local German/108M shards in filename/row order, without shuffle
or a sampling seed. Count words with len(text.split()). Allocate floor(n / 10)
per category, then largest remainders (category-name tie break) to round the
overall 10% half up. Only the last selected document per category may be cut,
at an original whitespace-word boundary; all earlier documents stay unchanged.

German/10M contains 10,791,084 words.
"""
import json

import pyarrow as pa
import pyarrow.parquet as pq

from .corpus_common import checksum, document_prefix, require_new_output, reuse_dataset, staged_output

SOURCE_REVISION = "69ff474c64a7e0df3a473b1b5d968a0fc18478ac"
SOURCE_WORDS = 107_910_839
CATEGORIES = (
    "child-available-speech", "child-books", "child-directed-speech", "child-news",
    "child-wiki", "educational", "padding-opensubtitles",
)
SHARDS = ("train-00000-of-00002.parquet", "train-00001-of-00002.parquet")
EXTRA_FIELDS = (pa.field("bytes", pa.int64()), pa.field("original-num-tokens", pa.int64()),
                pa.field("is-truncated", pa.bool_()))


def allocate_quotas(totals):
    target = (sum(totals.values()) + 5) // 10
    quotas = {category: words // 10 for category, words in totals.items()}
    order = sorted(totals, key=lambda category: (-(totals[category] % 10), category))
    for category in order[:target - sum(quotas.values())]:
        quotas[category] += 1
    return quotas


def select_prefixes(rows, quotas):
    remaining = dict(quotas)
    selected = {category: [] for category in quotas}
    for row in rows:
        category = row["category"]
        if category not in selected or row["language"] != "deu":
            raise ValueError("Unexpected category or language")
        if remaining[category] == 0:
            continue
        words = len(row["text"].split())
        if words <= 0 or words != row["num-tokens"]:
            raise ValueError("Unexpected source word count")
        take = min(words, remaining[category])
        selected[category].append(document_prefix(row, take, words))
        remaining[category] -= take
    if any(remaining.values()):
        raise ValueError("Source cannot fill category word quota")
    return selected


def source_rows(paths):
    for path in sorted(paths):
        for batch in pq.ParquetFile(path).iter_batches(batch_size=64):
            yield from batch.to_pylist()


def inputs(corpus):
    manifest_path = corpus / "_support/download_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    repo = next(entry for entry in manifest["repositories"] if entry["repo"] == "BabyLM-community/babylm-deu")
    if repo["revision"] != SOURCE_REVISION:
        raise ValueError("Unexpected upstream revision")
    paths = [corpus / "German/108M" / name for name in SHARDS]
    inventory = {entry["path"]: entry for entry in manifest["files"]}
    source_files = []
    for path in paths:
        if path.resolve() != path or not path.is_file():
            raise ValueError("Missing or linked source shard")
        entry = inventory[str(path.relative_to(corpus))]
        source_files.append({key: entry[key] for key in ("path", "size_bytes", "sha256")})
    schema = pq.ParquetFile(paths[0]).schema_arrow
    if not all(pq.ParquetFile(path).schema_arrow.equals(schema, check_metadata=True) for path in paths[1:]):
        raise ValueError("Source shard schemas differ")
    if any(field.name in schema.names for field in EXTRA_FIELDS):
        raise ValueError("Source has conflicting derived fields")
    source = {"repo": repo["repo"], "revision": SOURCE_REVISION, "files": source_files,
              "download_manifest_sha256": checksum(manifest_path)}
    return paths, pa.schema(list(schema) + list(EXTRA_FIELDS), metadata=schema.metadata), source


def category_totals(paths):
    totals = dict.fromkeys(CATEGORIES, 0)
    for row in source_rows(paths):
        if row["category"] not in totals or row["language"] != "deu":
            raise ValueError("Unexpected source category or language")
        words = len(row["text"].split())
        if words <= 0 or words != row["num-tokens"]:
            raise ValueError("Source word count differs from text")
        totals[row["category"]] += words
    if sum(totals.values()) != SOURCE_WORDS:
        raise ValueError(f"Source does not contain the expected {SOURCE_WORDS:,} words")
    return totals


def expected_subset(corpus):
    paths, schema, source = inputs(corpus)
    totals = category_totals(paths)
    quotas = allocate_quotas(totals)
    selected = select_prefixes(source_rows(paths), quotas)
    return selected, schema, source, totals, quotas


def describe(output, selected, source, totals, quotas):
    stats, seen = {}, set()
    for category, rows in selected.items():
        path = output / f"{category}.train.parquet"
        words = sum(row["num-tokens"] for row in rows)
        if not rows or words != quotas[category]:
            raise ValueError("Category word quota mismatch")
        document_ids = {row["doc-id"] for row in rows}
        if len(document_ids) != len(rows) or seen & document_ids:
            raise ValueError("Duplicate document ID")
        seen.update(document_ids)
        truncated = [
            {"doc-id": row["doc-id"], "original_words": row["original-num-tokens"],
             "selected_words": row["num-tokens"]}
            for row in rows if row["is-truncated"]
        ]
        stats[category] = {"filename": path.name, "documents": len(rows), "whitespace_tokens": words,
                           "text_bytes": sum(row["bytes"] for row in rows), "truncated_documents": truncated,
                           "size_bytes": path.stat().st_size, "sha256": checksum(path)}
    return {"format_version": 3, "derived_dataset": True,
            "size_label": "10% of original German words; legacy directory name 10M", "source": source,
            "algorithm": {"name": "category_word_ten_percent_prefix_v1", "fraction": "1/10",
                "word_count": "len(text.split())", "quota_rounding": "floor each share; largest remainders to half-up rounded total; category-name ties",
                "document_order": "ascending source shard filename, then original row order",
                "shuffle": False, "truncate_final_document": True},
            "original_category_words": totals, "category_quotas": quotas,
            "target_whitespace_tokens": sum(quotas.values()),
            "documents": sum(item["documents"] for item in stats.values()),
            "text_bytes": sum(item["text_bytes"] for item in stats.values()),
            "whitespace_tokens": sum(item["whitespace_tokens"] for item in stats.values()), "categories": stats}


def build(corpus, output):
    require_new_output(output)
    selected, schema, source, totals, quotas = expected_subset(corpus)
    output.mkdir()
    for category, rows in selected.items():
        pq.write_table(pa.Table.from_pylist(rows, schema=schema), output / f"{category}.train.parquet",
                       compression="zstd", compression_level=3, row_group_size=1024)
    manifest = describe(output, selected, source, totals, quotas)
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    return manifest


def prepare(corpus):
    target = corpus / "German/10M"
    filenames = (f"{category}.train.parquet" for category in CATEGORIES)
    def source_matches(manifest):
        return (
            manifest.get("source", {}).get("revision") == SOURCE_REVISION
            and manifest.get("algorithm", {}).get("name") == "category_word_ten_percent_prefix_v1"
        )

    if not reuse_dataset(target, (SOURCE_WORDS + 5) // 10, filenames, source_matches):
        with staged_output(target, ".ten-percent-subset-") as staged:
            build(corpus, staged)
    print(f"Ready: {target}", flush=True)
