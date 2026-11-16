"""Build the Turkish 10M and 100M BabyLM training corpora."""

from contextlib import ExitStack, closing
from hashlib import sha256
import json
from pathlib import Path
import random
import shutil

from huggingface_hub import HfFileSystem, hf_hub_download
import pyarrow as pa
import pyarrow.parquet as pq

from .corpus_common import checksum, document_prefix, reuse_dataset, staged_output

BABYLM_REPO = "BabyLM-community/babylm-tur"
BABYLM_REVISION = "30dd54383ae7fac713add39889d2311e8eb350ec"
BABYLM_FILE = "data/train-00000-of-00001.parquet"
BABYLM_SHA256 = "8e80f637175b09a02d4893270d26aed241064d430da16fe51b95877fbb8ff987"
BABYLM_DOCUMENTS = 889
TURMIX_REPO = "AdaMLLab/TurMix"
TURMIX_REVISION = "3fe16607ed37377ce9f6c63635bbea56576c0e5a"
TURMIX_CATEGORY = "padding-turmix"
SEED = 42
TARGETS = {"10M": 10_000_000, "100M": 100_000_000}
SCHEMA = pa.schema([
    ("text", pa.string()), ("doc-id", pa.string()), ("category", pa.string()),
    ("data-source", pa.string()), ("script", pa.string()),
    ("age-estimate", pa.string()), ("license", pa.string()), ("misc", pa.string()),
    ("num-tokens", pa.int64()), ("language", pa.string()), ("bytes", pa.int64()),
    ("original-num-tokens", pa.int64()), ("is-truncated", pa.bool_()),
])
CATEGORIES = ("child-directed-speech", "child-books", "padding-wikipedia", "padding-fineweb-c")


def read_babylm():
    local_file = Path(hf_hub_download(
        BABYLM_REPO, BABYLM_FILE, repo_type="dataset", revision=BABYLM_REVISION,
    ))
    if checksum(local_file) != BABYLM_SHA256:
        raise ValueError("BabyLM parquet does not match the pinned upstream file")
    categories = {category: [] for category in CATEGORIES}
    for row in pq.read_table(local_file).to_pylist():
        words = len(row["text"].split())
        if row["language"] != "tur" or words != row["num-tokens"]:
            raise ValueError("Unexpected BabyLM language or source word count")
        categories[row["category"]].append(document_prefix(row, words, words))
    if sum(map(len, categories.values())) != BABYLM_DOCUMENTS or not all(categories.values()):
        raise ValueError("Unexpected BabyLM document inventory")
    return categories


def turmix_rows():
    fs = HfFileSystem()
    root = f"datasets/{TURMIX_REPO}@{TURMIX_REVISION}/consensus"
    shards = sorted(fs.glob(f"{root}/*.parquet"))
    if not shards:
        raise ValueError("No TurMix matched parquet shards")
    random.Random(SEED).shuffle(shards)
    for shard in shards:
        print(f"Reading TurMix matched shard {Path(shard).name}", flush=True)
        with fs.open(shard, "rb", block_size=4 * 1024**2, cache_type="blockcache",
                     cache_options={"maxblocks": 4}) as stream:
            parquet = pq.ParquetFile(stream)
            for batch in parquet.iter_batches(
                batch_size=256, columns=["text", "id", "source"], use_threads=False,
            ):
                yield from batch.to_pylist()


def padding_path(root, label):
    return root / label / f"{TURMIX_CATEGORY}.train.parquet"


def write_padding(stage, categories, candidates, targets, *, base_words):
    remaining = {label: target - base_words for label, target in targets.items()}
    pending = {label: [] for label in targets}
    documents = dict.fromkeys(targets, 0)
    seen_text = {
        sha256(row["text"].encode("utf-8")).digest()
        for rows in categories.values()
        for row in rows
    }
    if min(remaining.values()) <= 0:
        raise ValueError("BabyLM corpus is at or above a requested target")

    with ExitStack() as stack:
        writers = {
            label: stack.enter_context(
                pq.ParquetWriter(padding_path(stage, label), SCHEMA)
            ) for label in targets
        }

        for candidate in candidates:
            text = candidate["text"]
            words = len(text.split()) if text else 0
            if not words:
                continue
            digest = sha256(text.encode("utf-8")).digest()
            if digest in seen_text:
                continue
            seen_text.add(digest)
            row = {
                "text": text,
                "doc-id": "turmix:" + candidate["id"],
                "category": TURMIX_CATEGORY, "data-source": f"{TURMIX_REPO}:matched",
                "script": "Latn", "language": "tur", "age-estimate": None, "license": None,
                "misc": json.dumps({"source_datasets": candidate["source"]}),
            }
            for label, target in targets.items():
                if not remaining[label]:
                    continue
                take = min(words, remaining[label])
                pending[label].append(document_prefix(row, take, words))
                remaining[label] -= take
                documents[label] += 1
                if len(pending[label]) == 512 or not remaining[label]:
                    writers[label].write_table(pa.Table.from_pylist(pending[label], schema=SCHEMA))
                    pending[label].clear()
                if not remaining[label]:
                    print(f"{label} corpus written with {target:,} words", flush=True)
            if not any(remaining.values()):
                break
        else:
            raise ValueError(f"TurMix stream ended before filling budgets: {remaining}")
    return {
        label: {
            "filename": padding_path(stage, label).name,
            "documents": documents[label],
            "whitespace_tokens": target - base_words - remaining[label],
        } for label, target in targets.items()
    }


def build(target, categories, candidates, targets):
    with staged_output(target, ".turkish-") as stage:
        small, large = targets
        for label in targets:
            (stage / label).mkdir(parents=True)
        base_stats = {}
        for category, rows in categories.items():
            filename = f"{category}.train.parquet"
            table = pa.Table.from_pylist(rows, schema=SCHEMA)
            pq.write_table(table, stage / small / filename)
            shutil.copyfile(stage / small / filename, stage / large / filename)
            base_stats[category] = {
                "filename": filename,
                "documents": len(rows),
                "whitespace_tokens": sum(row["num-tokens"] for row in rows),
            }

        base_words = sum(item["whitespace_tokens"] for item in base_stats.values())
        padding_stats = write_padding(stage, categories, candidates, targets, base_words=base_words)
        for label, words in targets.items():
            stats = {**base_stats, TURMIX_CATEGORY: padding_stats[label]}
            manifest = {
                "derived_dataset": True,
                "size_label": label,
                "repositories": [
                    {"repo": BABYLM_REPO, "revision": BABYLM_REVISION,
                     "file": BABYLM_FILE, "sha256": BABYLM_SHA256},
                    {"repo": TURMIX_REPO, "revision": TURMIX_REVISION,
                     "config": "matched", "split": "train"},
                ],
                "algorithm": {
                    "word_count": "len(text.split())", "seed": SEED,
                    "turmix_order": "seeded shuffle of sorted matched shard names, original row order",
                    "turmix_deduplication": "exact text, including against BabyLM",
                    "truncate_final_document": True,
                },
                "target_whitespace_tokens": words,
                "whitespace_tokens": words,
                "documents": sum(item["documents"] for item in stats.values()),
                "categories": stats,
            }
            (stage / label / "manifest.json").write_text(
                json.dumps(manifest, indent=2) + "\n", encoding="utf-8",
            )
    print(f"Published {target}", flush=True)


def prepare(corpus):
    target = corpus / "Turkish"
    filenames = [f"{category}.train.parquet" for category in (*CATEGORIES, TURMIX_CATEGORY)]
    def source_matches(manifest):
        repositories = {item["repo"]: item["revision"] for item in manifest.get("repositories", [])}
        return (
            repositories == {BABYLM_REPO: BABYLM_REVISION, TURMIX_REPO: TURMIX_REVISION}
            and manifest.get("algorithm", {}).get("seed") == SEED
        )

    if target.exists() or target.is_symlink():
        if target.is_symlink() or not all(
            reuse_dataset(target / label, words, filenames, source_matches)
            for label, words in TARGETS.items()
        ):
            raise FileExistsError(f"Expected both completed Turkish corpora: {target}")
        print(f"Ready: {target}", flush=True)
        return

    corpus.mkdir(parents=True, exist_ok=True)
    categories = read_babylm()
    with closing(turmix_rows()) as candidates:
        build(target, categories, candidates, TARGETS)
