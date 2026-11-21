"""Disk-backed full-source tokenization, chunking, padding and loss masks."""
import hashlib
import json
import os
import re
from collections.abc import Iterator
from fcntl import LOCK_EX, flock
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from ..corpora.corpus_common import checksum
from .train_tokenizer import SPECIAL_TOKEN_IDS, tokenizer_backend
from ..training.utils import staged_directory, write_json_atomic


CHUNK_BYTES = 2 * 1024 * 1024
# The next byte must be NFKC-stable, non-whitespace, and unable to start
# an added token. Exclude '/' because the original regex can attach it
# to punctuation across a newline. A CRLF pair stays together.
BOUNDARY = re.compile(rb"[\r\n](?=[!-.0-;=-~])")
CACHE_VERSION = 1


def token_cache_directory(root, language, size):
    project = root.resolve().parents[2]
    workspace = project.parent.parent
    workspace_cache = workspace / ".cache"
    on_lrz = (workspace / ".agents/config/cache/env.sh").is_file()
    default = workspace_cache if on_lrz else Path.home() / ".cache"
    base = Path(os.environ.get("XDG_CACHE_HOME") or default).expanduser()
    directory = (base / "projects/low-resource-generation/gpt2-baseline/tokens" / language / size).resolve()
    if on_lrz and (workspace_cache.resolve() != workspace_cache or not directory.is_relative_to(workspace_cache)):
        raise ValueError(f"Baseline token cache must be inside {workspace_cache}: {directory}")
    return directory


def _tokenizer_digest(tokenizer):
    if not tokenizer.is_fast:
        raise ValueError("Chunked baseline encoding requires a fast tokenizer")
    serialized = tokenizer.backend_tokenizer.to_str()
    current = json.loads(serialized)
    recipe = json.loads(tokenizer_backend().to_str())
    if (
        current["normalizer"] != recipe["normalizer"]
        or current["pre_tokenizer"] != recipe["pre_tokenizer"]
        or current["model"]["type"] != "BPE"
        or current["model"]["dropout"] is not None
        or any(
            token["content"] not in SPECIAL_TOKEN_IDS
            or any(token[field] for field in ("single_word", "lstrip", "rstrip", "normalized"))
            for token in current["added_tokens"]
        )
    ):
        raise ValueError("Chunked baseline encoding requires a tokenizer from src.data.train_tokenizer")
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _text_chunks(path: Path) -> Iterator[tuple[str, bool]]:
    def decode(chunk):
        # Match Path.read_text's universal newline handling.
        return chunk.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")

    with path.open("rb") as source:
        pending = b""
        while block := pending + source.read(CHUNK_BYTES - len(pending)):
            if len(block) < CHUNK_BYTES or not source.peek(1):
                yield decode(block), True
                return
            boundary = BOUNDARY.search(block, CHUNK_BYTES // 2)
            if boundary is None:
                for match in BOUNDARY.finditer(block, 0, CHUNK_BYTES // 2 + 1):
                    boundary = match
            if boundary is None:
                raise ValueError(f"No safe tokenizer split within {CHUNK_BYTES} bytes of {path}")
            cut = boundary.end()
            yield decode(block[:cut]), False
            pending = block[cut:]


def _encode(tokenizer, text):
    return tokenizer(text=[text], add_special_tokens=False, verbose=False)["input_ids"][0]


def _write_cache(target, tokenizer, files, identity):
    # The normalizer inserts a space after a newline and also prepends one
    # at the start of an input. A sentinel preserves the left block's newline
    # context and its encoded suffix can then be removed.
    sentinel = _encode(tokenizer, "A")
    offsets = [0]
    sources = []
    payload = identity["sequence_length"] - 2
    with staged_directory(target) as staging:
        with (staging / "tokens.i32").open("wb") as output:
            for path in files:
                start = end = offsets[-1]
                previous_samples = len(offsets) - 1
                for text, last in _text_chunks(path):
                    ids = _encode(tokenizer, text if last else text + "A")
                    if not last:
                        if not sentinel or ids[-len(sentinel):] != sentinel:
                            raise ValueError(f"Tokenizer split changed the sentinel encoding in {path}")
                        del ids[-len(sentinel):]
                    np.asarray(ids, dtype="<i4").tofile(output)
                    end += len(ids)
                if end > start:
                    offsets.extend(range(start + payload, end, payload))
                    offsets.append(end)
                sources.append({"file": path.name, "tokens": end - start,
                                "samples": len(offsets) - 1 - previous_samples})
        if len(offsets) == 1:
            raise ValueError("Empty training corpus")
        np.asarray(offsets, dtype="<i8").tofile(staging / "offsets.i64")
        write_json_atomic(staging / "metadata.json", {
            "identity": identity, "tokens": offsets[-1], "samples": len(offsets) - 1,
            "sources": sources,
        })


class CachedBabyLMDataset(Dataset):
    """Read source-local token windows from an immutable cache."""

    def __init__(self, directory, tokenizer, sequence_length, cache_directory):
        files = sorted(Path(directory).glob("*.train.txt"))
        if not files:
            raise FileNotFoundError(f"No cleaned training sources: {directory}")
        if sequence_length < 3:
            raise ValueError("sequence_length must leave room for payload, BOS and EOS")
        manifest = Path(directory) / "preparation.json"
        identity = {
            "version": CACHE_VERSION,
            "sequence_length": sequence_length,
            "tokenizer_sha256": _tokenizer_digest(tokenizer),
            "tokenizers_version": version("tokenizers"),
            "transformers_version": version("transformers"),
            "preparation_sha256": checksum(manifest) if manifest.is_file() else None,
            "sources": [],
        }
        for path in files:
            status = path.stat()
            identity["sources"].append({
                "path": str(path.resolve()), "bytes": status.st_size, "mtime_ns": status.st_mtime_ns,
            })
        self.cache_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        cache_directory.mkdir(parents=True, exist_ok=True)
        target = cache_directory / self.cache_id
        # One writer per condition, with only complete caches visible to readers.
        if target.is_symlink():
            raise ValueError(f"Token cache must not be a symbolic link: {target}")
        if not target.exists():
            lock_path = cache_directory / ".build.lock"
            if lock_path.is_symlink():
                raise ValueError(f"Token cache lock must not be a symbolic link: {lock_path}")
            with lock_path.open("a") as lock:
                flock(lock, LOCK_EX)
                if not target.exists():
                    print(f"Building baseline token cache: {target}", flush=True)
                    _write_cache(target, tokenizer, files, identity)

        metadata_path = target / "metadata.json"
        tokens, offsets = target / "tokens.i32", target / "offsets.i64"
        if any(part.is_symlink() for part in (metadata_path, tokens, offsets)):
            raise ValueError(f"Token cache must not contain symbolic links: {target}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (
            metadata["identity"] != identity
            or metadata["samples"] <= 0
            or tokens.stat().st_size != 4 * metadata["tokens"]
            or offsets.stat().st_size != 8 * (metadata["samples"] + 1)
        ):
            raise ValueError(f"Incompatible or incomplete token cache: {target}")
        self.tokens = np.memmap(tokens, mode="r", dtype="<i4")
        self.offsets = np.memmap(offsets, mode="r", dtype="<i8")
        self.bos, self.eos = tokenizer.bos_token_id, tokenizer.eos_token_id

    def __len__(self):
        return len(self.offsets) - 1

    def __getitem__(self, index):
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        start, stop = self.offsets[index:index + 2]
        item = np.empty(stop - start + 2, dtype=np.int64)
        item[0], item[-1] = self.bos, self.eos
        item[1:-1] = self.tokens[start:stop]
        return torch.from_numpy(item)


def collate_batch(batch, eos):
    tokens = pad_sequence(batch, padding_value=eos, batch_first=True)
    inputs, targets = tokens[:, :-1], tokens[:, 1:]
    mask = inputs != eos
    mask[:, 0] = True
    return inputs, targets, mask
