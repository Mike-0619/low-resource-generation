"""Shared word-boundary and output helpers for the BabyLM corpus builders."""

from contextlib import contextmanager
from hashlib import file_digest
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory


def checksum(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def document_prefix(row, words, original_words):
    """Keep the original text through the requested word and update its counts."""
    text = row["text"]
    truncated = words < original_words
    if truncated:
        for index, match in enumerate(re.finditer(r"\S+", text), 1):
            if index == words:
                text = text[:match.end()]
                break
        else:
            raise ValueError("Requested prefix exceeds document word count")
    return {
        **row,
        "text": text,
        "num-tokens": words,
        "bytes": len(text.encode("utf-8")),
        "original-num-tokens": original_words,
        "is-truncated": truncated,
    }


def require_new_output(target):
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"Refusing to overwrite {target}")


@contextmanager
def staged_output(target, prefix):
    require_new_output(target)
    with TemporaryDirectory(prefix=prefix, dir=target.parent) as tmp:
        staged = Path(tmp) / target.name
        yield staged
        require_new_output(target)
        staged.rename(target)


def reuse_dataset(directory, expected_words, filenames, source_matches):
    if not (directory.exists() or directory.is_symlink()):
        return False

    manifest_path = directory / "manifest.json"
    if directory.is_symlink() or manifest_path.is_symlink() or not manifest_path.is_file():
        raise FileExistsError(f"Expected a completed corpus with a manifest: {directory}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("whitespace_tokens") != expected_words
        or not manifest.get("derived_dataset")
        or not source_matches(manifest)
    ):
        raise ValueError(f"Existing corpus has a different word budget or source: {manifest_path}")
    for name in filenames:
        path = directory / name
        if path.is_symlink() or not path.is_file() or not path.stat().st_size:
            raise ValueError(f"Missing, linked or empty corpus file: {path}")
    return True
