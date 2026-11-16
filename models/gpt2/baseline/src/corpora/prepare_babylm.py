"""Prepare the pinned BabyLM corpora used by the GPT-2 experiments."""

import argparse
import json
import os
from pathlib import Path
from shutil import copyfile
from tempfile import TemporaryDirectory

from huggingface_hub import hf_hub_download

from . import data_deu, data_tur
from .corpus_common import checksum

INVENTORY = Path(__file__).with_name("download_manifest.json")
OFFICIAL_DATASETS = {
    "english-10m": "English/10M",
    "english-100m": "English/100M",
    "german-108m": "German/108M",
}
DATASETS = (*OFFICIAL_DATASETS, "german-10m", "turkish")


def validate_download(path, entry):
    if path.is_symlink() or not path.is_file():
        raise FileExistsError(f"Expected a regular corpus file: {path}")
    if path.stat().st_size != entry["size_bytes"] or checksum(path) != entry["sha256"]:
        raise ValueError(f"File does not match the pinned corpus inventory: {path}")


def download_file(corpus, repo, entry):
    target = corpus / entry["path"]
    if target.exists() or target.is_symlink():
        validate_download(target, entry)
        return

    filename = entry["source_url"].split(f"/resolve/{repo['revision']}/", 1)[1]
    cached = hf_hub_download(
        repo["repo"], filename, repo_type="dataset", revision=repo["revision"],
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".download-", dir=target.parent) as temporary:
        staged = Path(temporary) / target.name
        copyfile(cached, staged)
        validate_download(staged, entry)
        try:
            os.link(staged, target)
        except FileExistsError:
            validate_download(target, entry)
    print(f"Downloaded {target}", flush=True)


def prepare_official(corpus, destinations):
    manifest = json.loads(INVENTORY.read_text(encoding="utf-8"))
    target = corpus / "_support/download_manifest.json"
    expected_repos = {repo["destination"]: repo for repo in manifest["repositories"]}
    expected_files = {entry["path"]: entry for entry in manifest["files"]}
    if target.exists() or target.is_symlink():
        if target.is_symlink() or not target.is_file():
            raise FileExistsError(f"Expected a regular download manifest: {target}")
        existing = json.loads(target.read_text(encoding="utf-8"))
        if (
            {repo["destination"]: repo for repo in existing["repositories"]} != expected_repos
            or {entry["path"]: entry for entry in existing["files"]} != expected_files
        ):
            raise ValueError(f"Existing manifest differs from the pinned corpus inventory: {target}")

    for destination in destinations:
        repo = expected_repos[destination]
        for path, entry in expected_files.items():
            if Path(path).parent == Path(destination):
                download_file(corpus, repo, entry)
        print(f"Ready: {corpus / destination}", flush=True)

    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix=".manifest-", dir=target.parent) as temporary:
            staged = Path(temporary) / target.name
            copyfile(INVENTORY, staged)
            try:
                os.link(staged, target)
            except FileExistsError:
                if target.is_symlink() or target.read_bytes() != staged.read_bytes():
                    raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=(*DATASETS, "all"))
    args = parser.parse_args(argv)
    root = os.environ.get("CORPUS_ROOT")
    if not root:
        parser.error("Set CORPUS_ROOT to the directory where corpora should be stored.")
    corpus = Path(root).expanduser().resolve() / "babylm-2026"
    selected = DATASETS if args.dataset == "all" else (args.dataset,)
    official = {"german-10m": "German/108M", **OFFICIAL_DATASETS}
    destinations = dict.fromkeys(official[name] for name in selected if name in official)
    if destinations:
        prepare_official(corpus, destinations)
    if "german-10m" in selected:
        data_deu.prepare(corpus)
    if "turkish" in selected:
        data_tur.prepare(corpus)


if __name__ == "__main__":
    main()
