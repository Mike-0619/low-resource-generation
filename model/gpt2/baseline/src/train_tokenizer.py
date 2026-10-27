"""Official baseline BPE construction, extracted without ablation defaults."""

from pathlib import Path as _CachePath
import runpy as _cache_runpy
_configure_cache = _cache_runpy.run_path(str(_CachePath(__file__).resolve().parents[6] / "config/cache.py"))["configure"]
_configure_cache("low-resource-generation")

import argparse

from transformers import PreTrainedTokenizerFast
from tokenizers.trainers import BpeTrainer
from tokenizers import Regex, Tokenizer, decoders, normalizers, pre_tokenizers, processors
from tokenizers.models import BPE

from .utils import experiment_paths, load_experiment, require_empty, staged_directory

SPECIAL_TOKEN_IDS = {"<unk>": 0, "<s>": 1, "</s>": 2, "<pad>": 3, "<mask>": 4}

STANDARD_SPLIT_PATTERN = (
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*"
    r"[\p{Ll}\p{Lm}\p{Lo}\p{M}]+|"
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+"
    r"[\p{Ll}\p{Lm}\p{Lo}\p{M}]*|"
    r"\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)


def tokenizer_backend(vocab_size):
    backend = Tokenizer(BPE(unk_token="<unk>"))
    backend.normalizer = normalizers.Sequence(
        [
            normalizers.Prepend(" "),
            normalizers.NFKC(),
            normalizers.Replace(Regex(r"\n"), "\n "),
            normalizers.Replace(Regex(r" *\n"), "\n"),
        ]
    )
    backend.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(Regex(STANDARD_SPLIT_PATTERN), "isolated"),
            pre_tokenizers.ByteLevel(add_prefix_space=False, trim_offsets=True, use_regex=False),
            pre_tokenizers.Split(Regex(r".{1,24}"), "isolated"),
        ]
    )
    backend.post_processor = processors.TemplateProcessing(
        single="<s> $A </s>",
        pair="<s> $A </s> <s> $B </s>",
        special_tokens=[(token, SPECIAL_TOKEN_IDS[token]) for token in ("<s>", "</s>")],
    )
    backend.decoder = decoders.Sequence([
        decoders.ByteLevel(), decoders.Strip(" ", 1, 0), decoders.Replace("\n ", "\n")
    ])
    return backend, BpeTrainer(vocab_size=vocab_size, special_tokens=list(SPECIAL_TOKEN_IDS))


def train_tokenizer(cfg, root):
    paths = experiment_paths(cfg, root)
    files = sorted(paths.cleaned.glob("*.train.txt"))
    if not files:
        raise FileNotFoundError("Run python -m src.clean_dataset --config <config> first")
    target = paths.tokenizer
    require_empty(target)
    backend, trainer = tokenizer_backend(cfg["model"]["vocab_size"])
    backend.train([str(p) for p in files], trainer)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, bos_token="<s>", eos_token="</s>",
        unk_token="<unk>", pad_token="<pad>", model_max_length=1024,
    )
    if tokenizer.vocab_size != cfg["model"]["vocab_size"]:
        raise ValueError("Tokenizer did not reach the configured vocabulary size")
    with staged_directory(target) as staging:
        tokenizer.save_pretrained(staging)
    print(f"Trained independent tokenizer: {target}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Train an independent BPE tokenizer for one baseline corpus.')
    parser.add_argument("--config", required=True, help="Experiment config, relative to model/gpt2/baseline")
    args = parser.parse_args(argv)
    cfg, root = load_experiment(args.config)
    train_tokenizer(cfg, root)


if __name__ == "__main__":
    main()
