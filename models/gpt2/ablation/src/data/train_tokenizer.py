"""Build and validate nested, byte-complete BPE tokenizer families."""

import argparse
import json
from pathlib import Path

from tokenizers import AddedToken, Regex, Tokenizer
from tokenizers import decoders, normalizers, pre_tokenizers, processors
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from .clean_dataset import sources_for
from ..training.utils import (
    CONDITION_GROUPS,
    SPECIAL_TOKEN_IDS,
    VOCAB_SIZES,
    ConditionSpec,
    staged_directory,
    condition_paths,
    configs_for_condition,
)


BPE_SPLIT_PATTERN = (
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*"
    r"[\p{Ll}\p{Lm}\p{Lo}\p{M}]+|"
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+"
    r"[\p{Ll}\p{Lm}\p{Lo}\p{M}]*|"
    r"\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)


def _new_backend() -> Tokenizer:
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
            pre_tokenizers.Split(Regex(BPE_SPLIT_PATTERN), "isolated"),
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
    return backend


def truncate_backend(serialized_backend: str, vocab_size: int) -> Tokenizer:
    payload = json.loads(serialized_backend)
    model = payload["model"]
    vocab = {token: index for token, index in model["vocab"].items() if index < vocab_size}
    model["vocab"] = vocab
    model["merges"] = [
        merge for merge in model["merges"]
        if all(token in vocab for token in (*merge, "".join(merge)))
    ]
    return Tokenizer.from_str(json.dumps(payload, ensure_ascii=False))


def as_fast_tokenizer(backend: Tokenizer) -> PreTrainedTokenizerFast:
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
        mask_token="<mask>",
        model_max_length=1024,
        clean_up_tokenization_spaces=False,
    )


def load_tokenizer(path: str | Path) -> PreTrainedTokenizerFast:
    return AutoTokenizer.from_pretrained(path, local_files_only=True, use_fast=True)


def build_tokenizer_family(condition: ConditionSpec, *, overwrite: bool = False) -> Path:
    paths = condition_paths(condition)
    target = paths.tokenizer_root
    if target.exists() and not overwrite:
        return target

    with staged_directory(target) as staging:
        backend = _new_backend()
        backend.train(
            [str(paths.cleaned / source.filename) for source in sources_for(condition)],
            trainer=BpeTrainer(
                vocab_size=VOCAB_SIZES[-1],
                min_frequency=2,
                initial_alphabet=ByteLevel.alphabet(),
                show_progress=True,
                special_tokens=[
                    AddedToken(token, special=True, normalized=False)
                    for token in SPECIAL_TOKEN_IDS
                ],
            ),
        )
        serialized_backend = backend.to_str()
        for vocab_size in VOCAB_SIZES:
            as_fast_tokenizer(
                backend if vocab_size == VOCAB_SIZES[-1]
                else truncate_backend(serialized_backend, vocab_size)
            ).save_pretrained(staging / f"vocab_{vocab_size}")
    return target


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--condition",
        choices=("shared_spans", *CONDITION_GROUPS["shared_spans"]),
        default="shared_spans",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    for config in configs_for_condition(args.condition):
        root = build_tokenizer_family(config.condition, overwrite=args.overwrite)
        print(f"Ready: {config.condition.name} -> {root}", flush=True)


if __name__ == "__main__":
    main()
