"""Command-line entry point for the Paraphrase experiment stages."""

from .config import parse_args


def main(argv=None):
    args = parse_args(argv)
    mode = args.mode
    if mode == "train":
        from .gpt2 import run_paraphrase_finetune as run
    elif mode == "llama_benchmark":
        from ..evaluation.llama3 import run_llama_benchmark as run
    elif mode == "parapluie":
        from ..evaluation.parapluie import run_parapluie_evaluate as run
    elif mode == "database":
        from ..evaluation.aggregate import run_build_database as run
    elif mode == "judge":
        from ..evaluation.judge import run_llm_judge as run
    else:
        raise SystemExit(f"Unsupported Paraphrase mode: {mode}")
    run(args)


if __name__ == "__main__":
    main()
