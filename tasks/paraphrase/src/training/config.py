"""Experiment YAML and command-line overrides."""

import argparse
from pathlib import Path

import yaml


PROMPT_TYPES = ("simple_zero_shot", "zero_shot_preserve", "few_shot")


def str2bool(value):
    if value.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if value.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run a Paraphrase experiment stage", allow_abbrev=False)
    parser.add_argument("--config", type=Path, required=True, help="Path to experiment YAML")
    parser.add_argument(
        "--mode", required=True,
        choices=("train", "llama_benchmark", "parapluie", "database", "judge"),
    )
    parser.add_argument("--seed", type=int)

    # GPT-2 fine-tuning and generation
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--report_path", type=str)
    parser.add_argument("--parapluie_report_path", type=str)
    parser.add_argument("--max_length", type=int)
    parser.add_argument("--qqp_augment_size", type=int)
    parser.add_argument("--num_train_epochs", type=int)
    parser.add_argument("--per_device_train_batch_size", type=int)
    parser.add_argument("--gradient_accumulation_steps", type=int)
    parser.add_argument("--per_device_eval_batch_size", type=int)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--weight_decay", type=float)
    parser.add_argument("--warmup_steps", type=int)
    parser.add_argument("--lr_scheduler_type", type=str)
    parser.add_argument("--logging_steps", type=int)
    parser.add_argument("--eval_strategy", type=str)
    parser.add_argument("--save_strategy", type=str)
    parser.add_argument("--save_total_limit", type=int)
    parser.add_argument("--load_best_model_at_end", type=str2bool)
    parser.add_argument("--metric_for_best_model", type=str)
    parser.add_argument("--bf16", type=str2bool)
    parser.add_argument("--tf32", type=str2bool)
    parser.add_argument("--optim", type=str)
    parser.add_argument("--report_to", type=str)
    parser.add_argument("--inference_batch_size", type=int)
    parser.add_argument("--max_new_tokens", type=int)
    parser.add_argument("--min_new_tokens", type=int)
    parser.add_argument("--num_beams", type=int)
    parser.add_argument("--repetition_penalty", type=float)
    parser.add_argument("--length_penalty", type=float)
    parser.add_argument("--do_sample", type=str2bool)
    parser.add_argument("--early_stopping", type=str2bool)

    # Llama benchmarking
    parser.add_argument("--model_id", type=str)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--max_input_length", type=int)
    parser.add_argument("--with_parapluie", type=str2bool)
    parser.add_argument("--judge_model_id", type=str)
    parser.add_argument(
        "--prompt_type",
        choices=PROMPT_TYPES,
    )

    # Report aggregation
    parser.add_argument("--reports", nargs='+')
    parser.add_argument("--names", nargs='+')
    parser.add_argument("--output_json", type=str)
    parser.add_argument("--output_csv", type=str)

    # ParaPLUIE report scoring
    parser.add_argument("--from_report", type=str)
    parser.add_argument("--out_report", type=str)

    # LLM judge
    parser.add_argument("--judge_model_path", type=str)
    parser.add_argument("--db_path", type=str)
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--limit_samples", type=int)
    parser.add_argument("--port", type=int)
    parser.add_argument("--model_name", type=str)

    args = parser.parse_args(argv)
    with args.config.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        parser.error(f"{args.config} must contain a YAML mapping")

    runs = config.pop("runs", None)
    if runs is not None or args.mode == "llama_benchmark":
        prompt = args.prompt_type or config.get("prompt_type", "few_shot")
        if prompt not in PROMPT_TYPES:
            parser.error(f"Unsupported prompt type: {prompt}")
        if runs is not None:
            if prompt not in runs:
                parser.error(f"No run configured for {prompt} in {args.config}")
            config.update(runs[prompt])
        config["prompt_type"] = prompt

    config.update({key: value for key, value in vars(args).items() if value is not None})
    return argparse.Namespace(**config)
