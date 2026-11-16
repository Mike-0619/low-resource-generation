"""Run one PersonaChat training or evaluation stage."""

import argparse
import logging
import sys

from transformers import set_seed
from yaml import YAMLError

from ..evaluation.evaluator import run_personachat_evaluate
from .config import PipelineArguments, load_config
from .trainer import run_personachat_finetune


def run_task(config: PipelineArguments) -> None:
    set_seed(config.seed)
    logging.getLogger(__name__).info(
        "%s/%s using %s → %s", config.mode, config.stage, config.model_path, config.output_dir,
    )
    if config.mode == "train":
        run_personachat_finetune(config)
    else:
        run_personachat_evaluate(config)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a PersonaChat YAML configuration")
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
    except (OSError, ValueError, TypeError, YAMLError) as exc:
        parser.error(f"could not load {args.config}: {exc}")
    if config.mode not in ("train", "eval"):
        parser.error("mode must be train or eval")
    if config.stage not in ("sft", "dpo"):
        parser.error("stage must be sft or dpo")
    logging.basicConfig(
        level=logging.INFO, stream=sys.stdout,
        format="[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S",
    )
    run_task(config)


if __name__ == "__main__":
    main()
