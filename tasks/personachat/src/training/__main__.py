import sys

from transformers import set_seed, HfArgumentParser
from utils.config_parser import PipelineArguments
from utils.logger import get_logger

from .trainer import run_personachat_finetune
from ..evaluation.evaluator import run_personachat_evaluate

def run_task(config_path: str):
    """
    Entry point for the PersonaChat task.
    Takes the path to the YAML config file as input.
    """
    parser = HfArgumentParser((PipelineArguments,))
    logger = get_logger(__name__)

    try:
        args = parser.parse_yaml_file(yaml_file=config_path)[0]
    except Exception as e:
        logger.error(f"Failed to parse YAML file at {config_path}. Error: {e}")
        sys.exit(1)

    set_seed(args.seed)

    logger.info("="*60)
    logger.info("START PIPELINE: PERSONACHAT")
    logger.info(f"Mode             : {args.mode.upper()} | Stage: {args.stage.upper()}")
    logger.info(f"Model Path       : {args.model_path}")
    logger.info(f"Output Directory : {args.output_dir}")
    logger.info(f"Use LoRA         : {args.use_lora}")
    logger.info("="*60)

    # Execute the appropriate mode
    if args.mode == "train":
        run_personachat_finetune(args)
    elif args.mode == "eval":
        run_personachat_evaluate(args)
    else:
        logger.error(f"Invalid mode: {args.mode}")
        raise ValueError(f"Invalid mode: {args.mode}. Choose 'train' or 'eval'.")

if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
        run_task(sys.argv[1])
    else:
        print("Usage: python -m src.training path/to/config.yaml")
        sys.exit(1)
