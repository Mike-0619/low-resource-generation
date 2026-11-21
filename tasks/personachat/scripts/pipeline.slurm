#!/bin/bash

#SBATCH --job-name=personachat-pipeline
#SBATCH --output=tasks/personachat/logs/pipeline/%x-%j.out
#SBATCH --error=tasks/personachat/logs/pipeline/%x-%j.err

#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=09:00:00

set -e

source .env
source .venv/bin/activate

cd tasks/personachat

uv run --active --no-sync python -B -u -m src.training.pipeline --config configs/sft.yaml
uv run --active --no-sync python -B -u -m src.training.pipeline --config configs/dpo.yaml
uv run --active --no-sync python -B -u -m src.training.pipeline --config configs/eval.yaml
uv run --active --no-sync python -B -u -m src.evaluation.aggregate \
    --file results/eval/dpo/dpo_eval_results.json --output results/aggregate/pipeline
