#!/bin/bash

#SBATCH --job-name=personachat-sft
#SBATCH --output=tasks/personachat/logs/sft/%x-%j.out
#SBATCH --error=tasks/personachat/logs/sft/%x-%j.err

#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00

set -e

source .env
source .venv/bin/activate

cd tasks/personachat

uv run --active --no-sync python -B -u -m src.training.pipeline --config configs/sft.yaml "$@"
