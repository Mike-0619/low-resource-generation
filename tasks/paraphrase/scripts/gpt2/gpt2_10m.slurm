#!/bin/bash

#SBATCH --job-name=paraphrase-gpt2-10m
#SBATCH --output=tasks/paraphrase/logs/gpt2/%x-%j.out
#SBATCH --error=tasks/paraphrase/logs/gpt2/%x-%j.err

#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=32G
#SBATCH --time=01:30:00

set -e

source .env
source .venv/bin/activate

cd tasks/paraphrase
export PYTHONDONTWRITEBYTECODE=1 NCCL_P2P_DISABLE=0 TOKENIZERS_PARALLELISM=false

exec uv run --active --no-sync python -u -m src.training.jobs gpt2 --size 10M \
  --config configs/gpt2/gpt2_10m.yaml "$@"
