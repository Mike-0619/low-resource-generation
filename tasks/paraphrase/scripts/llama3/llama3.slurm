#!/bin/bash

#SBATCH --job-name=paraphrase-llama
#SBATCH --output=tasks/paraphrase/logs/llama3/%x-%j.out
#SBATCH --error=tasks/paraphrase/logs/llama3/%x-%j.err

#SBATCH --partition=lrz-hgx-h100-94x4,lrz-hgx-a100-80x4,lrz-dgx-a100-80x8
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=32G
#SBATCH --time=02:00:00

set -e

source .env
source .venv/bin/activate

cd tasks/paraphrase
export PYTHONDONTWRITEBYTECODE=1 NCCL_P2P_DISABLE=0 TOKENIZERS_PARALLELISM=false

exec uv run --active --no-sync python -u -m src.training.jobs llama \
  --config configs/llama3/llama3.yaml "$@"
