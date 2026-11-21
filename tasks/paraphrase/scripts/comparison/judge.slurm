#!/bin/bash

#SBATCH --job-name=paraphrase-judge
#SBATCH --output=tasks/paraphrase/logs/comparison/%x-%j.out
#SBATCH --error=tasks/paraphrase/logs/comparison/%x-%j.err

#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=96G
#SBATCH --time=04:00:00

set -e

source .env
source .venv/bin/activate

cd tasks/paraphrase
export PYTHONDONTWRITEBYTECODE=1 NCCL_P2P_DISABLE=0 TOKENIZERS_PARALLELISM=false

exec uv run --active --no-sync python -u -m src.training.jobs judge \
  --config configs/comparison/judge_with_100m.yaml "$@"
