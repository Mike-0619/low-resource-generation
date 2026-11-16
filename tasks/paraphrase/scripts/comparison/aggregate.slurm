#!/bin/bash

#SBATCH --job-name=paraphrase-aggregate
#SBATCH --output=tasks/paraphrase/logs/comparison/%x-%j.out
#SBATCH --error=tasks/paraphrase/logs/comparison/%x-%j.err

#SBATCH --partition=lrz-cpu
#SBATCH --qos=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=00:10:00

set -e

source .env
source .venv/bin/activate

cd tasks/paraphrase
export PYTHONDONTWRITEBYTECODE=1 NCCL_P2P_DISABLE=0 TOKENIZERS_PARALLELISM=false

exec uv run --active --no-sync python -u -m src.training.pipeline \
  --config configs/comparison/aggregate_with_100m.yaml "$@" --mode database
