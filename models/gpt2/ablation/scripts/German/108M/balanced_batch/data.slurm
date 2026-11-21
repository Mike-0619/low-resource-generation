#!/bin/bash

#SBATCH --job-name=gpt2-ablation-data-108m-de-balanced_batch
#SBATCH --output=models/gpt2/ablation/logs/German/108M/balanced_batch/data/%j.out
#SBATCH --error=models/gpt2/ablation/logs/German/108M/balanced_batch/data/%j.err

#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00

set -e

source .env
source .venv/bin/activate

cd models/gpt2/ablation
export PYTHONDONTWRITEBYTECODE=1 RAYON_NUM_THREADS=8

uv run --active --no-sync python -u -m src.data.data_utils \
  --config configs/German/108M/balanced_batch.yaml
