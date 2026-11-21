#!/bin/bash

#SBATCH --job-name=gpt2-tokenizer-100m-tr
#SBATCH --output=models/gpt2/baseline/logs/Turkish/100M/tokenizer/%j.out
#SBATCH --error=models/gpt2/baseline/logs/Turkish/100M/tokenizer/%j.err

#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00

set -e

source .env
source .venv/bin/activate

cd models/gpt2/baseline
export RAYON_NUM_THREADS=4

uv run --active --no-sync python -u -m src.data.train_tokenizer \
  --config configs/Turkish/100M.yaml
