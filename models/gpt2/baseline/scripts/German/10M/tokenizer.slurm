#!/bin/bash

#SBATCH --job-name=gpt2-tokenizer-10m-de
#SBATCH --output=models/gpt2/baseline/logs/German/10M/tokenizer/%j.out
#SBATCH --error=models/gpt2/baseline/logs/German/10M/tokenizer/%j.err

#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00

set -e

source .env
source .venv/bin/activate

cd models/gpt2/baseline
export RAYON_NUM_THREADS=4

uv run --active --no-sync python -u -m src.data.train_tokenizer \
  --config configs/German/10M.yaml
