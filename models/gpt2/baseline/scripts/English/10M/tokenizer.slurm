#!/bin/bash

#SBATCH --job-name=gpt2-tokenizer-10m-en
#SBATCH --output=models/gpt2/baseline/logs/English/10M/tokenizer/%j.out
#SBATCH --error=models/gpt2/baseline/logs/English/10M/tokenizer/%j.err

#SBATCH --partition=lrz-cpu
#SBATCH --qos=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00

set -e

source .env
source .venv/bin/activate

cd models/gpt2/baseline
export RAYON_NUM_THREADS=4

uv run --active --no-sync python -u -m src.data.train_tokenizer \
  --config configs/English/10M.yaml
