#!/bin/bash

#SBATCH --job-name=gpt2-ablation-tokenizer-100m-de
#SBATCH --output=models/gpt2/ablation/logs/German/100M/tokenizer/%j.out
#SBATCH --error=models/gpt2/ablation/logs/German/100M/tokenizer/%j.err

#SBATCH --partition=lrz-cpu
#SBATCH --qos=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00

set -e

source .env
source .venv/bin/activate

cd models/gpt2/ablation
export PYTHONDONTWRITEBYTECODE=1 RAYON_NUM_THREADS=8

uv run --active --no-sync python -u -m src.data.train_tokenizer \
  --condition 100m_de_spans
