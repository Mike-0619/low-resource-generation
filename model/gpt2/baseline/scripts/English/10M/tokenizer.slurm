#!/bin/bash

#SBATCH --job-name=gpt2-tokenizer-10m-en
#SBATCH --output=model/gpt2/baseline/logs/English/10M/tokenizer/%j.out
#SBATCH --error=model/gpt2/baseline/logs/English/10M/tokenizer/%j.err

#SBATCH --partition=lrz-cpu
#SBATCH --qos=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00

set -e
set +x

source .env
source .venv/bin/activate

cd model/gpt2/baseline
export CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 OMP_NUM_THREADS=4 RAYON_NUM_THREADS=4

uv run --active --locked --no-sync python -m src.train_tokenizer \
  --config configs/English/10M.yaml
