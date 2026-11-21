#!/bin/bash

#SBATCH --job-name=gpt2-baseline-10m-tr-s42
#SBATCH --output=models/gpt2/baseline/logs/Turkish/10M/train/%j.out
#SBATCH --error=models/gpt2/baseline/logs/Turkish/10M/train/%j.err

#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00

set -e

source .env
source .venv/bin/activate

cd models/gpt2/baseline
export OMP_NUM_THREADS=8

uv run --active --no-sync python -u -m src.training.train \
  --config configs/Turkish/10M.yaml
