#!/bin/bash

#SBATCH --job-name=gpt2-baseline-108m-de-s42
#SBATCH --output=models/gpt2/baseline/logs/German/108M/train/%j.out
#SBATCH --error=models/gpt2/baseline/logs/German/108M/train/%j.err

#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00

set -e

source .env
source .venv/bin/activate

cd models/gpt2/baseline
export OMP_NUM_THREADS=8

uv run --active --no-sync python -u -m src.training.train \
  --config configs/German/108M.yaml
