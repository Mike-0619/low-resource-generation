#!/bin/bash

#SBATCH --job-name=gpt2-baseline-10m-de-s42
#SBATCH --output=models/gpt2/baseline/logs/German/10M/train/%j.out
#SBATCH --error=models/gpt2/baseline/logs/German/10M/train/%j.err

#SBATCH --partition=lrz-hgx-h100-94x4,lrz-hgx-a100-80x4,lrz-dgx-a100-80x8
#SBATCH --qos=gpu
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
  --config configs/German/10M.yaml
