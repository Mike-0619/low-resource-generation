#!/bin/bash

#SBATCH --job-name=greedy_gpt2_German_108M_full
#SBATCH --output=tasks/lambada/logs/gpt2/German/108M/%j.out
#SBATCH --error=tasks/lambada/logs/gpt2/German/108M/%j.err

#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:05:00

set -e

source .env
source .venv/bin/activate

uv run --active --no-sync python -u -m tasks.lambada.src.greedy \
  --backend local --language de \
  --model "models/gpt2/baseline/results/German/108M/seed_42/final" \
  --context_limit 512 --dtype float32 --bos always --batch_size 8 --device cuda \
  --output_dir "tasks/lambada/results/gpt2/German/108M"
