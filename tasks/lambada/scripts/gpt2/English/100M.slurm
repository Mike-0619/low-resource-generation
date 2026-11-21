#!/bin/bash

#SBATCH --job-name=greedy_gpt2_English_100M_full
#SBATCH --output=tasks/lambada/logs/gpt2/English/100M/%j.out
#SBATCH --error=tasks/lambada/logs/gpt2/English/100M/%j.err

#SBATCH --partition=lrz-hgx-h100-94x4,lrz-hgx-a100-80x4,lrz-dgx-a100-80x8
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:05:00

set -e

if (( $# )); then
  printf 'Usage: %s\n' "$0" >&2
  exit 2
fi

source .env
source .venv/bin/activate

uv run --active --no-sync python -u -m tasks.lambada.src.greedy \
  --backend local --language en \
  --model "models/gpt2/baseline/results/English/100M/seed_42/final" \
  --context_limit 512 --dtype float32 --bos always --batch_size 8 --device cuda \
  --output_dir "tasks/lambada/results/gpt2/English/100M"
