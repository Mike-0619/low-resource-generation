#!/bin/bash

#SBATCH --job-name=greedy_llama3_en_full
#SBATCH --output=tasks/lambada/logs/llama3/full/%j.out
#SBATCH --error=tasks/lambada/logs/llama3/full/%j.err

#SBATCH --partition=lrz-hgx-h100-94x4,lrz-hgx-a100-80x4,lrz-dgx-a100-80x8
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:10:00

set -e

source .env
source .venv/bin/activate

uv run --active --no-sync python -u -m tasks.lambada.src.greedy \
  --backend local --language en \
  --model "models/llama3/Meta-Llama-3-8B" \
  --context_limit 8192 --dtype bfloat16 --bos always --all --batch_size 8 --device cuda \
  --output_dir "tasks/lambada/results/llama3/full"
