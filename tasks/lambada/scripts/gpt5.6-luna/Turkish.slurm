#!/bin/bash

#SBATCH --job-name=greedy_gpt5_6_luna_tr
#SBATCH --output=tasks/lambada/logs/gpt5.6-luna/Turkish/%j.out
#SBATCH --error=tasks/lambada/logs/gpt5.6-luna/Turkish/%j.err

#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:30:00

set -e

source .env
source .venv/bin/activate

uv run --active --no-sync python -u -m tasks.lambada.src.greedy \
  --backend api --language tr \
  --model "openai/gpt-5.6-luna" \
  --reasoning_effort none --concurrency 4 \
  --output_dir "tasks/lambada/results/gpt5.6-luna/Turkish"
