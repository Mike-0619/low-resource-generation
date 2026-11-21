#!/bin/bash

#SBATCH --job-name=greedy_gpt5_6_luna_en_full
#SBATCH --output=tasks/lambada/logs/gpt5.6-luna/English/%j.out
#SBATCH --error=tasks/lambada/logs/gpt5.6-luna/English/%j.err

#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:30:00

set -e

if (( $# )); then
  printf 'Usage: %s\n' "$0" >&2
  exit 2
fi

source .env
source .venv/bin/activate

uv run --active --no-sync python -u -m tasks.lambada.src.greedy \
  --backend api --language en \
  --model "openai/gpt-5.6-luna" \
  --reasoning_effort none --concurrency 4 \
  --output_dir "tasks/lambada/results/gpt5.6-luna/English"
