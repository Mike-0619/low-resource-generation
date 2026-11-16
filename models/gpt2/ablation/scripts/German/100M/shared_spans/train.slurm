#!/bin/bash

#SBATCH --job-name=gpt2-ablation-100m-de-shared_spans
#SBATCH --output=models/gpt2/ablation/logs/German/100M/shared_spans/train/%A_%a.out
#SBATCH --error=models/gpt2/ablation/logs/German/100M/shared_spans/train/%A_%a.err

#SBATCH --partition=lrz-hgx-h100-94x4,lrz-hgx-a100-80x4,lrz-dgx-a100-80x8
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --array=0-5

set -e

source .env
source .venv/bin/activate

cd models/gpt2/ablation
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8

if [[ $# -ne 1 || ! $1 =~ ^(42|43|44)$ ]]; then
  printf 'Usage: %s {42|43|44}\n' "$0" >&2
  exit 2
fi

if [[ ! ${SLURM_ARRAY_TASK_ID:-} =~ ^[0-5]$ ]]; then
  printf 'SLURM_ARRAY_TASK_ID must be 0 through 5\n' >&2
  exit 2
fi

vocab_sizes=(512 1024 2048 4096 8192 16384)
vocab_size="${vocab_sizes[SLURM_ARRAY_TASK_ID]}"

uv run --active --no-sync python -u -m src.training.train \
  --config configs/German/100M/shared_spans.yaml \
  --vocab-size "$vocab_size" \
  --seed "$1"
