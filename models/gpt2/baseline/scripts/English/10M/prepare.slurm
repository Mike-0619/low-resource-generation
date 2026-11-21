#!/bin/bash

#SBATCH --job-name=gpt2-baseline-prepare-10m-en
#SBATCH --output=models/gpt2/baseline/logs/English/10M/prepare/%j.out
#SBATCH --error=models/gpt2/baseline/logs/English/10M/prepare/%j.err

#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:20:00

set -e

source .env
source .venv/bin/activate

cd models/gpt2/baseline

uv run --active --no-sync python -u -m src.data.clean_dataset \
  --config configs/English/10M.yaml
