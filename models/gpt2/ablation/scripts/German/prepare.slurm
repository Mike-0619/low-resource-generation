#!/bin/bash

#SBATCH --job-name=gpt2-ablation-prepare-de
#SBATCH --output=models/gpt2/ablation/logs/German/prepare/%j.out
#SBATCH --error=models/gpt2/ablation/logs/German/prepare/%j.err

#SBATCH --partition=lrz-cpu
#SBATCH --qos=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00

set -e

source .env
source .venv/bin/activate

cd models/gpt2/ablation
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 RAYON_NUM_THREADS=8

uv run --active --no-sync python -u -m src.models.initialization \
  --config configs/German/100M/shared_spans.yaml

uv run --active --no-sync python -u -m src.data.clean_dataset \
  --condition 100m_de_spans

uv run --active --no-sync python -u -m src.data.clean_dataset \
  --condition 100m_de_balanced

uv run --active --no-sync python -u -m src.data.train_tokenizer \
  --condition 100m_de_spans

uv run --active --no-sync python -u -m src.data.data_utils \
  --config configs/German/100M/shared_spans.yaml

uv run --active --no-sync python -u -m src.data.data_utils \
  --config configs/German/100M/balanced_batch.yaml
