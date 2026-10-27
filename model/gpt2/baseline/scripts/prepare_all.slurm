#!/bin/bash

#SBATCH --job-name=gpt2-baseline-prepare-all
#SBATCH --output=model/gpt2/baseline/logs/preparation/%j.out
#SBATCH --error=model/gpt2/baseline/logs/preparation/%j.err

#SBATCH --partition=lrz-cpu
#SBATCH --qos=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:20:00

set -e
set +x

source .env
source .venv/bin/activate

cd model/gpt2/baseline
export CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 OMP_NUM_THREADS=4

for config in configs/English/10M.yaml configs/English/100M.yaml \
              configs/German/10M.yaml configs/German/108M.yaml; do
  uv run --active --locked --no-sync python -m src.clean_dataset --config "$config"
done
