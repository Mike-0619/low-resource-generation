#!/bin/bash

#SBATCH --job-name=personachat-dpo
#SBATCH --output=tasks/personachat/logs/dpo/%x-%j.out
#SBATCH --error=tasks/personachat/logs/dpo/%x-%j.err

#SBATCH --partition=lrz-hgx-h100-94x4
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00

set -e

source .env
source .venv/bin/activate

cd tasks/personachat

uv run --active --no-sync python -B -u -m src.training.pipeline --config configs/dpo.yaml "$@"
