#!/bin/bash

#SBATCH --job-name=personachat-eval
#SBATCH --output=tasks/personachat/logs/eval/%x-%j.out
#SBATCH --error=tasks/personachat/logs/eval/%x-%j.err

#SBATCH --partition=lrz-hgx-h100-94x4,lrz-hgx-a100-80x4,lrz-dgx-a100-80x8
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=01:00:00

set -e

source .env
source .venv/bin/activate

cd tasks/personachat

uv run --active --no-sync python -B -u -m src.training.pipeline --config configs/eval.yaml "$@"
