#!/bin/bash

# Get the absolute path of the directory containing this bash script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$( dirname "$SCRIPT_DIR" )"
PROJECT_ROOT="$(cd "$ROOT_DIR/../.." && pwd)"
source "$PROJECT_ROOT/.env" || exit $?
source "$PROJECT_ROOT/.venv/bin/activate" || exit $?
export PYTHONDONTWRITEBYTECODE=1

# Define the list of models to process
MODELS=(
    "pythia-160m"
    "pythia-410m"
    "pythia-1b"
    "pythia-1.4b"
    # Add other models here as needed
)

# Prefix without the underscore
PREFIX="checkpoints_v1"

# Enable nullglob so the loop doesn't run if no directories are found
shopt -s nullglob

for MODEL in "${MODELS[@]}"; do
    # Define directories relative to the root location for each model
    BASE_EVAL_DIR="${ROOT_DIR}/outputs/eval/${MODEL}"
    MODEL_OUT_DIR="${ROOT_DIR}/figures/${MODEL}"

    # Ensure the output directory for the current model exists
    mkdir -p "$MODEL_OUT_DIR"

    echo "Scanning directories: ${BASE_EVAL_DIR}/${PREFIX}*"
    echo "=================================================="

    # Loop through all matching directories for the specific model
    for DIR in "${BASE_EVAL_DIR}/${PREFIX}"*; do
        if [ -d "$DIR" ]; then
            # Extract the base name (e.g., checkpoints_v1 or checkpoints_v1_20260924_220330)
            DIR_NAME=$(basename "$DIR")
            
            # Remove the prefix to get the suffix (will be empty "" or "_2026...")
            SUFFIX=${DIR_NAME#"$PREFIX"}
            
            # Define the output image path with the extracted suffix
            OUT_IMAGE="${MODEL_OUT_DIR}/checkpoint_eval_metrics${SUFFIX}.png"
            
            echo ">> Processing: $DIR"
            echo ">> Saving image to: $OUT_IMAGE"
            
            # Run the Python script
            uv run --project "$PROJECT_ROOT" --active --locked --no-sync python "${SCRIPT_DIR}/plot_checkpoints_eval.py" -d "$DIR" -o "$OUT_IMAGE"
            
            echo "--------------------------------------------------"
        fi
    done
done

echo "Finished scanning and plotting for all models!"