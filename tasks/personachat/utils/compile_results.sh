#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Get the absolute path of the directory containing this bash script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$( dirname "$SCRIPT_DIR" )"
PROJECT_ROOT="$(cd "$ROOT_DIR/../.." && pwd)"
source "$PROJECT_ROOT/.env" || exit $?
source "$PROJECT_ROOT/.venv/bin/activate" || exit $?
export PYTHONDONTWRITEBYTECODE=1

# Define default input and output directories
DEFAULT_EVAL_DIR="${ROOT_DIR}/outputs/eval"
DEFAULT_OUT_DIR="${ROOT_DIR}/results"

echo "=================================================="
echo " STARTING RESULTS COMPILATION"
echo " Time: $(date)"
echo "=================================================="

# Check if the user provided a specific target directory as an argument
# Usage: ./run_compile.sh [custom_dir_path]
if [ -n "$1" ]; then
    TARGET_DIR="$1"
    
    # If the user targets a very specific deep folder (e.g. rev_main_seed_42), 
    # we turn off batch mode to output a single file without nesting.
    # Otherwise, we keep batch=true to preserve folder structures.
    echo ">> Target directory provided: $TARGET_DIR"
    echo ">> Running Python script in BATCH = FALSE mode for specific target..."
    
    uv run --project "$PROJECT_ROOT" --active --locked --no-sync python "${SCRIPT_DIR}/compile_results.py" \
        -d "$TARGET_DIR" \
        -o "${DEFAULT_OUT_DIR}/custom_compilation" \
        --batch false
else
    TARGET_DIR="$DEFAULT_EVAL_DIR"
    echo ">> No target directory provided. Defaulting to all evaluations."
    echo ">> Scanning: $TARGET_DIR"
    echo ">> Running Python script in BATCH = TRUE mode..."
    
    # Run the Python compilation script globally
    uv run --project "$PROJECT_ROOT" --active --locked --no-sync python "${SCRIPT_DIR}/compile_results.py" \
        -d "$TARGET_DIR" \
        -o "$DEFAULT_OUT_DIR" \
        --batch true
fi

echo "=================================================="
echo " COMPILATION FINISHED SUCCESSFULLY!"
echo " Please check your CSV and JSON files in:"
echo " -> ${DEFAULT_OUT_DIR}"
echo "=================================================="