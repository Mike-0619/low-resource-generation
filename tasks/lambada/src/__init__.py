"""LAMBADA package paths and generation budget."""
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = EVAL_DIR.parents[1]

MAX_NEW_TOKENS = 128
