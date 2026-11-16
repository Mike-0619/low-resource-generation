#!/bin/bash

set -e

case "$1" in
  aggregate) module=aggregate ;;
  plot) module=plot_checkpoints ;;
  *)
    printf 'Usage: %s {aggregate|plot} [OPTIONS]\n' "$0" >&2
    exit 2
    ;;
esac
shift

source .env
source .venv/bin/activate

cd tasks/personachat

uv run --active --no-sync python -B -u -m "src.evaluation.$module" "$@"
