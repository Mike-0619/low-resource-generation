#!/usr/bin/env bash
# Source with a project name, or execute with --print0 for Python entrypoints.
_cache_env_keys=(WORKSPACE_ROOT XDG_CACHE_HOME UV_CACHE_DIR HF_HOME HF_HUB_CACHE
  HUGGINGFACE_HUB_CACHE HF_DATASETS_CACHE HF_ASSETS_CACHE
  HF_XET_CACHE HF_MODULES_CACHE HF_EVALUATE_CACHE HF_METRICS_CACHE TORCH_HOME TORCHINDUCTOR_CACHE_DIR TRITON_HOME
  TRITON_CACHE_DIR TRITON_DUMP_DIR TRITON_OVERRIDE_DIR CUDA_CACHE_PATH NLTK_DATA
  MPLCONFIGDIR TIKTOKEN_CACHE_DIR PYTHONDONTWRITEBYTECODE PYTEST_ADDOPTS)

_cache_env_configure() {
  local cache_project="${1:-workspace}" cache_root cache_base cache_name cache_value cache_resolved
  local cache_hub='' cache_candidate cache_key
  cache_root="$(realpath -m -- "$(dirname -- "${BASH_SOURCE[0]}")/..")" || return 1
  cache_base="$cache_root/.cache"
  if [[ "$(realpath -m -- "$cache_base")" != "$cache_base" ]]; then
    printf 'Cache root must not redirect outside its physical workspace: %s\n' "$cache_base" >&2
    return 1
  fi
  case "$cache_project" in ''|.|..|*/*) printf 'Invalid cache project: %s\n' "$cache_project" >&2; return 1;; esac
  _cache_env_set() {
    local cache_name="$1" cache_value="${!1:-$2}" cache_resolved
    case "$cache_value" in '~') cache_value="$HOME";; '~/'*) cache_value="$HOME/${cache_value:2}";; esac
    [[ "$cache_value" = /* ]] || cache_value="$cache_root/$cache_value"
    cache_resolved="$(realpath -m -- "$cache_value")" || return 1
    case "$cache_resolved" in "$cache_base"|"$cache_base"/*) ;;
      *) printf 'Cache path rejected: %s=%s; allowed root: %s\n' "$cache_name" "${!cache_name:-$2}" "$cache_base" >&2; return 1;;
    esac
    export "$cache_name=$cache_resolved"
  }
  export WORKSPACE_ROOT="${WORKSPACE_ROOT:-$cache_root}"
  _cache_env_set XDG_CACHE_HOME "$cache_base" || return 1
  _cache_env_set UV_CACHE_DIR "$cache_base/uv" || return 1
  _cache_env_set HF_HOME "$cache_base/huggingface" || return 1
  for cache_key in HF_HUB_CACHE HUGGINGFACE_HUB_CACHE TRANSFORMERS_CACHE; do
    if [[ -n "${!cache_key:-}" ]]; then
      _cache_env_set "$cache_key" "$HF_HOME/hub" || return 1
      cache_candidate="${!cache_key}"
      if [[ -n "$cache_hub" && "$cache_hub" != "$cache_candidate" ]]; then
        printf 'Conflicting Hugging Face hub cache aliases: %s=%s, previous=%s\n' "$cache_key" "$cache_candidate" "$cache_hub" >&2
        return 1
      fi
      cache_hub="$cache_candidate"
    fi
  done
  export HF_HUB_CACHE="${cache_hub:-$HF_HOME/hub}"
  export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
  if [[ -n "${TRANSFORMERS_CACHE:-}" ]]; then export TRANSFORMERS_CACHE="$HF_HUB_CACHE"; fi
  _cache_env_set HF_HUB_CACHE "$HF_HOME/hub" || return 1
  _cache_env_set HF_DATASETS_CACHE "$HF_HOME/datasets" || return 1
  _cache_env_set HF_ASSETS_CACHE "$HF_HOME/assets" || return 1
  _cache_env_set HF_XET_CACHE "$HF_HOME/xet" || return 1
  _cache_env_set HF_MODULES_CACHE "$HF_HOME/modules" || return 1
  _cache_env_set HF_EVALUATE_CACHE "$HF_HOME/evaluate" || return 1
  _cache_env_set HF_METRICS_CACHE "$HF_HOME/metrics" || return 1
  _cache_env_set TORCH_HOME "$cache_base/torch" || return 1
  _cache_env_set TORCHINDUCTOR_CACHE_DIR "$cache_base/torchinductor" || return 1
  _cache_env_set TRITON_HOME "$cache_base/triton/home" || return 1
  _cache_env_set TRITON_CACHE_DIR "$cache_base/triton/cache" || return 1
  _cache_env_set TRITON_DUMP_DIR "$cache_base/triton/dump" || return 1
  _cache_env_set TRITON_OVERRIDE_DIR "$cache_base/triton/override" || return 1
  _cache_env_set CUDA_CACHE_PATH "$cache_base/cuda" || return 1
  _cache_env_set NLTK_DATA "$cache_base/nltk_data" || return 1
  _cache_env_set MPLCONFIGDIR "$cache_base/projects/$cache_project/matplotlib" || return 1
  _cache_env_set TIKTOKEN_CACHE_DIR "$cache_base/projects/$cache_project/tiktoken" || return 1
  if [[ -n "${PYTHONPYCACHEPREFIX:-}" ]]; then
    _cache_env_set PYTHONPYCACHEPREFIX "$cache_base/projects/$cache_project/pycache" || return 1
  fi
  export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"
  case " ${PYTEST_ADDOPTS:-} " in *'no:cacheprovider'*) ;;
    *) export PYTEST_ADDOPTS="${PYTEST_ADDOPTS:+$PYTEST_ADDOPTS }-p no:cacheprovider";; esac
  unset -f _cache_env_set
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  _cache_env_configure "${1:-workspace}" || exit $?
  if [[ "${2:-}" == --print0 ]]; then
    for _cache_env_key in "${_cache_env_keys[@]}"; do
      printf '%s\0%s\0' "$_cache_env_key" "${!_cache_env_key}"
    done
    if [[ -n "${TRANSFORMERS_CACHE:-}" ]]; then printf 'TRANSFORMERS_CACHE\0%s\0' "$TRANSFORMERS_CACHE"; fi
    if [[ -n "${PYTHONPYCACHEPREFIX:-}" ]]; then printf 'PYTHONPYCACHEPREFIX\0%s\0' "$PYTHONPYCACHEPREFIX"; fi
  fi
else
  _cache_env_configure "${1:-workspace}"
fi
