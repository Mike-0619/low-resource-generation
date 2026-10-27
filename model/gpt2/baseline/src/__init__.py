"""Independent LRZ BabyLM 2026 English and German baselines."""

from pathlib import Path as _CachePath
import runpy as _cache_runpy
_configure_cache = _cache_runpy.run_path(str(_CachePath(__file__).resolve().parents[6] / "config/cache.py"))["configure"]
_configure_cache('low-resource-generation')

