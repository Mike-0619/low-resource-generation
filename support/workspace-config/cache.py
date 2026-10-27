"""Use the workspace's shell cache policy before third-party imports."""
from pathlib import Path
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
KEYS = tuple("WORKSPACE_ROOT XDG_CACHE_HOME UV_CACHE_DIR HF_HOME HF_HUB_CACHE HUGGINGFACE_HUB_CACHE HF_DATASETS_CACHE HF_ASSETS_CACHE HF_XET_CACHE HF_MODULES_CACHE HF_EVALUATE_CACHE HF_METRICS_CACHE TORCH_HOME TORCHINDUCTOR_CACHE_DIR TRITON_HOME TRITON_CACHE_DIR TRITON_DUMP_DIR TRITON_OVERRIDE_DIR CUDA_CACHE_PATH NLTK_DATA MPLCONFIGDIR TIKTOKEN_CACHE_DIR PYTHONDONTWRITEBYTECODE PYTEST_ADDOPTS".split())

def configure(project="workspace"):
    result = subprocess.run(["bash", str(ROOT / "config/cache.sh"), project, "--print0"],
                            capture_output=True, text=True, env=os.environ)
    if result.returncode:
        raise ValueError(result.stderr.strip())
    fields = result.stdout.rstrip("\0").split("\0")
    values = dict(zip(fields[::2], fields[1::2], strict=True))
    os.environ.update(values)
    sys.dont_write_bytecode = values["PYTHONDONTWRITEBYTECODE"] not in ("", "0")
    return values
