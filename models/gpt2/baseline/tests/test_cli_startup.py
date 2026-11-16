"""Command help works in a checkout without workspace cache helpers."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
MODULES = {
    "src.corpora.prepare_babylm": "--dataset",
    "src.data.clean_dataset": "--config",
    "src.data.train_tokenizer": "--config",
    "src.training.train": "--config",
}


class CLIStartupTest(unittest.TestCase):
    def test_help_without_workspace_configuration(self):
        with tempfile.TemporaryDirectory(dir=os.environ.get("XDG_CACHE_HOME")) as directory:
            root = Path(directory) / "standalone project"
            shutil.copytree(ROOT / "src", root / "src", ignore=shutil.ignore_patterns("__pycache__"))
            env = {key: value for key, value in os.environ.items()
                   if key not in {"WORKSPACE_ROOT", "PYTHONPATH"}}
            for module, option in MODULES.items():
                with self.subTest(module=module):
                    result = subprocess.run(
                        [sys.executable, "-B", "-m", module, "--help"],
                        cwd=root, env=env, capture_output=True, text=True, timeout=90,
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn(option, result.stdout)
