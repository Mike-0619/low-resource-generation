
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from src.training.utils import PROJECT_ROOT, SEEDS, VOCAB_SIZES


FAMILIES = {
    ("English", "10M"): "10m_en",
    ("English", "100M"): "100m_en",
    ("German", "100M"): "100m_de",
}
PROTOCOLS = ("shared_spans", "balanced_batch")
PREFIX = "run --active --no-sync python -u -m "


def expected_calls(relative, vocab_size, seed):
    if relative.name == "prepare.slurm":
        language = relative.parts[0]
        sizes = ("10M", "100M") if language == "English" else ("100M",)
        calls = [f"src.models.initialization --config configs/{language}/{sizes[0]}/shared_spans.yaml"]
        for size in sizes:
            condition = FAMILIES[language, size]
            calls.extend([
                f"src.data.clean_dataset --condition {condition}_spans",
                f"src.data.clean_dataset --condition {condition}_balanced",
                f"src.data.train_tokenizer --condition {condition}_spans",
                *(f"src.data.data_utils --config configs/{language}/{size}/{protocol}.yaml"
                  for protocol in PROTOCOLS),
            ])
        return calls
    language, size = relative.parts[:2]
    if relative.name == "tokenizer.slurm":
        return [f"src.data.train_tokenizer --condition {FAMILIES[language, size]}_spans"]
    protocol = relative.parts[2]
    config = f"configs/{language}/{size}/{protocol}.yaml"
    if relative.name == "data.slurm":
        return [f"src.data.data_utils --config {config}"]
    return [f"src.training.train --config {config} --vocab-size {vocab_size} --seed {seed}"]


class LauncherTest(unittest.TestCase):
    def test_scripts_from_project_root_and_slurm_copy(self):
        scripts = sorted((PROJECT_ROOT / "scripts").rglob("*.slurm"))
        self.assertEqual(len(scripts), 17)
        self.assertEqual(
            {path.relative_to(PROJECT_ROOT / "scripts").as_posix()
             for path in scripts if path.name == "train.slurm"},
            {f"{language}/{size}/{protocol}/train.slurm"
             for language, size in FAMILIES for protocol in PROTOCOLS},
        )
        self.assertFalse(any(path.name.startswith("seed_") for path in scripts))
        self.assertEqual(
            {script.relative_to(PROJECT_ROOT / "scripts").as_posix()
             for script in scripts if script.name == "prepare.slurm"},
            {"English/prepare.slurm", "German/prepare.slurm"},
        )
        self.assertFalse((PROJECT_ROOT / "scripts/prepare_all.slurm").exists())
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "test project"
            work = root / "models/gpt2/ablation"
            work.mkdir(parents=True)
            (root / ".env").write_text('export CORPUS_ROOT="$PWD/test-corpora"\n')
            activate = root / ".venv/bin/activate"
            activate.parent.mkdir(parents=True)
            activate.write_text('export PATH="$PWD/.venv/bin:$PATH"\nexport VIRTUAL_ENV="$PWD/.venv"\n')
            binary = activate.parent / "uv"
            binary.write_text(
                '#!/bin/bash\nprintf "%s\\n" "$PWD|$VIRTUAL_ENV|$CORPUS_ROOT|$*" >> "$CAPTURE"\n'
                'exit "${FIXTURE_EXIT:-0}"\n'
            )
            binary.chmod(0o755)
            env = {
                "PATH": os.defpath,
                "CAPTURE": str(root / "calls"),
            }
            capture = Path(env["CAPTURE"])
            spool = root / "slurm-copy"
            for script in scripts:
                relative = script.relative_to(PROJECT_ROOT / "scripts")
                direct = work / "scripts" / relative
                direct.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(script, direct)
                text = direct.read_text()
                directives = dict(
                    line.removeprefix("#SBATCH --").split("=", 1)
                    for line in text.splitlines() if line.startswith("#SBATCH --")
                )
                is_train = script.name == "train.slurm"
                is_prepare = script.name == "prepare.slurm"
                with self.subTest(script=str(relative)):
                    subprocess.run(["bash", "-n", str(direct)], check=True)
                    self.assertTrue(text.startswith("#!/bin/bash\n\n#SBATCH --job-name="))
                    self.assertTrue(directives["job-name"].startswith("gpt2-ablation-"))
                    self.assertIn("set -e\n\nsource .env\nsource .venv/bin/activate", text)
                    self.assertEqual("OMP_NUM_THREADS=" in text, is_train or is_prepare)
                    self.assertEqual("RAYON_NUM_THREADS=" in text, not is_train)
                    if is_train or is_prepare:
                        self.assertIn("OMP_NUM_THREADS=" + directives["cpus-per-task"], text)
                    if not is_train:
                        self.assertIn("RAYON_NUM_THREADS=" + directives["cpus-per-task"], text)
                    for variable in ("CUDA_VISIBLE_DEVICES", "HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE"):
                        self.assertNotIn(variable, text)
                if is_train:
                    self.assertEqual(directives["gres"], "gpu:1")
                    self.assertEqual(directives["array"], "0-5")
                    self.assertEqual(directives["cpus-per-task"], "8")
                    self.assertEqual(directives["mem"], "16G" if "10M" in relative.parts else "32G")
                    self.assertEqual(directives["time"], "01:00:00" if "10M" in relative.parts else "06:00:00")
                    protocol_path = relative.parent.as_posix()
                    logs = f"models/gpt2/ablation/logs/{protocol_path}/train/%A_%a"
                    self.assertEqual(directives["output"], logs + ".out")
                    self.assertEqual(directives["error"], logs + ".err")
                if is_prepare:
                    language = relative.parts[0]
                    logs = f"models/gpt2/ablation/logs/{language}/prepare/%j"
                    self.assertEqual(directives["output"], logs + ".out")
                    self.assertEqual(directives["error"], logs + ".err")
                    self.assertEqual(directives["partition"], "lrz-cpu")
                    self.assertEqual(directives["qos"], "cpu")
                    self.assertEqual(directives["cpus-per-task"], "8")
                    self.assertEqual(directives["mem"], "64G")
                    self.assertEqual(directives["time"], "12:00:00")
                shutil.copyfile(direct, spool)
                for launcher in (direct, spool):
                    variants = (
                        [(seed, index, size) for seed in SEEDS for index, size in enumerate(VOCAB_SIZES)]
                        if is_train else [(None, None, None)]
                    )
                    for seed, array_index, vocab_size in variants:
                        with self.subTest(script=str(relative), copied=launcher == spool,
                                          seed=seed, index=array_index):
                            job_env = {**env, "SLURM_ARRAY_TASK_ID": str(array_index)} if is_train else env
                            args = [str(seed)] if is_train else []
                            capture.unlink(missing_ok=True)
                            subprocess.run(["bash", str(launcher), *args], cwd=root, check=True, env=job_env)
                            records = capture.read_text().splitlines()
                            self.assertEqual(records, [
                                f"{work}|{root / '.venv'}|{root / 'test-corpora'}|{PREFIX}{call}"
                                for call in expected_calls(relative, vocab_size, seed)
                            ])
                    if is_train:
                        invalid = [
                            ([], "0", "Usage:"),
                            (["45"], "0", "Usage:"),
                            (["42", "43"], "0", "Usage:"),
                            (["42"], None, "SLURM_ARRAY_TASK_ID"),
                            (["42"], "-1", "SLURM_ARRAY_TASK_ID"),
                            (["42"], "6", "SLURM_ARRAY_TASK_ID"),
                            (["42"], "abc", "SLURM_ARRAY_TASK_ID"),
                        ]
                        for args, array_index, message in invalid:
                            job_env = env if array_index is None else {**env, "SLURM_ARRAY_TASK_ID": array_index}
                            capture.unlink(missing_ok=True)
                            process = subprocess.run(["bash", str(launcher), *args], cwd=root, env=job_env,
                                                     text=True, capture_output=True)
                            with self.subTest(script=str(relative), copied=launcher == spool,
                                              args=args, index=array_index):
                                self.assertEqual(process.returncode, 2)
                                self.assertIn(message, process.stderr)
                                self.assertFalse(capture.exists())
                if is_prepare:
                    capture.unlink(missing_ok=True)
                    process = subprocess.run(["bash", str(direct)], cwd=root, env={**env, "FIXTURE_EXIT": "23"})
                    self.assertEqual(process.returncode, 23)
                    self.assertEqual(len(capture.read_text().splitlines()), 1)

