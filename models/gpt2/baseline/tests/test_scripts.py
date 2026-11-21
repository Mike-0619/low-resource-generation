"""Independent job scripts: root-relative execution and environment regressions."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[4]
PREPARATION_SIZES = {
    "English": ("10M", "100M"),
    "German": ("10M", "108M"),
    "Turkish": ("10M", "100M"),
}
EXPECTED = {}
for language, sizes in PREPARATION_SIZES.items():
    script_dir = f"models/gpt2/baseline/scripts/{language}"
    for size in sizes:
        for stage, module, exports in (
            ("prepare", "data.clean_dataset", {}),
            ("tokenizer", "data.train_tokenizer", {"RAYON_NUM_THREADS": "4"}),
            ("train", "training.train", {"OMP_NUM_THREADS": "8"}),
        ):
            EXPECTED[f"{script_dir}/{size}/{stage}.slurm"] = {
                "commands": [["python", "-u", "-m", f"src.{module}", "--config",
                              f"configs/{language}/{size}.yaml"]],
                "exports": exports,
            }

class ScriptTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='independent scripts ', dir=os.environ.get('XDG_CACHE_HOME'))
        self.addCleanup(self.temporary.cleanup)
        self.temp = Path(self.temporary.name)
        self.workspace = self.temp / 'workspace with spaces'
        self.root = self.workspace / 'Projects/project with spaces'
        self.cache = self.workspace / '.cache'
        self.default_datasets = self.cache / 'projects/low-resource-generation/datasets'
        cache_helper = self.workspace / '.agents/config/cache/env.sh'
        cache_helper.parent.mkdir(parents=True)
        shutil.copyfile(ROOT.parents[1] / '.agents/config/cache/env.sh', cache_helper)
        for directory in ['.venv/bin', 'fixture-bin', 'models/gpt2/baseline']:
            (self.root / directory).mkdir(parents=True, exist_ok=True)
        for relative in EXPECTED:
            target = self.root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / relative, target)
        self.dotenv = self.root / '.env'
        self.dotenv.write_text((ROOT / '.env.example').read_text().replace(
            str(ROOT.parents[1]), str(self.workspace)).replace(
            '${OPENAI_API_KEY:-}', '${OPENAI_API_KEY:-DOTENV_FIXTURE_MARKER}'))
        self.activate = self.root / '.venv/bin/activate'
        self.activate.write_text('export VIRTUAL_ENV="$PWD/.venv"\n')
        self.uv = self.root / 'fixture-bin/uv'
        self.uv.write_text(f'#!{sys.executable}\n' + '''import json, os, sys
print(json.dumps({'args':sys.argv[1:], 'cwd':os.getcwd(),
 'venv':os.environ.get('VIRTUAL_ENV'), 'corpus':os.environ.get('CORPUS_ROOT'),
 'cache':os.environ.get('HF_HOME'), 'datasets':os.environ.get('HF_DATASETS_CACHE'),
 'workspace':os.environ.get('WORKSPACE_ROOT'), 'project_cache':os.environ.get('MPLCONFIGDIR'),
 'api_url':os.environ.get('OPENAI_BASE_URL'),
 'xdg_cache':os.environ.get('XDG_CACHE_HOME'), 'uv_cache':os.environ.get('UV_CACHE_DIR'),
 'bytecode':os.environ.get('PYTHONDONTWRITEBYTECODE'),
 'api':os.environ.get('OPENAI_API_KEY') in ['DOTENV_FIXTURE_MARKER','EXPORTED_FIXTURE_MARKER'],
 'api_override':os.environ.get('OPENAI_API_KEY')=='EXPORTED_FIXTURE_MARKER',
 'task_env':{k:os.environ[k] for k in ['HF_HUB_OFFLINE','HF_DATASETS_OFFLINE','OMP_NUM_THREADS','RAYON_NUM_THREADS','CUDA_VISIBLE_DEVICES'] if k in os.environ}}))
sys.exit(int(os.environ.get('FIXTURE_EXIT','0')))
''')
        self.uv.chmod(0o700)
        self.env = {'HOME':str(self.root), 'PATH':str(self.uv.parent)+':'+os.defpath}

    def execute(self, relative, extra=None, spooled=False, code=0, cwd=None):
        script = self.root / relative
        if spooled:
            copied = self.temp / 'slurm_script'
            shutil.copyfile(script, copied)
            script = copied
        result = subprocess.run(['bash', str(script)],
            cwd=cwd or self.root, env={**self.env, **(extra or {})}, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, code, result.stderr)
        for secret in ['DOTENV_FIXTURE_MARKER', 'EXPORTED_FIXTURE_MARKER']:
            self.assertNotIn(secret, result.stdout + result.stderr)
        return [json.loads(line) for line in result.stdout.splitlines() if line.startswith('{')]

    def test_slurm_partition_and_qos_match_task(self):
        script_root = ROOT / "models/gpt2/baseline/scripts"
        self.assertEqual({path.relative_to(ROOT).as_posix() for path in script_root.rglob("*.slurm")},
                         set(EXPECTED))
        for relative in EXPECTED:
            gpu = Path(relative).name == "train.slurm"
            lines = (ROOT / relative).read_text().splitlines()
            partition = 'lrz-hgx-h100-94x4,lrz-hgx-a100-80x4,lrz-dgx-a100-80x8' if gpu else 'lrz-cpu'
            self.assertEqual([line for line in lines if line.startswith('#SBATCH --partition=')],
                             ['#SBATCH --partition=' + partition])
            self.assertEqual([line for line in lines if line.startswith('#SBATCH --qos=')],
                             ['#SBATCH --qos=' + ('gpu' if gpu else 'cpu')])
            self.assertEqual(any(line.startswith('#SBATCH --gres=') for line in lines), gpu)
            language, size, filename = Path(relative).parts[-3:]
            log = f"models/gpt2/baseline/logs/{language}/{size}/{Path(filename).stem}/%j"
            self.assertIn(f"#SBATCH --output={log}.out", lines)
            self.assertIn(f"#SBATCH --error={log}.err", lines)

    def test_training_host_memory_matches_corpus_size(self):
        for language, sizes in PREPARATION_SIZES.items():
            for size in sizes:
                with self.subTest(language=language, size=size):
                    script = f"models/gpt2/baseline/scripts/{language}/{size}/train.slurm"
                    lines = (ROOT / script).read_text().splitlines()
                    self.assertEqual([line for line in lines if line.startswith("#SBATCH --mem=")],
                                     ["#SBATCH --mem=" + ("16G" if size == "10M" else "64G")])

    def test_all_jobs_direct_and_slurm_copies(self):
        for relative, expected in EXPECTED.items():
            for spooled in [False, True]:
                with self.subTest(script=relative, spooled=spooled):
                    records = self.execute(relative, spooled=spooled)
                    self.assertEqual([r['args'] for r in records],
                                     [['run','--active','--no-sync'] + command for command in expected['commands']])
                    for record in records:
                        self.assertEqual(record['cwd'],str(self.root / 'models/gpt2/baseline'))
                        self.assertEqual(record['venv'],str(self.root / '.venv'))
                        self.assertEqual(record['task_env'],expected['exports'])
                        self.assertTrue(record['api'])

    def test_prepare_propagates_the_cleaning_exit_status(self):
        for relative, expected in EXPECTED.items():
            if relative.endswith("/prepare.slurm"):
                with self.subTest(script=relative):
                    records = self.execute(relative, extra={"FIXTURE_EXIT": "23"}, code=23)
                    command, = expected["commands"]
                    self.assertEqual([r["args"] for r in records],
                                     [["run", "--active", "--no-sync", *command]])

    def test_environment_defaults_and_exported_overrides(self):
        job = next(iter(EXPECTED))
        record = self.execute(job)[0]
        self.assertEqual(record['workspace'], str(self.workspace))
        self.assertEqual(record['corpus'], str(self.workspace / 'Corpora'))
        self.assertEqual(record['xdg_cache'], str(self.cache))
        self.assertEqual(record['uv_cache'], str(self.cache / 'uv'))
        self.assertEqual(record['cache'], str(self.cache / 'huggingface'))
        self.assertEqual(record['datasets'], str(self.default_datasets))
        self.assertEqual(record['project_cache'], str(self.cache / 'projects/low-resource-generation/matplotlib'))
        self.assertEqual(record['bytecode'], '1')
        self.assertEqual(record['api_url'], 'https://openrouter.ai/api/v1')
        shared_cache = str(self.cache / 'alternate cache')
        record = self.execute(job, extra={'XDG_CACHE_HOME': shared_cache})[0]
        self.assertEqual(record['xdg_cache'], shared_cache)
        self.assertEqual(record['uv_cache'], str(self.cache / 'uv'))
        overrides = {
            'XDG_CACHE_HOME': shared_cache,
            'HF_HOME': str(self.cache / 'custom/hf'),
            'HF_DATASETS_CACHE': str(self.cache / 'custom/datasets'),
            'UV_CACHE_DIR': str(self.cache / 'custom/uv'),
            'CORPUS_ROOT': str(self.temp / 'custom/corpora'),
            'OPENAI_API_KEY': 'EXPORTED_FIXTURE_MARKER',
            'OPENAI_BASE_URL': 'https://provider.example/v1',
            'PYTHONDONTWRITEBYTECODE': '1',
        }
        record = self.execute(job, extra=overrides)[0]
        self.assertEqual(record['cache'], overrides['HF_HOME'])
        self.assertEqual(record['datasets'], overrides['HF_DATASETS_CACHE'])
        self.assertEqual(record['xdg_cache'], shared_cache)
        self.assertEqual(record['uv_cache'], overrides['UV_CACHE_DIR'])
        self.assertEqual(record['corpus'], overrides['CORPUS_ROOT'])
        self.assertEqual(record['bytecode'], '1')
        self.assertTrue(record['api_override'])
        self.assertEqual(record['api_url'], overrides['OPENAI_BASE_URL'])
        record = self.execute(job, extra={key: '' for key in overrides})[0]
        self.assertEqual(record['corpus'], str(self.workspace / 'Corpora'))
        self.assertEqual(record['xdg_cache'], str(self.cache))
        self.assertEqual(record['uv_cache'], str(self.cache / 'uv'))
        self.assertEqual(record['cache'], str(self.cache / 'huggingface'))
        self.assertEqual(record['datasets'], str(self.default_datasets))
        self.assertEqual(record['bytecode'], '1')
        self.assertEqual(record['api_url'], 'https://openrouter.ai/api/v1')
        self.assertTrue(record['api'])

    def test_outside_cache_overrides_stop_before_launch(self):
        for variable in ('XDG_CACHE_HOME', 'UV_CACHE_DIR', 'HF_HOME', 'HF_DATASETS_CACHE'):
            with self.subTest(variable=variable):
                self.assertEqual(self.execute(next(iter(EXPECTED)),
                    extra={variable: str(self.temp / 'outside-cache')}, code=1), [])

    def test_missing_setup_stops_before_execution(self):
        job = next(iter(EXPECTED))
        text = self.dotenv.read_text(); self.dotenv.unlink()
        self.assertEqual(self.execute(job,code=1),[])
        self.dotenv.write_text(text)
        text = self.activate.read_text(); self.activate.unlink()
        self.assertEqual(self.execute(job,code=1),[])
        self.activate.write_text(text)
        self.uv.rename(self.uv.with_suffix('.disabled'))
        self.assertEqual(self.execute(job,code=127),[])

    def test_child_and_environment_failures_propagate(self):
        jobs = [job for job in EXPECTED if job.endswith(("/tokenizer.slurm", "/train.slurm"))]
        for job in jobs:
            for spooled in (False, True):
                with self.subTest(job=job, spooled=spooled):
                    records = self.execute(job, extra={"FIXTURE_EXIT": "23"}, spooled=spooled, code=23)
                    self.assertEqual(len(records), 1)
        self.activate.write_text('return 17\n')
        self.assertEqual(self.execute(jobs[0], code=17), [])

    def test_wrong_working_directory_fails(self):
        self.assertEqual(self.execute(next(iter(EXPECTED)),cwd=self.temp,code=1),[])

if __name__ == '__main__':
    unittest.main()
