"""Independent job scripts: root-relative execution and environment regressions."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]
EXPECTED = {'tasks/lambada/scripts/gpt2/English/100M.slurm': {'commands': [['python', '-u', '-m', 'tasks.lambada.src.greedy', '--backend', 'local', '--language', 'en', '--model', 'models/gpt2/baseline/results/English/100M/seed_42/final', '--context_limit', '512', '--dtype', 'float32', '--bos', 'always', '--all', '--batch_size', '8', '--device', 'cuda', '--output_dir', 'tasks/lambada/results/gpt2/English/100M/seed_42/full']], 'exports': {}}, 'tasks/lambada/scripts/gpt2/English/10M.slurm': {'commands': [['python', '-u', '-m', 'tasks.lambada.src.greedy', '--backend', 'local', '--language', 'en', '--model', 'models/gpt2/baseline/results/English/10M/seed_42/final', '--context_limit', '512', '--dtype', 'float32', '--bos', 'always', '--all', '--batch_size', '8', '--device', 'cuda', '--output_dir', 'tasks/lambada/results/gpt2/English/10M/seed_42/full']], 'exports': {}}, 'tasks/lambada/scripts/gpt2/German/108M.slurm': {'commands': [['python', '-u', '-m', 'tasks.lambada.src.greedy', '--backend', 'local', '--language', 'de', '--model', 'models/gpt2/baseline/results/German/108M/seed_42/final', '--context_limit', '512', '--dtype', 'float32', '--bos', 'always', '--all', '--batch_size', '8', '--device', 'cuda', '--output_dir', 'tasks/lambada/results/gpt2/German/108M/seed_42/full']], 'exports': {}}, 'tasks/lambada/scripts/gpt2/German/10M.slurm': {'commands': [['python', '-u', '-m', 'tasks.lambada.src.greedy', '--backend', 'local', '--language', 'de', '--model', 'models/gpt2/baseline/results/German/10M/seed_42/final', '--context_limit', '512', '--dtype', 'float32', '--bos', 'always', '--all', '--batch_size', '8', '--device', 'cuda', '--output_dir', 'tasks/lambada/results/gpt2/German/10M/seed_42/full']], 'exports': {}}, 'tasks/lambada/scripts/gpt5.6/luna.slurm': {'commands': [['python', '-u', '-m', 'tasks.lambada.src.greedy', '--backend', 'api', '--language', 'en', '--model', 'openai/gpt-5.6-luna', '--reasoning_effort', 'none', '--all', '--concurrency', '4', '--output_dir', 'tasks/lambada/results/gpt5.6/luna/full']], 'exports': {}}, 'tasks/lambada/scripts/gpt5.6/sol.slurm': {'commands': [['python', '-u', '-m', 'tasks.lambada.src.greedy', '--backend', 'api', '--language', 'en', '--model', 'openai/gpt-5.6-sol', '--reasoning_effort', 'none', '--all', '--concurrency', '4', '--output_dir', 'tasks/lambada/results/gpt5.6/sol/full']], 'exports': {}}, 'tasks/lambada/scripts/llama3/llama3.slurm': {'commands': [['python', '-u', '-m', 'tasks.lambada.src.greedy', '--backend', 'local', '--language', 'en', '--model', 'models/llama3/Meta-Llama-3-8B', '--context_limit', '8192', '--dtype', 'bfloat16', '--bos', 'always', '--all', '--batch_size', '8', '--device', 'cuda', '--output_dir', 'tasks/lambada/results/llama3/full']], 'exports': {}}}
MODEL = False

class ScriptTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='independent scripts ', dir=os.environ.get('XDG_CACHE_HOME'))
        self.addCleanup(self.temporary.cleanup)
        self.temp = Path(self.temporary.name)
        self.root = self.temp / 'project with spaces'
        for directory in ['.venv/bin', 'fixture-bin', 'models/gpt2/baseline']:
            (self.root / directory).mkdir(parents=True, exist_ok=True)
        for relative in EXPECTED:
            target = self.root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / relative, target)
        self.dotenv = self.root / '.env'
        self.dotenv.write_text((ROOT / '.env.example').read_text().replace(
            '${OPENAI_API_KEY:-}', '${OPENAI_API_KEY:-DOTENV_FIXTURE_MARKER}'))
        self.activate = self.root / '.venv/bin/activate'
        self.activate.write_text('export VIRTUAL_ENV="$PWD/.venv"\n')
        self.uv = self.root / 'fixture-bin/uv'
        self.uv.write_text(f'#!{sys.executable}\n' + '''import json, os, sys
print(json.dumps({'args':sys.argv[1:], 'cwd':os.getcwd(),
 'venv':os.environ.get('VIRTUAL_ENV'), 'corpus':os.environ.get('CORPUS_ROOT'),
 'cache':os.environ.get('HF_HOME'), 'datasets':os.environ.get('HF_DATASETS_CACHE'),
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

    def test_help_without_workspace_configuration(self):
        shutil.copytree(ROOT / 'tasks/lambada/src', self.root / 'tasks/lambada/src',
                        ignore=shutil.ignore_patterns('__pycache__'))
        env = {key: value for key, value in os.environ.items()
               if key not in {'WORKSPACE_ROOT', 'PYTHONPATH'}}
        result = subprocess.run(
            [sys.executable, '-B', '-m', 'tasks.lambada.src.greedy', '--help'],
            cwd=self.root, env=env, capture_output=True, text=True, timeout=90,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--backend', result.stdout)

    def test_slurm_partition_and_qos_match_task(self):
        for relative, expected in EXPECTED.items():
            command = expected['commands'][0]
            gpu = 'src.training' in command or ('--backend' in command and command[command.index('--backend') + 1] == 'local')
            lines = (ROOT / relative).read_text().splitlines()
            partition = 'lrz-hgx-h100-94x4,lrz-hgx-a100-80x4,lrz-dgx-a100-80x8' if gpu else 'lrz-cpu'
            self.assertEqual([line for line in lines if line.startswith('#SBATCH --partition=')],
                             ['#SBATCH --partition=' + partition])
            self.assertEqual([line for line in lines if line.startswith('#SBATCH --qos=')],
                             ['#SBATCH --qos=' + ('gpu' if gpu else 'cpu')])
            self.assertEqual(any(line.startswith('#SBATCH --gres=') for line in lines), gpu)

    def test_all_jobs_direct_and_slurm_copies(self):
        for relative, expected in EXPECTED.items():
            for spooled in [False, True]:
                with self.subTest(script=relative, spooled=spooled):
                    records = self.execute(relative, spooled=spooled)
                    self.assertEqual([r['args'] for r in records],
                                     [['run','--active','--no-sync'] + command for command in expected['commands']])
                    for record in records:
                        self.assertEqual(record['cwd'],str(self.root / 'models/gpt2/baseline' if MODEL else self.root))
                        self.assertEqual(record['venv'],str(self.root / '.venv'))
                        self.assertEqual(record['task_env'],expected['exports'])
                        self.assertTrue(record['api'])

    def test_environment_defaults_and_exported_overrides(self):
        job = next(iter(EXPECTED))
        record = self.execute(job)[0]
        self.assertEqual(record['corpus'], str(self.root / 'corpora'))
        self.assertEqual(record['xdg_cache'], str(self.root / '.cache'))
        self.assertEqual(record['uv_cache'], str(self.root / '.cache/uv'))
        for key in ('cache', 'datasets', 'bytecode'):
            self.assertIsNone(record[key])
        shared_cache = str(self.temp / 'shared cache')
        record = self.execute(job, extra={'XDG_CACHE_HOME': shared_cache})[0]
        self.assertEqual(record['xdg_cache'], shared_cache)
        self.assertEqual(record['uv_cache'], str(Path(shared_cache) / 'uv'))
        overrides = {
            'XDG_CACHE_HOME': shared_cache,
            'HF_HOME': str(self.temp / 'custom/hf'),
            'HF_DATASETS_CACHE': str(self.temp / 'custom/datasets'),
            'UV_CACHE_DIR': str(self.temp / 'custom/uv'),
            'CORPUS_ROOT': str(self.temp / 'custom/corpora'),
            'OPENAI_API_KEY': 'EXPORTED_FIXTURE_MARKER',
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
        record = self.execute(job, extra={
            'CORPUS_ROOT': '', 'OPENAI_API_KEY': '', 'XDG_CACHE_HOME': '', 'UV_CACHE_DIR': '',
        })[0]
        self.assertEqual(record['corpus'], str(self.root / 'corpora'))
        self.assertEqual(record['xdg_cache'], str(self.root / '.cache'))
        self.assertEqual(record['uv_cache'], str(self.root / '.cache/uv'))
        self.assertTrue(record['api'])

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
        job = next(iter(EXPECTED))
        self.assertEqual(len(self.execute(job,extra={'FIXTURE_EXIT':'23'},code=23)),1)
        self.activate.write_text('return 17\n')
        self.assertEqual(self.execute(job,code=17),[])

    def test_wrong_working_directory_fails(self):
        self.assertEqual(self.execute(next(iter(EXPECTED)),cwd=self.temp,code=1),[])

if __name__ == '__main__':
    unittest.main()
