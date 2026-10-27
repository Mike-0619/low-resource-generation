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
EXPECTED = {'model/gpt2/baseline/scripts/English/100M/tokenizer.slurm': {'commands': [['python',
                                                                   '-m',
                                                                   'src.train_tokenizer',
                                                                   '--config',
                                                                   'configs/English/100M.yaml']],
                                                     'exports': {'CUDA_VISIBLE_DEVICES': '',
                                                                 'HF_HUB_OFFLINE': '1',
                                                                 'HF_DATASETS_OFFLINE': '1',
                                                                 'OMP_NUM_THREADS': '4',
                                                                 'RAYON_NUM_THREADS': '4'}},
 'model/gpt2/baseline/scripts/English/100M/train.slurm': {'commands': [['python',
                                                               '-m',
                                                               'src.training',
                                                               '--config',
                                                               'configs/English/100M.yaml']],
                                                 'exports': {'HF_HUB_OFFLINE': '1',
                                                             'HF_DATASETS_OFFLINE': '1',
                                                             'OMP_NUM_THREADS': '8'}},
 'model/gpt2/baseline/scripts/English/10M/tokenizer.slurm': {'commands': [['python',
                                                                  '-m',
                                                                  'src.train_tokenizer',
                                                                  '--config',
                                                                  'configs/English/10M.yaml']],
                                                    'exports': {'CUDA_VISIBLE_DEVICES': '',
                                                                'HF_HUB_OFFLINE': '1',
                                                                'HF_DATASETS_OFFLINE': '1',
                                                                'OMP_NUM_THREADS': '4',
                                                                'RAYON_NUM_THREADS': '4'}},
 'model/gpt2/baseline/scripts/English/10M/train.slurm': {'commands': [['python',
                                                              '-m',
                                                              'src.training',
                                                              '--config',
                                                              'configs/English/10M.yaml']],
                                                'exports': {'HF_HUB_OFFLINE': '1',
                                                            'HF_DATASETS_OFFLINE': '1',
                                                            'OMP_NUM_THREADS': '8'}},
 'model/gpt2/baseline/scripts/German/108M/tokenizer.slurm': {'commands': [['python',
                                                                  '-m',
                                                                  'src.train_tokenizer',
                                                                  '--config',
                                                                  'configs/German/108M.yaml']],
                                                    'exports': {'CUDA_VISIBLE_DEVICES': '',
                                                                'HF_HUB_OFFLINE': '1',
                                                                'HF_DATASETS_OFFLINE': '1',
                                                                'OMP_NUM_THREADS': '4',
                                                                'RAYON_NUM_THREADS': '4'}},
 'model/gpt2/baseline/scripts/German/108M/train.slurm': {'commands': [['python',
                                                              '-m',
                                                              'src.training',
                                                              '--config',
                                                              'configs/German/108M.yaml']],
                                                'exports': {'HF_HUB_OFFLINE': '1',
                                                            'HF_DATASETS_OFFLINE': '1',
                                                            'OMP_NUM_THREADS': '8'}},
 'model/gpt2/baseline/scripts/German/10M/tokenizer.slurm': {'commands': [['python',
                                                                 '-m',
                                                                 'src.train_tokenizer',
                                                                 '--config',
                                                                 'configs/German/10M.yaml']],
                                                   'exports': {'CUDA_VISIBLE_DEVICES': '',
                                                               'HF_HUB_OFFLINE': '1',
                                                               'HF_DATASETS_OFFLINE': '1',
                                                               'OMP_NUM_THREADS': '4',
                                                               'RAYON_NUM_THREADS': '4'}},
 'model/gpt2/baseline/scripts/German/10M/train.slurm': {'commands': [['python',
                                                             '-m',
                                                             'src.training',
                                                             '--config',
                                                             'configs/German/10M.yaml']],
                                               'exports': {'HF_HUB_OFFLINE': '1',
                                                           'HF_DATASETS_OFFLINE': '1',
                                                           'OMP_NUM_THREADS': '8'}},
 'model/gpt2/baseline/scripts/prepare_all.slurm': {'commands': [['python',
                                                        '-m',
                                                        'src.clean_dataset',
                                                        '--config',
                                                        'configs/English/10M.yaml'],
                                                       ['python',
                                                        '-m',
                                                        'src.clean_dataset',
                                                        '--config',
                                                        'configs/English/100M.yaml'],
                                                       ['python',
                                                        '-m',
                                                        'src.clean_dataset',
                                                        '--config',
                                                        'configs/German/10M.yaml'],
                                                       ['python',
                                                        '-m',
                                                        'src.clean_dataset',
                                                        '--config',
                                                        'configs/German/108M.yaml']],
                                          'exports': {'CUDA_VISIBLE_DEVICES': '',
                                                      'HF_HUB_OFFLINE': '1',
                                                      'HF_DATASETS_OFFLINE': '1',
                                                      'OMP_NUM_THREADS': '4'}}}
MODEL = True

class ScriptTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='independent scripts ', dir=os.environ['XDG_CACHE_HOME'])
        self.addCleanup(self.temporary.cleanup)
        self.temp = Path(self.temporary.name)
        self.workspace = self.temp / 'workspace with spaces'
        self.root = self.workspace / 'Projects' / 'low-resource-generation'
        (self.workspace / 'config').mkdir(parents=True)
        shutil.copyfile(ROOT.parents[1] / 'config/cache.sh', self.workspace / 'config/cache.sh')
        for directory in ['.venv/bin', 'fixture-bin', 'model/gpt2/baseline']:
            (self.root / directory).mkdir(parents=True, exist_ok=True)
        for relative in EXPECTED:
            target = self.root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / relative, target)
        self.dotenv = self.root / '.env'
        self.dotenv.write_text((ROOT / '.env.example').read_text().replace(
            '/path/to/workspace', str(self.workspace)).replace(
            '${OPENAI_API_KEY-}', '${OPENAI_API_KEY-DOTENV_FIXTURE_MARKER}'))
        self.activate = self.root / '.venv/bin/activate'
        self.activate.write_text('export VIRTUAL_ENV="$PWD/.venv"\n')
        self.uv = self.root / 'fixture-bin/uv'
        self.uv.write_text(f'#!{sys.executable}\n' + '''import json, os, sys
print(json.dumps({'args':sys.argv[1:], 'cwd':os.getcwd(),
 'venv':os.environ.get('VIRTUAL_ENV'),
 'cache':os.environ.get('HF_HOME'), 'datasets':os.environ.get('HF_DATASETS_CACHE'),
 'uv_cache':os.environ.get('UV_CACHE_DIR'), 'bytecode':os.environ.get('PYTHONDONTWRITEBYTECODE'),
 'api':os.environ.get('OPENAI_API_KEY') in ['DOTENV_FIXTURE_MARKER','EXPORTED_FIXTURE_MARKER'],
 'api_override':os.environ.get('OPENAI_API_KEY')=='EXPORTED_FIXTURE_MARKER',
 'task_env':{k:os.environ[k] for k in ['HF_HUB_OFFLINE','HF_DATASETS_OFFLINE','OMP_NUM_THREADS','RAYON_NUM_THREADS','CUDA_VISIBLE_DEVICES'] if k in os.environ}}))
sys.exit(int(os.environ.get('FIXTURE_EXIT','0')))
''')
        self.uv.chmod(0o700)
        self.env = {'HOME':str(self.root), 'PATH':str(self.uv.parent)+':'+os.defpath}

    def execute(self, relative, extra=None, spooled=False, code=0, trace=False, cwd=None):
        script = self.root / relative
        if spooled:
            copied = self.temp / 'slurm_script'
            shutil.copyfile(script, copied)
            script = copied
        result = subprocess.run(['bash'] + (['-x'] if trace else []) + [str(script)],
            cwd=cwd or self.root, env={**self.env, **(extra or {})}, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, code, result.stderr)
        for secret in ['DOTENV_FIXTURE_MARKER', 'EXPORTED_FIXTURE_MARKER']:
            self.assertNotIn(secret, result.stdout + result.stderr)
        return [json.loads(line) for line in result.stdout.splitlines() if line.startswith('{')]

    def test_slurm_partition_and_qos_match_task(self):
        for relative, expected in EXPECTED.items():
            command = expected['commands'][0]
            gpu = 'src.training' in command or ('--backend' in command and command[command.index('--backend') + 1] != 'gpt')
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
                                     [['run','--active','--locked','--no-sync'] + command for command in expected['commands']])
                    for record in records:
                        self.assertEqual(record['cwd'],str(self.root / 'model/gpt2/baseline' if MODEL else self.root))
                        self.assertEqual(record['venv'],str(self.root / '.venv'))
                        self.assertEqual(record['task_env'],expected['exports'])
                        self.assertTrue(record['api'])

    def test_environment_defaults_and_exported_overrides(self):
        job = next(iter(EXPECTED))
        record = self.execute(job)[0]
        self.assertEqual(record['cache'],str(self.workspace / '.cache/huggingface'))
        self.assertEqual(record['uv_cache'],str(self.workspace / '.cache/uv'))
        self.assertEqual(record['datasets'],record['cache']+'/datasets')
        self.assertEqual(record['bytecode'],'1')
        record = self.execute(job,extra={'HF_HOME':str(self.workspace / '.cache/custom/hf'),
            'HF_DATASETS_CACHE':str(self.workspace / '.cache/custom/datasets'),'UV_CACHE_DIR':str(self.workspace / '.cache/custom/uv'),
            'OPENAI_API_KEY':'EXPORTED_FIXTURE_MARKER','PYTHONDONTWRITEBYTECODE':'0'},trace=True)[0]
        self.assertEqual(record['cache'],str(self.workspace / '.cache/custom/hf'))
        self.assertEqual(record['datasets'],str(self.workspace / '.cache/custom/datasets'))
        self.assertEqual(record['uv_cache'],str(self.workspace / '.cache/custom/uv'))
        self.assertEqual(record['bytecode'],'0')
        self.assertTrue(record['api_override'])
        self.assertEqual(self.execute(job,extra={'HF_HOME':''})[0]['cache'],str(self.workspace / '.cache/huggingface'))

    def test_outside_cache_rejected_before_uv(self):
        self.assertEqual(self.execute(next(iter(EXPECTED)), extra={'UV_CACHE_DIR':'/outside/uv'}, code=1), [])

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
