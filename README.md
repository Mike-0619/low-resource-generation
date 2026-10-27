# Low-resource generation

GPT-2 baselines for English 10M/100M and German 10M/108M BabyLM corpora.
See [experiment settings and data preparation](model/gpt2/baseline/README.md).
This snapshot includes training code and the parent project's pinned environment.
Corpora, tokenizers, model weights, logs and results are not included.

## Setup

Requires Linux x86_64, CPython 3.14 and uv on PATH. The locked PyTorch build uses
CUDA 13.2; training requires a compatible driver and an allocated GPU.

The module retains the parent project's workspace layout:

```text
<workspace>/
|-- config/cache.py
|-- config/cache.sh
|-- .cache/
|-- Corpora/babylm-2026/
`-- Projects/low-resource-generation/
    |-- pyproject.toml
    |-- uv.lock
    |-- .env
    |-- .venv/
    `-- model/gpt2/baseline/
```

Clone this repository into `<workspace>/Projects/low-resource-generation`.
Before setup, provide `<workspace>/config/cache.py` and
`<workspace>/config/cache.sh` through your workspace configuration. These external
dependencies are not included in this repository. Python entrypoints require
`cache.py`, which calls `cache.sh`; the environment template also sources
`cache.sh`. Both helpers must implement the workspace cache policy described below.
A fresh clone alone is therefore insufficient to run the code.

Once these files are available, run from the repository root:

```bash
source ../../config/cache.sh low-resource-generation
uv sync --locked
test -f .env || cp .env.example .env
# Set CORPUS_ROOT in .env to your external corpus directory.
set +x
source .env
source .venv/bin/activate
```

All disk caches must stay under the physical workspace's `.cache/`. Existing
workspace cache helpers remain authoritative. Overrides outside that boundary,
including symlink escapes, are rejected; `WORKSPACE_ROOT` cannot move the cache
boundary. Python bytecode is disabled by default. The environment template has
optional API settings inherited from the parent project; baseline training does
not require an API key.

Raw corpora and provenance records must be supplied separately as described in
the [baseline guide](model/gpt2/baseline/README.md#prepare-the-raw-data).
The German subset generator is external and is not part of this snapshot.

## Running and submission

Run scripts from the repository root. Each script loads `.env` and the shared
`.venv`, then enters `model/gpt2/baseline`. Create log directories first:

```bash
mkdir -p model/gpt2/baseline/logs/preparation
for experiment in English/10M English/100M German/10M German/108M; do
  mkdir -p "model/gpt2/baseline/logs/$experiment/tokenizer" \
           "model/gpt2/baseline/logs/$experiment/train"
done
```

Run preparation, tokenizer training and model training in that order. To submit
preparation on LRZ, use `sbatch model/gpt2/baseline/scripts/prepare_all.slurm`.
Then use the corresponding `tokenizer.slurm` and `train.slurm` under
`model/gpt2/baseline/scripts/<language>/<size>/`; wait for each stage to finish.
These commands submit real jobs. On already allocated resources, invoke the
same scripts with `bash`. Adapt literal partition/QOS directives for other clusters.
For a scheduler-only check use `sbatch --test-only path/to/job.slurm`.
Existing artifacts are protected from overwrite; training resume is unsupported.

## Offline tests

After loading the environment above:

```bash
cd model/gpt2/baseline
CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 OMP_NUM_THREADS=1 RAYON_NUM_THREADS=1 \
  uv run --active --locked --no-sync python -B -m unittest discover -s tests
```

Tests use temporary fixtures and small CPU models. They do not submit jobs or
run full training. From the same directory, run
`uv run --active --locked --no-sync python -B -m src.clean_dataset --help`
for CLI help; substitute `src.train_tokenizer` or `src.training` for the other stages.

When publishing trained weights separately, include the exact tokenizer used
by that model. Re-training a tokenizer is appropriate for a fresh experiment,
not a substitute for the tokenizer paired with existing weights.
