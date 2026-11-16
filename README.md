# Low-resource generation

Research code for low-resource language generation, including GPT-2 pretraining,
vocabulary-size experiments, paraphrase generation, persona-based dialogue and
English/German LAMBADA evaluation.

## Components

- [GPT-2 baseline training](models/gpt2/baseline/README.md)
- [GPT-2 vocabulary ablation](models/gpt2/ablation/README.md)
- [Paraphrase generation](tasks/paraphrase/README.md)
- [PersonaChat](tasks/personachat/README.md)
- [LAMBADA evaluation](tasks/lambada/README.md)

## Setup

All tasks use the root uv environment on Linux x86_64 with Python 3.14 and
PyTorch CUDA 13.2. The optional Paraphrase judge uses the same environment.
See the [judge setup](tasks/paraphrase/README.md#optional-local-judge).

From the project root, create and load a local configuration, then install
the locked dependencies:

```bash
test -f .env || cp .env.example .env
source .env
export PYTHONDONTWRITEBYTECODE=1
uv sync --locked
source .venv/bin/activate
```

In each new shell, load `.env` from the project root, set `PYTHONDONTWRITEBYTECODE=1`
and activate the environment before running commands. Git ignores the local
`.env` file, which can hold machine-specific settings and credentials.

Caches use library defaults. To change them, set standard variables such as
`HF_HOME`, `HF_DATASETS_CACHE` and `UV_CACHE_DIR` before starting a program.
On LRZ, keep workspace cache initialization in the local `.env` and load it before
running uv or Python. No external workspace helpers are required by the public code.

## GPT-2 data

The default `.env` puts raw corpora in the checkout's `corpora/` directory.
Git ignores this directory. To prepare English 10M, run from the project root:

```bash
(cd models/gpt2/baseline && uv run --active --no-sync python -m src.corpora.prepare_babylm --dataset english-10m)
```

This prepares `corpora/babylm-2026/English/10M/`. To share the same raw corpus
across multiple checkouts, set the same absolute `CORPUS_ROOT` in each local `.env`:

```bash
export CORPUS_ROOT="$HOME/Corpora"
```

Use the parent directory of `babylm-2026/`. The cluster's local `.env` can point to
an existing shared corpus directory. See the [corpus instructions](models/gpt2/baseline/README.md#prepare-raw-corpora)
for the other datasets.

## Running and submission

Follow each module's README for data preparation, training and evaluation.
The Slurm scripts target LRZ; adjust cluster resource directives for other systems.
Submit GPT-2, LAMBADA and Paraphrase scripts from the project root. They load `.env` and `.venv`.
Create the script's log directory before submission. For example:

```bash
mkdir -p models/gpt2/ablation/logs/English/prepare
sbatch --chdir="$PWD" models/gpt2/ablation/scripts/English/prepare.slurm
```

This prepares the English Ablation datasets. Wait for each stage to finish before starting
the next. Use allocated resources for full preparation and training. Existing
outputs are protected from overwrite.

## API configuration

Set `OPENAI_API_KEY` and `OPENAI_BASE_URL` in `.env` for API evaluation.
The example uses OpenRouter; change the endpoint and model name for other providers.
API requests incur provider charges.

## Offline tests

After loading the root environment, set:

```bash
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 RAYON_NUM_THREADS=1
```

Run the test command in each module's README. Full test suites use small CPU
models. Focused cache, configuration and script tests do not run models.

## License

The repository retains its [Apache 2.0 license](LICENSE). The imported LRZ source includes its original [MIT license and copyright notice](LICENSE-MIT).
