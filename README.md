# Low-Resource Generation

Train GPT-2 on BabyLM corpora, study how vocabulary size affects training, and evaluate or adapt language models on downstream tasks.

**First run?** Start with English 10M. [Setup](#setup) → [Download the corpus](#gpt-2-data) → [Train the baseline](#running-and-submission)

## Components

| Module | Use it to |
| --- | --- |
| [GPT-2 Baseline](models/gpt2/baseline/README.md) | Pretrain GPT-2 on English, German, or Turkish corpora. |
| [GPT-2 Ablation](models/gpt2/ablation/README.md) | Compare six vocabulary sizes under two sampling protocols. |
| [LAMBADA](tasks/lambada/README.md) | Evaluate next-word prediction in English and German. |
| [Paraphrase](tasks/paraphrase/README.md) | Fine-tune GPT-2 for paraphrasing and compare it with Llama 3.2. |
| [PersonaChat](tasks/personachat/README.md) | Fine-tune and evaluate models for persona-based dialogue. |

## Setup

Use `uv` on Linux x86‑64. The lockfile targets Python 3.14 and PyTorch with CUDA 13.2. Training requires a compatible NVIDIA GPU.

Run from the repository root. If `.env` already exists, start with `source .env`.

```bash
if [ ! -e .env ]; then
  cp .env.example .env
fi
source .env
uv sync --locked
source .venv/bin/activate
```

In a new Bash shell, reload `.env` with `source .env` and activate `.venv`.
Merge future template changes into an existing `.env`; do not replace its credentials.

The example is a standalone Bash configuration. `XDG_CACHE_HOME` defaults to
`$HOME/.cache`; Hugging Face uses its `huggingface/` subdirectory and uv uses
`uv/`. Set `HF_HOME`, `HF_DATASETS_CACHE` or `UV_CACHE_DIR` to override them.
Nonempty exported values take precedence; empty values use defaults.
Existing local `.env` settings are preserved, including any shared cache setup.
Python bytecode is disabled by default. Git ignores `.env` and optional machine
notes in `README.local.md`.

## GPT-2 data

From the repository root with the environment active, download the pinned English 10M corpus.

```bash
cd models/gpt2/baseline
uv run --active --no-sync python -B -m src.corpora.prepare_babylm --dataset english-10m
```

English 10M lands in `$CORPUS_ROOT/babylm-2026/English/10M/`. The example
defaults `CORPUS_ROOT` to `$HOME/Corpora/`. Override it with the directory
containing `babylm-2026/` when raw inputs are stored elsewhere; raw corpora are
stored separately from caches.
[Corpus options and access requirements](models/gpt2/baseline/README.md#prepare-raw-corpora)

## Running and submission

For English 10M, stay in `models/gpt2/baseline/`. Follow [Clean and tokenize](models/gpt2/baseline/README.md#clean-and-tokenize), then [Train a model](models/gpt2/baseline/README.md#train-a-model). All three commands use `configs/English/10M.yaml`.

Create the matching log directory and submit from the repository root.
Slurm scripts load the root `.env` and `.venv`. They specify CPU, memory, time
and GPU count; supply your site's partition and optional QoS on the command line.
Replace `YOUR_CPU_PARTITION` and `YOUR_GPU_PARTITION` in the examples below.

```bash
sbatch --partition=YOUR_CPU_PARTITION models/gpt2/baseline/scripts/English/10M/prepare.slurm
sbatch --partition=YOUR_GPU_PARTITION --gres=gpu:1 models/gpt2/baseline/scripts/English/10M/train.slurm
```

Add `--qos=YOUR_QOS` when required. If the site requires a GPU type, override
`--gres`, for example `--gres=gpu:TYPE:1`. Command-line options override the
script's defaults; shell variables are not expanded inside `#SBATCH` directives.
See the [Slurm documentation](https://slurm.schedmd.com/sbatch.html).
Keep machine-specific commands in the ignored `README.local.md`.
[Baseline Slurm guide](models/gpt2/baseline/README.md#run-with-slurm) ·
[Ablation Slurm guide](models/gpt2/ablation/README.md#run-with-slurm)

## API configuration

For API evaluation, set `OPENAI_API_KEY` and `OPENAI_BASE_URL` in `.env`. Choose the model ID for your provider. API usage may incur charges. [LAMBADA API example](tasks/lambada/README.md#api-model)

## Offline tests

With the environment active, set the offline and CPU options.

```bash
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 RAYON_NUM_THREADS=1
```

Run the test command in the relevant module’s README. The offline suites use small fixtures and CPU models.

## License

This project is released under the [MIT License](LICENSE).
