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
cp .env.example .env
source .env
export PYTHONDONTWRITEBYTECODE=1
uv sync --locked
source .venv/bin/activate
```

In a new shell, reload `.env`, set `PYTHONDONTWRITEBYTECODE=1`, and activate `.venv`.

The example `.env` defaults to `.cache/` for Hugging Face and Baseline tokens
and `.cache/uv/` for uv. Git ignores `.cache/`. To share caches, set
`XDG_CACHE_HOME` to a common absolute path. Override individual caches with
`HF_HOME`, `HF_DATASETS_CACHE` or `UV_CACHE_DIR`. On LRZ, keep the local `.env`
that loads the workspace cache settings.

## GPT-2 data

From the repository root with the environment active, download the pinned English 10M corpus.

```bash
cd models/gpt2/baseline
uv run --active --no-sync python -B -m src.corpora.prepare_babylm --dataset english-10m
```

English 10M lands in `corpora/babylm-2026/English/10M/` by default. To share
raw corpora, set `CORPUS_ROOT` in each local `.env` to the same absolute path
containing `babylm-2026/`. Git ignores `.env` and the default `corpora/`.
[Corpus options and access requirements](models/gpt2/baseline/README.md#prepare-raw-corpora)

## Running and submission

For English 10M, stay in `models/gpt2/baseline/`. Follow [Clean and tokenize](models/gpt2/baseline/README.md#clean-and-tokenize), then [Train a model](models/gpt2/baseline/README.md#train-a-model). All three commands use `configs/English/10M.yaml`.

Create the job log directory and submit from the repository root. Slurm scripts target LRZ and load the root `.env` and `.venv`. [Baseline Slurm guide](models/gpt2/baseline/README.md#run-on-lrz) · [Ablation Slurm guide](models/gpt2/ablation/README.md#run-on-lrz)

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
