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

On a fresh clone, run these commands from the repository root. If you already have a `.env`, keep it and start with `source .env`.

```bash
cp .env.example .env
source .env
export PYTHONDONTWRITEBYTECODE=1
uv sync --locked
source .venv/bin/activate
```

In a new shell, reload `.env`, set `PYTHONDONTWRITEBYTECODE=1`, and activate `.venv`.

Libraries use their default cache locations. Set `HF_HOME`, `HF_DATASETS_CACHE`, or `UV_CACHE_DIR` in your local `.env` to override them. LRZ users can load their workspace cache settings there.

## GPT-2 data

From the repository root with the environment active, download the pinned English 10M corpus.

```bash
cd models/gpt2/baseline
python -B -m src.corpora.prepare_babylm --dataset english-10m
```

By default, the corpus goes to `corpora/babylm-2026/English/10M/` under the repository root. To share data across checkouts, set `CORPUS_ROOT` in each checkout’s `.env` to the absolute path of the directory containing `babylm-2026/`. Git ignores `.env` and the root `corpora/` directory. [Other corpora and access requirements](models/gpt2/baseline/README.md#prepare-raw-corpora)

## Running and submission

For English 10M, stay in the Baseline directory. Clean the corpus, train the tokenizer, and pretrain GPT-2 using `configs/English/10M.yaml` for all three steps. [Baseline commands](models/gpt2/baseline/README.md#clean-and-tokenize)

For Slurm, submit from the repository root and create the job’s log directory before submission. The scripts load the root `.env` and `.venv`. Their resource directives target LRZ. [Baseline Slurm guide](models/gpt2/baseline/README.md#run-on-lrz) · [Ablation Slurm guide](models/gpt2/ablation/README.md#run-on-lrz)

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
