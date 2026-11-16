# LAMBADA Evaluation

Next-word evaluation for local language models and API backends on English and
German LAMBADA. Complete the root [setup](../../README.md#setup) first.

## Data

Dataset versions and language-specific cleanup rules are defined in
[data.py](src/data.py). Use `--language en` for English or `--language de` for German.
Online loading and offline fallback use the same configured Hugging Face dataset cache.

## Usage

Run from the repository root with the environment active. Use `--limit N`
for a subset or `--all` for the full dataset.

### Local model

On an allocated GPU, evaluate the existing Baseline English 10M export on 20 samples.

```bash
uv run --active --no-sync python -B -m tasks.lambada.src.greedy \
  --backend local \
  --model models/gpt2/baseline/results/English/10M/seed_42/final \
  --language en --context_limit 512 --dtype float32 --bos always \
  --limit 20 --output_dir tasks/lambada/results/local-example
```

### API model

Set `OPENAI_API_KEY` and `OPENAI_BASE_URL` using the root
[API configuration](../../README.md#api-configuration). API requests incur provider charges.

```bash
uv run --active --no-sync python -B -m tasks.lambada.src.greedy \
  --backend api --model openai/gpt-5.6-luna \
  --language en --reasoning_effort none \
  --limit 20 --output_dir tasks/lambada/results/api-example
```

## Run on LRZ

The English GPT-2 10M job evaluates the full split using the Baseline export.

```bash
mkdir -p tasks/lambada/logs/gpt2/English/10M/seed_42/full
sbatch tasks/lambada/scripts/gpt2/English/10M.slurm
```

Create the matching log directory for the other [Slurm jobs](scripts/).
Local model jobs accept H100 and A100. API jobs use CPU. [Slurm submission guide](../../README.md#running-and-submission)

## Outputs

Each run writes the following files to `--output_dir`. The LRZ scripts put this directory under `tasks/lambada/results/`.

| File | Contents |
| --- | --- |
| `metrics.json` | Accuracy and run settings |
| `samples.json`, `samples.jsonl` | Individual predictions |
| `human_eval.csv` | Predictions for manual review |

Accuracy compares the first generated word with the reference, preserving case
and treating `ß/ss` and `ẞ/SS` as equivalent after configured edge-punctuation
cleanup. Empty, invalid and unfinished answers count as incorrect. All selected
samples are included in the denominator. See [scoring.py](src/scoring.py).

Use a fresh output directory for each run. Nonempty directories require explicit
overwrite; human annotations prevent overwrite.

## Tests

Load the [offline-test environment](../../README.md#offline-tests) before running.

```bash
PYTHONPATH=. uv run --active --no-sync python -B -m unittest discover -s tasks/lambada/tests
```
