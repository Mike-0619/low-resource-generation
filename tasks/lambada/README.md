# LAMBADA Evaluation

Next-word evaluation for local language models and API backends on English and
German LAMBADA. Complete the root [setup](../../README.md#setup) first.

## Data

Dataset versions and language-specific cleanup rules are defined in
[data.py](src/data.py). Use `--language en` for English or `--language de` for German.
Online loading and offline fallback use the same configured Hugging Face dataset cache.

## Usage

Run from the repository root with the environment active. Evaluation uses the
full dataset by default. Use `--limit N` for debugging in a separate output
directory. Limited runs cannot replace existing full results; this is checked
before model loading or API requests.

### Local model

On an allocated GPU, evaluate the existing Baseline English 10M export on the full test split.

```bash
uv run --active --no-sync python -B -m tasks.lambada.src.greedy \
  --backend local \
  --model models/gpt2/baseline/results/English/10M/seed_42/final \
  --language en --context_limit 512 --dtype float32 --bos always \
  --output_dir tasks/lambada/results/local-example
```

### API model

Set `OPENAI_API_KEY` and `OPENAI_BASE_URL` using the root
[API configuration](../../README.md#api-configuration). API requests incur provider charges.

```bash
uv run --active --no-sync python -B -m tasks.lambada.src.greedy \
  --backend api --model openai/gpt-5.6-luna \
  --language en --reasoning_effort none \
  --output_dir tasks/lambada/results/api-example
```

## Run with Slurm

The English GPT-2 10M job evaluates the full split using the Baseline export.

```bash
mkdir -p tasks/lambada/logs/gpt2/English/10M
sbatch --partition=YOUR_GPU_PARTITION --gres=gpu:1 tasks/lambada/scripts/gpt2/English/10M.slurm
```

Create the matching log directory for the other [Slurm jobs](scripts/).
Local model jobs request one GPU. API jobs use CPU. [Slurm submission guide](../../README.md#running-and-submission)

### German LAMBADA with Llama 3 and GPT-5.6 Luna

From the repository root, submit the full evaluations on the 5,147 eligible Boldt
test examples.

```bash
mkdir -p tasks/lambada/logs/{llama3-8b,gpt5.6-luna}/German
sbatch --partition=YOUR_GPU_PARTITION --gres=gpu:1 tasks/lambada/scripts/llama3-8b/German.slurm
sbatch --partition=YOUR_CPU_PARTITION tasks/lambada/scripts/gpt5.6-luna/German.slurm
```

The jobs use the same model and generation settings as the corresponding
English jobs. Luna uses the configured API endpoint and incurs provider charges.

### Turkish LAMBADA entry points

Turkish LAMBADA data is pending. The evaluator recognizes `--language tr` and
reports the pending dataset before loading a model, calling the API, or writing
evaluation results.

| Model | Slurm script | Result directory under `results/` |
| --- | --- | --- |
| GPT-2 Turkish 10M | [10M.slurm](scripts/gpt2/Turkish/10M.slurm) | `gpt2/Turkish/10M/` |
| GPT-2 Turkish 100M | [100M.slurm](scripts/gpt2/Turkish/100M.slurm) | `gpt2/Turkish/100M/` |
| Llama 3 8B | [Turkish.slurm](scripts/llama3-8b/Turkish.slurm) | `llama3-8b/Turkish/` |
| GPT-5.6 Luna | [Turkish.slurm](scripts/gpt5.6-luna/Turkish.slurm) | `gpt5.6-luna/Turkish/` |

Once the dataset is configured in [data.py](src/data.py), create the matching
directories under `tasks/lambada/logs/` before submission. Each script evaluates
the full dataset and takes no positional arguments.

## Directory layout

Model directories use `gpt2`, `gpt5.6-luna`, and `llama3-8b`. Language
directories use `English`, `German`, and `Turkish`.

| Artifact | GPT-2 | Llama 3 and Luna |
| --- | --- | --- |
| Script under `scripts/` | `gpt2/<Language>/<Size>.slurm` | `<Model>/<Language>.slurm` |
| Results under `results/` | `gpt2/<Language>/<Size>/` | `<Model>/<Language>/` |
| Logs under `logs/` | `gpt2/<Language>/<Size>/<job_id>.out` and `.err` | `<Model>/<Language>/<job_id>.out` and `.err` |

The local score summary is `results.md`. The answer workbook is
`results/human_annotation/answer_key.xlsx`.

## Outputs

Each run writes the following files to `--output_dir`. The Slurm scripts put this directory under `tasks/lambada/results/`.

| File | Contents |
| --- | --- |
| `metrics.json` | Accuracy and run settings |
| `samples.json` | Individual predictions |

Accuracy compares the first generated word with the reference, preserving case
and treating `ß/ss` and `ẞ/SS` as equivalent after configured edge-punctuation
cleanup. Empty, invalid and unfinished answers count as incorrect. All selected
samples are included in the denominator. See [scoring.py](src/scoring.py).

Rerunning an evaluation automatically replaces its previous results after both
new files have been written successfully. Model, data, API, or write failures
preserve the previous results. Other files in the output directory are retained.
If publication succeeds but deleting the old backup fails, the run still succeeds
and prints a cleanup warning. The backup is retained and cleanup is retried before
the next evaluation.

Use separate output directories for concurrent evaluations. Interrupted directory
publication is recovered before the next evaluation.

## Tests

Load the [offline-test environment](../../README.md#offline-tests) before running.

```bash
PYTHONPATH=. uv run --active --no-sync python -B -m unittest discover -s tasks/lambada/tests
```
