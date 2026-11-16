# Paraphrase Generation

Fine-tune GPT-2 on MRPC, compare with Llama 3.2 3B, and evaluate the
generated paraphrases with SacreBLEU, ParaPLUIE and an optional local Qwen judge.

## Setup

Complete the [project setup](../../README.md#setup) and export the English
GPT-2 models for the selected sizes. [Baseline training](../../models/gpt2/baseline/README.md#train-a-model)

## Running on LRZ

Run `sbatch` or `bash` from the repository root. Create the job log directories first.

```bash
mkdir -p tasks/paraphrase/logs/{gpt2,llama3,comparison,pipeline}
```

| Job | Slurm file under `scripts/` | YAML under `configs/` |
| --- | --- | --- |
| GPT-2 10M fine-tuning, BLEU and ParaPLUIE | [gpt2/gpt2_10m.slurm](scripts/gpt2/gpt2_10m.slurm) | [gpt2/gpt2_10m.yaml](configs/gpt2/gpt2_10m.yaml) |
| GPT-2 100M fine-tuning, BLEU and ParaPLUIE | [gpt2/gpt2_100m.slurm](scripts/gpt2/gpt2_100m.slurm) | [gpt2/gpt2_100m.yaml](configs/gpt2/gpt2_100m.yaml) |
| Llama inference and scoring for all three prompts | [llama3/llama3.slurm](scripts/llama3/llama3.slurm) | [llama3/llama3.yaml](configs/llama3/llama3.yaml) |
| Aggregate reports by sample ID (CPU) | [comparison/aggregate.slurm](scripts/comparison/aggregate.slurm) | [comparison/aggregate_with_100m.yaml](configs/comparison/aggregate_with_100m.yaml), [comparison/aggregate_without_100m.yaml](configs/comparison/aggregate_without_100m.yaml) |
| Score the aggregate with Qwen (GPU) | [comparison/judge.slurm](scripts/comparison/judge.slurm) | [comparison/judge_with_100m.yaml](configs/comparison/judge_with_100m.yaml), [comparison/judge_without_100m.yaml](configs/comparison/judge_without_100m.yaml) |
| Run the full experiment | [pipeline.slurm](scripts/pipeline.slurm) | Selects the configs above for each stage |

GPT-2 and Llama jobs support H100 or A100. The judge and full pipeline request H100.

Submit 10M and Llama for either comparison. Include 100M for the full comparison.

```bash
sbatch tasks/paraphrase/scripts/gpt2/gpt2_10m.slurm
sbatch tasks/paraphrase/scripts/llama3/llama3.slurm
sbatch tasks/paraphrase/scripts/gpt2/gpt2_100m.slurm  # full comparison only
```

After the model jobs succeed, submit the aggregate. Run the optional judge after aggregation succeeds.

```bash
sbatch tasks/paraphrase/scripts/comparison/aggregate.slurm
sbatch tasks/paraphrase/scripts/comparison/judge.slurm
```

Both scripts include 100M by default. To omit it, use these configs in the same order.

```bash
sbatch tasks/paraphrase/scripts/comparison/aggregate.slurm --config configs/comparison/aggregate_without_100m.yaml
sbatch tasks/paraphrase/scripts/comparison/judge.slurm --config configs/comparison/judge_without_100m.yaml
```

To run the complete experiment in one job, prepare the local judge first.

```bash
sbatch tasks/paraphrase/scripts/pipeline.slurm
```

Use `--without-100m` to omit 100M, `--train_only` to stop after aggregation,
or `--judge_only` to score an existing aggregate. Use `bash` instead of `sbatch`
for a local run with the required resources.

## Configurations and results

The GPT-2 configs use `models/gpt2/baseline/results/English/{10M,100M}/seed_42/final`
from the project root. To override the training input, set `PATH_10M_BASE` or
`PATH_100M_BASE` to an absolute checkpoint path.

| Artifact | Path relative to `tasks/paraphrase/` |
| --- | --- |
| GPT-2 weights and training records | `results/gpt2/{10m_seed_42,100m_seed_42}/final/` |
| GPT-2 BLEU and ParaPLUIE reports | `results/gpt2/{10m_seed_42,100m_seed_42}/` |
| Llama reports | `results/llama3/<prompt>/` |
| Combined reports and judge output | `results/comparison/{with_100m,without_100m}/` |
| Slurm job logs | `logs/<group>/%x-%j.out`, `logs/<group>/%x-%j.err` (experiment group, job name and ID) |
| Judge service logs | `logs/comparison/server_<job-or-local>_<port>.log` for standalone jobs, `logs/pipeline/` for pipeline jobs |

Use `--config FILE` for standalone jobs or copy `configs/` for the pipeline
and pass it with `--config-dir DIR`. Keep its `gpt2/`, `llama3/` and
`comparison/` subdirectories. Both options accept absolute paths. Relative config, model,
input and output paths resolve from `tasks/paraphrase/`.

Point scoring or aggregation configs to existing reports to reuse them. For a
new seed, update the run directory and dependent report paths. Generation,
scoring and aggregation need fresh output paths. The judge resumes an
existing output for the same comparison.

For a single scoring stage, load the root environment and enter `tasks/paraphrase/`.
This example reads the GPT-2 10M generation report specified in the YAML.

```bash
cd tasks/paraphrase
uv run --active --no-sync python -B -m src.training.pipeline \
  --mode parapluie --config configs/gpt2/gpt2_10m.yaml
```

The pipeline modes are `train`, `llama_benchmark`, `parapluie`, `database` and
`judge`. For Llama generation or scoring, pass `--prompt_type` to select a run.
The Llama config defaults to `few_shot` for single-stage runs.

BLEU and ParaPLUIE are computed separately. The ParaPLUIE report preserves
BLEU scores and sample IDs from the generation report. Aggregation checks
that sample IDs, source texts and references match across systems.

## Optional local judge

Set `CUDA_ROOT` to your CUDA 13.2 toolkit directory, with `bin/nvcc`, CUDA
headers, and cuBLAS available. Build the optional judge dependency from the
repository root with the project environment active.

```bash
export PATH="$CUDA_ROOT/bin:$PATH"
export CUDACXX="$CUDA_ROOT/bin/nvcc"
export CMAKE_ARGS="-DGGML_CUDA=ON -DCUDAToolkit_ROOT=$CUDA_ROOT -DCMAKE_CUDA_ARCHITECTURES=80;90"
uv sync --locked --group judge --no-binary-package llama-cpp-python
```

Download the default judge's GGUF shards from `tasks/paraphrase/`.

```bash
cd tasks/paraphrase
uv run --active --no-sync python -B -m src.models.download
```

The shards go under `tasks/paraphrase/models/UD-IQ4_NL/`. Each judge YAML
points to the first shard. For Slurm, return to the repository root before
submitting. The launcher loads CUDA libraries from the root `.venv`.

## Source layout

- `src/data/` loads MRPC and QQP and defines prompts and training examples.
- `src/models/` loads and downloads models.
- `src/training/` contains GPT-2 fine-tuning, generation and the job entry points.
- `src/evaluation/` contains the Llama comparison, metrics, reports and judge.

## Offline tests

Load the [root offline-test environment](../../README.md#offline-tests), then
run from the project root:

```bash
uv run --active --no-sync python -B -m unittest discover -s tasks/paraphrase/tests
```
