# Paraphrase generation

Fine-tune GPT-2 on MRPC, compare with Llama 3.2 3B, and evaluate the
generated paraphrases with SacreBLEU, ParaPLUIE and an optional local Qwen judge.

## Setup

Follow the [root uv setup](../../README.md#setup). Launchers load the project's
`.env` and root `.venv`. Run both `sbatch` and `bash` from the project root.
Create the job log directories before submitting:

```bash
mkdir -p tasks/paraphrase/logs/{gpt2,llama3,comparison,pipeline}
```

## Running on LRZ

| Job | Slurm file under `scripts/` | YAML under `configs/` |
| --- | --- | --- |
| GPT-2 10M fine-tuning, BLEU and ParaPLUIE | `gpt2/gpt2_10m.slurm` | `gpt2/gpt2_10m.yaml` |
| GPT-2 100M fine-tuning, BLEU and ParaPLUIE | `gpt2/gpt2_100m.slurm` | `gpt2/gpt2_100m.yaml` |
| Llama inference and scoring for all three prompts | `llama3/llama3.slurm` | `llama3/llama3.yaml` |
| Aggregate reports by sample ID (CPU) | `comparison/aggregate.slurm` | `comparison/aggregate_with_100m.yaml`, `comparison/aggregate_without_100m.yaml` |
| Score the aggregate with Qwen (GPU) | `comparison/judge.slurm` | `comparison/judge_with_100m.yaml`, `comparison/judge_without_100m.yaml` |
| Run the full experiment | `pipeline.slurm` | Selects the configs above for each stage |

Submit 10M and Llama for either comparison. Include 100M for the full comparison.

```bash
sbatch tasks/paraphrase/scripts/gpt2/gpt2_10m.slurm
sbatch tasks/paraphrase/scripts/llama3/llama3.slurm
sbatch tasks/paraphrase/scripts/gpt2/gpt2_100m.slurm  # full comparison only
```

After those jobs finish, submit the aggregate. Once it succeeds, submit the judge.

```bash
sbatch tasks/paraphrase/scripts/comparison/aggregate.slurm
sbatch tasks/paraphrase/scripts/comparison/judge.slurm
```

Both scripts default to the comparison with 100M. For the comparison without
100M, run the same two scripts in order with these configs.

```bash
sbatch tasks/paraphrase/scripts/comparison/aggregate.slurm --config configs/comparison/aggregate_without_100m.yaml
sbatch tasks/paraphrase/scripts/comparison/judge.slurm --config configs/comparison/judge_without_100m.yaml
```

To run the complete experiment in one job:

```bash
sbatch tasks/paraphrase/scripts/pipeline.slurm
```

The pipeline includes 100M by default. Pass `--without-100m` to omit it.
Use `--train_only` to stop after aggregation, or `--judge_only` to score an
existing aggregate. Use `bash` in place of `sbatch` for a
local run with the required resources.

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

Independent launchers accept `--config FILE` to select a specific experiment
YAML. Copy `configs/` to customize a run, preserving its `gpt2/`, `llama3/`,
and `comparison/` subdirectories. Pass the copy to the pipeline with
`--config-dir DIR`.
These options take absolute paths or paths relative to `tasks/paraphrase/`.
Relative model, input and output paths inside each YAML resolve from the task
directory.

To reuse earlier reports, point the scoring or aggregation config to their
paths. When changing the training seed, choose a matching run directory and
update dependent report paths. Generation, scoring and aggregation require
fresh output paths. The judge resumes an existing output for the same comparison.

For a single Python stage, load the root environment and run:

```bash
cd tasks/paraphrase
uv run --active --no-sync python -m src.training.pipeline \
  --mode parapluie --config configs/gpt2/gpt2_10m.yaml
```

The pipeline modes are `train`, `llama_benchmark`, `parapluie`, `database` and
`judge`. For Llama generation or scoring, pass `--prompt_type` to select a run.
The Llama config defaults to `few_shot` for single-stage runs.

BLEU and ParaPLUIE are computed separately. The ParaPLUIE report preserves
BLEU scores and sample IDs from the generation report. Aggregation checks
that sample IDs, source texts and references match across systems.

## Optional local judge

Set CUDA_ROOT to your CUDA 13.2 toolkit directory. Run from the project root
after loading the root environment.

```bash
CUDA_ROOT="/path/to/cuda-13.2"
export PATH="$CUDA_ROOT/bin:$PATH"
export CUDACXX="$CUDA_ROOT/bin/nvcc"
export CMAKE_ARGS="-DGGML_CUDA=ON -DCUDAToolkit_ROOT=$CUDA_ROOT -DCMAKE_CUDA_ARCHITECTURES=80;90"
uv sync --locked --group judge --no-binary-package llama-cpp-python
```

The toolkit must include the CUDA headers and cuBLAS. The launcher uses CUDA libraries from the root
`.venv`. Run on an allocated NVIDIA GPU and supply the
configured GGUF model files for judge runs.

## Source layout

- `src/data/` loads MRPC and QQP and defines prompts and training examples.
- `src/models/` loads and downloads models.
- `src/training/` contains GPT-2 fine-tuning, generation and the job entry points.
- `src/evaluation/` contains the Llama comparison, metrics, reports and judge.

## Offline tests

Load the [root offline-test environment](../../README.md#offline-tests), then
run from the project root:

```bash
uv run --active --locked --no-sync python -B -m unittest discover -s tasks/paraphrase/tests
```
