# PersonaChat

Fine-tune a language model for persona-grounded dialogue and evaluate how well it
ranks the correct response. The default run uses Llama 3.2 3B and the PersonaChat
dataset. Follow the [project setup](../../README.md#setup) and sign in to
Hugging Face if the configured model requires access.

## Run on LRZ

Submit from the repository root. Create the four job log directories before submission.

```bash
mkdir -p tasks/personachat/logs/{sft,dpo,eval,pipeline}
```

| Stage | Script under `scripts/` | Configuration | Output |
| --- | --- | --- | --- |
| Supervised fine-tuning (SFT) | [training/sft.slurm](scripts/training/sft.slurm) | [sft.yaml](configs/sft.yaml) | `results/sft/` |
| Preference optimization (DPO) | [training/dpo.slurm](scripts/training/dpo.slurm) | [dpo.yaml](configs/dpo.yaml) | `results/dpo/` |
| Response ranking | [evaluation/eval.slurm](scripts/evaluation/eval.slurm) | [eval.yaml](configs/eval.yaml) | `results/eval/dpo/` |
| SFT, DPO, evaluation and aggregation | [pipeline.slurm](scripts/pipeline.slurm) | All three defaults | Stage outputs and `results/aggregate/pipeline/dpo/` |

For standalone jobs, submit SFT first, DPO after SFT succeeds, and evaluation
after DPO succeeds. DPO reads the SFT checkpoint, and evaluation defaults to the
DPO checkpoint. The pipeline runs all three stages and then summarizes the
single report at `results/eval/dpo/dpo_eval_results.json`. It writes
`compiled_results.csv` and `compiled_results.json` to
`results/aggregate/pipeline/dpo/`. These paths are relative to `tasks/personachat/`.

```bash
sbatch tasks/personachat/scripts/pipeline.slurm
```

SFT, DPO and the pipeline request H100. Standalone evaluation accepts H100
or A100.

Standalone jobs accept `--config FILE`. Relative config, model and output
paths resolve from `tasks/personachat/`. To evaluate SFT or a saved checkpoint,
copy `configs/eval.yaml` to `configs/my_eval.yaml` in the task directory and
set `model_path` and `output_dir` for that run.

```bash
sbatch tasks/personachat/scripts/evaluation/eval.slurm --config configs/my_eval.yaml
```

## Run one stage directly

Load the root environment, then enter `tasks/personachat/`. Run training or
evaluation on an allocated GPU.

```bash
cd tasks/personachat
uv run --active --no-sync python -B -u -m src.training.pipeline --config configs/sft.yaml
```

## Reports

Evaluation ranks responses by mean loss over the final assistant turn and
reports Hits@1, Hits@5, and MRR. Run the report commands from the repository
root. Their input and output paths are relative to `tasks/personachat/`.

```bash
bash tasks/personachat/scripts/evaluation/reports.sh aggregate
```

Aggregation scans `results/eval/` by default and writes CSV and JSON under
`results/aggregate/`. Pass `--file FILE` to summarize one report or `--dir DIR`
to scan a different directory.

To plot a checkpoint curve, evaluate several checkpoints of the same model.
Use a separate eval config for each checkpoint, set `model_path` to that
checkpoint, and write each report to its own subdirectory under
`results/eval/checkpoints/`.

```bash
bash tasks/personachat/scripts/evaluation/reports.sh plot --dir results/eval/checkpoints
```

The default figure is `results/figures/checkpoint_eval_metrics.png`. Both
report commands accept `--output` to change where they write.
