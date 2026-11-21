# PersonaChat

Fine-tune a model for persona-grounded dialogue and evaluate response ranking.
The default uses Llama 3.2 3B and PersonaChat. Complete the
[project setup](../../README.md#setup) and sign in to Hugging Face if needed.

## Run with Slurm

From the repository root, create the job log directories before submitting.

```bash
mkdir -p tasks/personachat/logs/{sft,dpo,eval,pipeline}
```

| Stage | Script under `scripts/` | Configuration | Output |
| --- | --- | --- | --- |
| Supervised fine-tuning (SFT) | [training/sft.slurm](scripts/training/sft.slurm) | [sft.yaml](configs/sft.yaml) | `results/sft/` |
| Preference optimization (DPO) | [training/dpo.slurm](scripts/training/dpo.slurm) | [dpo.yaml](configs/dpo.yaml) | `results/dpo/` |
| Response ranking | [evaluation/eval.slurm](scripts/evaluation/eval.slurm) | [eval.yaml](configs/eval.yaml) | `results/eval/dpo/` |
| SFT, DPO, evaluation and aggregation | [pipeline.slurm](scripts/pipeline.slurm) | All three defaults | Stage outputs and `results/aggregate/pipeline/dpo/` |

For standalone runs, submit SFT, DPO, then evaluation, waiting for each
stage to succeed. Each stage uses the previous model output by default.
The pipeline also summarizes `results/eval/dpo/dpo_eval_results.json` as
`compiled_results.csv` and `compiled_results.json` in
`results/aggregate/pipeline/dpo/` under `tasks/personachat/`.

```bash
sbatch --partition=YOUR_GPU_PARTITION --gres=gpu:1 tasks/personachat/scripts/pipeline.slurm
```

SFT, DPO, evaluation and pipeline jobs request one GPU.

Standalone jobs accept `--config FILE`. Relative paths resolve from
`tasks/personachat/`. To evaluate SFT or a saved checkpoint, copy
`configs/eval.yaml` to `configs/my_eval.yaml` and set `model_path` and
`output_dir`.

```bash
sbatch --partition=YOUR_GPU_PARTITION --gres=gpu:1 tasks/personachat/scripts/evaluation/eval.slurm --config configs/my_eval.yaml
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

For a checkpoint curve, set `model_path` in a separate config for each
checkpoint of the same model. Save each report to its own subdirectory
under `results/eval/checkpoints/`.

```bash
bash tasks/personachat/scripts/evaluation/reports.sh plot --dir results/eval/checkpoints
```

The default figure is `results/figures/checkpoint_eval_metrics.png`. Both
report commands accept `--output` to change where they write.
