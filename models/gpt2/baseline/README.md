# GPT-2 Baseline Training

Train a GPT-2 model and tokenizer for each language and corpus size. Complete the
[project setup](../../../README.md#setup) and [set the raw corpus directory](../../../README.md#gpt-2-data).

## Prepare data

### Prepare raw corpora

With the root environment active, prepare English 10M.

```bash
cd models/gpt2/baseline
uv run --active --no-sync python -B -m src.corpora.prepare_babylm --dataset english-10m
```

Raw corpus paths are relative to `$CORPUS_ROOT/babylm-2026/`.

| Experiment | Raw corpus | `--dataset` | Config | LRZ preparation |
| --- | --- | --- | --- | --- |
| English 10M | `English/10M/` | `english-10m` | [YAML](configs/English/10M.yaml) | [prepare.slurm](scripts/English/10M/prepare.slurm) |
| English 100M | `English/100M/` | `english-100m` | [YAML](configs/English/100M.yaml) | [prepare.slurm](scripts/English/100M/prepare.slurm) |
| German 10M | `German/10M/` | `german-10m` | [YAML](configs/German/10M.yaml) | [prepare.slurm](scripts/German/10M/prepare.slurm) |
| German 108M | `German/108M/` | `german-108m` | [YAML](configs/German/108M.yaml) | [prepare.slurm](scripts/German/108M/prepare.slurm) |
| Turkish 10M | `Turkish/10M/` | `turkish` | [YAML](configs/Turkish/10M.yaml) | [prepare.slurm](scripts/Turkish/10M/prepare.slurm) |
| Turkish 100M | `Turkish/100M/` | `turkish` | [YAML](configs/Turkish/100M.yaml) | [prepare.slurm](scripts/Turkish/100M/prepare.slurm) |

- The [download inventory](src/corpora/download_manifest.json) pins upstream revisions and checksums.
- The `german-10m` selector first prepares German 108M and then takes a deterministic 10% subset with 10,791,084 words. It preserves document order within each category and truncates the last selected document as needed.
- The `turkish` selector prepares both sizes with the same BabyLM documents and deduplicated TurMix stream (seed 42). The 10M TurMix portion is a prefix of the 100M portion. Accept the [BabyLM access conditions](https://huggingface.co/datasets/BabyLM-community/babylm-tur) and run `uv run --active --no-sync hf auth login` first. [TurMix dataset](https://huggingface.co/datasets/AdaMLLab/TurMix)

Use `--dataset all` to prepare all raw corpora. The builder reuses matching files and reports conflicts. Derived corpora include a `manifest.json` with source and sampling details.

### Clean and tokenize

Stay in `models/gpt2/baseline/`.

```bash
uv run --active --no-sync python -B -u -m src.data.clean_dataset --config configs/English/10M.yaml
uv run --active --no-sync python -B -u -m src.data.train_tokenizer --config configs/English/10M.yaml
```

## Train a model

Use the same YAML on a supported CUDA GPU. Training builds the token cache on
demand. [BabyLM 2026 GPT-2 reference implementation at commit 127d0a5](https://github.com/babylm-org/babylm-baselines/blob/127d0a55def41e878afcfbd7dd85eb6732c2abdf/strict-gpt2/training.py)

```bash
uv run --active --no-sync python -B -u -m src.training.train --config configs/English/10M.yaml
```

## Run on LRZ

After preparing the raw corpus, create the English 10M log directories from the repository root.

```bash
mkdir -p models/gpt2/baseline/logs/English/10M/{prepare,tokenizer,train}
```

Use `sbatch` to submit the stages in order. [Slurm submission guide](../../../README.md#running-and-submission)

| Stage | English 10M example | Log directory under Baseline |
| --- | --- | --- |
| Clean English 10M | [prepare.slurm](scripts/English/10M/prepare.slurm) | `logs/English/10M/prepare/` |
| Train the tokenizer | [tokenizer.slurm](scripts/English/10M/tokenizer.slurm) | `logs/English/10M/tokenizer/` |
| Pretrain GPT-2 | [train.slurm](scripts/English/10M/train.slurm) | `logs/English/10M/train/` |

Wait for each stage to finish before submitting the next. For other configurations, match the language and size under [scripts/](scripts/).

## Outputs

For English 10M, cleaned text is in `data/English/10M/cleaned/` and the tokenizer
is in `tokenizer/English/10M/`. The run directory
`results/English/10M/seed_42/` contains the model export in `final/`, training records in `records/`, and saved checkpoints in `checkpoints/`.

## Tests

Load the [offline test environment](../../../README.md#offline-tests) and run from the Baseline directory.

```bash
uv run --active --no-sync python -B -m unittest discover -s tests
```
