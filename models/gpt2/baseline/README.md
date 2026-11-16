# GPT-2 Baseline Training

Train a GPT-2 model and tokenizer for each language and corpus size. Complete the
[project setup](../../../README.md#setup) and [set the raw corpus directory](../../../README.md#gpt-2-data).

## Prepare data

### Prepare raw corpora

From the repository root, enter the Baseline directory and prepare English 10M.

```bash
cd models/gpt2/baseline
uv run --active --no-sync python -m src.corpora.prepare_babylm --dataset english-10m
```

| `--dataset` | Output under `$CORPUS_ROOT/babylm-2026/` | Configuration |
| --- | --- | --- |
| `english-10m` | `English/10M/` | [10M](configs/English/10M.yaml) |
| `english-100m` | `English/100M/` | [100M](configs/English/100M.yaml) |
| `german-108m` | `German/108M/` | [108M](configs/German/108M.yaml) |
| `german-10m` | `German/108M/`, `German/10M/` | [10M](configs/German/10M.yaml) |
| `turkish` | `Turkish/10M/`, `Turkish/100M/` | [10M](configs/Turkish/10M.yaml), [100M](configs/Turkish/100M.yaml) |
| `all` | All corpora above | Use the matching configuration |

- The [download inventory](src/corpora/download_manifest.json) pins upstream revisions and checksums.
- German 10M is a deterministic 10% subset of German 108M with 10,791,084 words. It preserves document order within each category and truncates the last selected document as needed.
- The Turkish builder prepares both sizes with the same BabyLM documents and deduplicated TurMix stream (seed 42). The 10M TurMix portion is a prefix of the 100M portion. Accept the [BabyLM access conditions](https://huggingface.co/datasets/BabyLM-community/babylm-tur) and run `uv run --active --no-sync hf auth login` first. [TurMix dataset](https://huggingface.co/datasets/AdaMLLab/TurMix)

The builder reuses matching files and reports conflicts. Derived corpora include a `manifest.json` with source and sampling details.

### Clean and tokenize

Run from the Baseline directory.

```bash
uv run --active --no-sync python -u -m src.data.clean_dataset --config configs/English/10M.yaml
uv run --active --no-sync python -u -m src.data.train_tokenizer --config configs/English/10M.yaml
```

## Train a model

Use the same YAML on a supported CUDA GPU. Training builds the token cache on
demand. [BabyLM 2026 GPT-2 reference implementation at commit 127d0a5](https://github.com/babylm-org/babylm-baselines/blob/127d0a55def41e878afcfbd7dd85eb6732c2abdf/strict-gpt2/training.py)

```bash
uv run --active --no-sync python -u -m src.training.train --config configs/English/10M.yaml
```

## Run on LRZ

Submit from the repository root and create each job's log directory first.
[Slurm submission guide](../../../README.md#running-and-submission)

| Stage | English 10M example | Log directory under Baseline |
| --- | --- | --- |
| Clean both English sizes | [prepare.slurm](scripts/English/prepare.slurm) | `logs/English/prepare/` |
| Train the tokenizer | [tokenizer.slurm](scripts/English/10M/tokenizer.slurm) | `logs/English/10M/tokenizer/` |
| Pretrain GPT-2 | [train.slurm](scripts/English/10M/train.slurm) | `logs/English/10M/train/` |

Wait for each stage to finish before starting the next. Use the matching language
and size under [scripts/](scripts/) for other configurations.

## Outputs

For English 10M, cleaned text is in `data/English/10M/cleaned/` and the tokenizer
is in `tokenizer/English/10M/`. The run directory
`results/English/10M/seed_42/` contains `final/` and `records/`.

## Tests

Load the [offline test environment](../../../README.md#offline-tests) and run from the Baseline directory.

```bash
uv run --active --no-sync python -B -m unittest discover -s tests
```
