# GPT-2 Vocabulary Ablation

Compare vocabulary sizes 512, 1,024, 2,048, 4,096, 8,192, and 16,384 across seeds
42, 43, and 44. Complete the [project setup](../../../README.md#setup) and
[set the raw corpus directory](../../../README.md#gpt-2-data).

| Protocol | What it holds fixed |
| --- | --- |
| `shared_spans` | Text spans across vocabularies, with a batch size of 16 |
| `balanced_batch` | Optimizer updates per epoch, with separate tokenization and adjusted batch sizes |

## Prepare data

### Raw corpora

Use the raw corpora under `$CORPUS_ROOT/babylm-2026/`. [Raw corpus preparation](../baseline/README.md#prepare-raw-corpora)

Append `_spans` or `_balanced` to the condition prefix for the cleaning command.

| Experiment | Raw corpus | Condition prefix | Configurations | LRZ preparation |
| --- | --- | --- | --- | --- |
| English 10M | `English/10M/` | `10m_en` | [shared_spans](configs/English/10M/shared_spans.yaml), [balanced_batch](configs/English/10M/balanced_batch.yaml) | [prepare.slurm](scripts/English/10M/prepare.slurm) |
| English 100M | `English/100M/` | `100m_en` | [shared_spans](configs/English/100M/shared_spans.yaml), [balanced_batch](configs/English/100M/balanced_batch.yaml) | [prepare.slurm](scripts/English/100M/prepare.slurm) |
| German 108M | `German/108M/` | `108m_de` | [shared_spans](configs/German/108M/shared_spans.yaml), [balanced_batch](configs/German/108M/balanced_batch.yaml) | [prepare.slurm](scripts/German/108M/prepare.slurm) |

English inputs are training text files. German 108M uses the two full-corpus
Parquet shards. [Input filenames](src/data/clean_dataset.py)

### Build training inputs

Load the root environment, then enter `models/gpt2/ablation/`. The English 10M example
prepares shared initial weights, a nested tokenizer family, and the shared-span cache.

```bash
cd models/gpt2/ablation
uv run --active --no-sync python -B -u -m src.models.initialization --config configs/English/10M/shared_spans.yaml
uv run --active --no-sync python -B -u -m src.data.clean_dataset --condition 10m_en_spans
uv run --active --no-sync python -B -u -m src.data.train_tokenizer --condition 10m_en_spans
uv run --active --no-sync python -B -u -m src.data.data_utils --config configs/English/10M/shared_spans.yaml
```

For `balanced_batch`, reuse the tokenizer and initial weights. Keep the matching
`shared_spans` cache, which supplies the reference update count, then prepare
the additional data.

```bash
uv run --active --no-sync python -B -u -m src.data.clean_dataset --condition 10m_en_balanced
uv run --active --no-sync python -B -u -m src.data.data_utils --config configs/English/10M/balanced_batch.yaml
```

## Train a model

Run from the Ablation directory on a supported CUDA GPU. Use the corresponding
YAML for the protocol you prepared.

```bash
uv run --active --no-sync python -B -u -m src.training.train \
  --config configs/English/10M/shared_spans.yaml --vocab-size 512 --seed 42
```

## Run on LRZ

From the repository root, submit the preparation job for the chosen corpus. It
prepares shared initial weights, cleans both protocols, builds their common
tokenizer, and prepares the shared-span cache before the balanced-batch cache.

```bash
mkdir -p models/gpt2/ablation/logs/English/10M/prepare
sbatch models/gpt2/ablation/scripts/English/10M/prepare.slurm
```

After preparation succeeds, submit the matching training script with one seed.
Each array trains that seed's six vocabulary sizes. [Slurm submission guide](../../../README.md#running-and-submission)

```bash
mkdir -p models/gpt2/ablation/logs/English/10M/shared_spans/train
sbatch models/gpt2/ablation/scripts/English/10M/shared_spans/train.slurm 42
```

Training logs go to `logs/<language>/<size>/<protocol>/train/` under Ablation
as `%A_%a.out` and `%A_%a.err`.

## Outputs

The English 10M inputs use `data/English/10M/<protocol>/` for cleaned text
and token caches, and `tokenizer/English/10M/` for the common tokenizer.
Initial weights are shared across all corpora under `initializations/`.

The example run writes to
`results/English/10M/shared_spans/vocab_512/seed_42/`, with model files in
`final/` and run records in `records/`.

### Resume a run

From the Ablation directory, use the original configuration, vocabulary size,
and seed to resume an unfinished run from `checkpoints/last/`.

```bash
uv run --active --no-sync python -B -u -m src.training.train \
  --config configs/English/10M/shared_spans.yaml --vocab-size 512 --seed 42 --resume
```

## Tests

Load the [offline test environment](../../../README.md#offline-tests) and run from the Ablation directory.

```bash
uv run --active --no-sync python -B -m unittest discover -s tests
```
