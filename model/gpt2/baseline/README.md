# BabyLM 2026 GPT-2 baselines

English and German GPT-2 experiments adapted from the
[official strict-gpt2 recipe](https://github.com/babylm-org/babylm-baselines/tree/127d0a55def41e878afcfbd7dd85eb6732c2abdf/strict-gpt2).
This directory is part of the parent repository, not a standalone Python project.
It uses the repository's shared uv environment and pinned dependencies.
See the root [setup and launcher guide](../../../README.md).

## Experiments

| Config | Raw corpus |
|---|---|
| [English 10M](configs/English/10M.yaml) | Official BabyLM-2026-Strict-Small |
| [English 100M](configs/English/100M.yaml) | Official BabyLM-2026-Strict |
| [German 10M](configs/German/10M.yaml) | Local original-order 10% subset: 10,791,084 words (legacy 10M label) |
| [German 108M](configs/German/108M.yaml) | Full official babylm-deu: 107,910,839 words |

Each experiment trains its own 16,384-token BPE and randomly initialized GPT-2:
12 layers, 768 hidden units, 12 heads, 1,024 positions, sequence length 512,
dropouts 0.1, seed 42, 10 epochs and global batch 16. Training uses AdamW,
cosine scheduling, BF16 and SDPA by default. Complete settings are in the YAML files.

## Prepare the raw data

Obtain the official English corpora and full German corpus from the Hugging Face
repositories and revisions identified by `dataset.repo_id` and `dataset.revision`.
The German 10M dataset is a local derived subset, not an official release.
The external `_support/data_deu.py` tool reads the pinned German 108M shards in
filename/row order and takes the first 10% of each category's whitespace words.
It floors each category's share, then assigns remaining words by largest
fractional remainder (category-name tie break) to reach the half-up rounded
overall target of 10,791,084 words. Documents are not shuffled and there is no
sampling seed. Only the final document of each category may be shortened at an
original word boundary; earlier documents and internal whitespace are preserved.
The legacy `German/10M` directory now means this 10% subset, about 10.79M words.
The model training seed remains 42. FLORES is not included; LAMBADA is unchanged.

Generate or verify using `uv run --script "$CORPUS_ROOT/babylm-2026/_support/data_deu.py"`
(add `--verify-only` for existing data). The generator refuses to overwrite data.
Category quotas, actual counts and truncated documents are recorded in the
subset manifest. The German 10M config uses its actual document count and
10,791,084 for both expected raw words and `training.words_per_epoch`.
After replacing a subset, clean it, train a new tokenizer, then train a new model;
old subset artifacts must not be reused. This module neither downloads corpora nor
generates subsets, and no raw data is included in Git.

Set `CORPUS_ROOT` in the project `.env` to your external corpus directory:

```text
$CORPUS_ROOT/babylm-2026/
|-- English/{10M,100M}/          # six source *.train.txt files per experiment
|-- German/10M/                 # seven category *.train.parquet files + manifest.json
|-- German/108M/                 # two official train-*.parquet shards
`-- _support/                  # data_deu.py + shared download_manifest.json
```

Exact filenames are listed in `dataset.files`; paths are relative to
`$CORPUS_ROOT/<dataset.corpus_path>`. German Parquet inputs require `text`,
`category` and `language` columns. Language must be `deu`, categories must match
the config, and actual document/whitespace-word totals must match its expectations.

Configured manifests must exist and contain JSON. `dataset.manifest` is relative
to the experiment's raw corpus directory; `dataset.download_manifest` is relative
to `CORPUS_ROOT`. They record provenance, without cross-validation. If preparing
your own inputs without these records, omit the optional `manifest` or
`download_manifest` field in your own config; do not fabricate a manifest.

## Data processing

Cleaning preserves raw inputs and applies the reference's source-specific rules:

- All sources normalize spaces/tabs/nonbreaking spaces and spaces before the
  ASCII and CJK punctuation defined in the [cleaning implementation](src/clean_dataset.py).
- Books, wiki and educational sources also remove HTML tags.
- Subtitle sources remove entire lines containing `subtitle` or `ondertitel`
  (case-insensitive), or the Chinese word for subtitles (Unicode `\u5b57\u5e55`).
- English is processed per source file. German is cleaned per document, then
  joined by category with a newline between records, including empty cleaned rows.
  Input filenames are sorted; record order within each file is preserved.
- No deduplication, lowercasing, resampling or word-count top-up is performed.
  Cleaning can reduce corpus size. NFKC normalization happens during tokenization.

Cleaning writes source `*.train.txt` files and `preparation.json` under
`data/<language>/<size>/cleaned/`. The report records provenance, input/output
hashes and before/after statistics; recorded hashes do not imply verified manifests.

Tokenizer training reads only the sorted cleaned `*.train.txt` files, excluding
the JSON report. Each independent BPE is saved to `tokenizer/<language>/<size>/`.
Model training tokenizes each full source, splits into at most 510 text tokens,
keeps tail chunks and adds BOS/EOS. Batches use EOS padding and the reference's
input-based loss mask: positions whose input is EOS are masked, except position 0.

## Run

Run from the project root. Follow the root [environment and submission guide](../../../README.md#running-and-submission).
The nine independent scripts comprise `prepare_all.slurm`, and
`tokenizer.slurm` / `train.slurm` under `scripts/<language>/<size>/` for the four
configurations. Run preparation, tokenization and training in that order; skip
completed stages. Python checks protect existing artifacts from overwrite.

On already allocated GPU resources, for example:

```bash
bash model/gpt2/baseline/scripts/English/100M/train.slurm
```

For Slurm, create the selected job's log directory first and submit directly
from the project root, for example `sbatch model/gpt2/baseline/scripts/English/100M/train.slurm`.
Each script specifies its LRZ partition and QOS; no shell environment setup is needed
before submission. Other clusters must adapt these directives.
Use `sbatch --test-only path/to/job.slurm` to check scheduling parameters without running the task.
Training uses one CUDA GPU with BF16 support. Training requests 16 GB RAM for
10M, 32 GB for English 100M and 64 GB for German 108M. The German 108M
request has not yet been validated by a successful full run.

For custom Python runs, load the environment as documented in the root README,
then enter `model/gpt2/baseline` and run `uv run --active --locked --no-sync python -m
src.training --config configs/English/100M.yaml`. Cleaning and tokenization use
the same pattern with their respective Python modules.

## Outputs and reproduction limits

- `results/<language>/<size>/seed_<seed>/checkpoints/`: epoch/word-milestone checkpoints.
- `results/<language>/<size>/seed_<seed>/final/`: Hugging Face model and tokenizer.
- `results/<language>/<size>/seed_<seed>/records/`: `config.yaml` (config/runtime),
  `metrics.json` (ordered zero-based epoch/loss records, atomically updated after
  each completed epoch), and `run_summary.json` (written after final export).
- `logs/<language>/<size>/train/`: Slurm `.out` and `.err` files only.

Existing artifacts are not overwritten, and training resume is not supported.
Training records are grouped by seed; the scheduler log directory is shared
across seeds and uses job IDs for filenames. Do not run multiple jobs for the same seed. Data, tokenizers, results and logs are Git-ignored; portable scripts are included.

Optional FA3 requires a local official package, Hopper SM90 and attention dropout
0; choosing it changes the default dropout recipe. No automatic fallback is used.
German experiments extend the reference recipe. The software stack differs from
the official environment (including its Transformers 4.57.3); see the root
`pyproject.toml` and `uv.lock`. Bitwise or result equivalence is not claimed.

## Tests

From this directory with the root environment activated:

```bash
CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 OMP_NUM_THREADS=1 RAYON_NUM_THREADS=1 uv run --active --locked --no-sync python -B -m unittest discover -s tests
```

Tests use temporary fixtures and small CPU models. Script tests simulate execution from the project root and Slurm temporary copies;
they never schedule jobs.
