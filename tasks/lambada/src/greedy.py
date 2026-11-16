"""Free-greedy LAMBADA evaluation with local models or Responses APIs."""
import argparse
import csv
import json
from contextlib import closing
from pathlib import Path

from . import MAX_NEW_TOKENS
from . import api, data, local, scoring

from tqdm import tqdm

HUMAN_FIELDS = (
    "sample_id", "context", "normalized_prediction", "normalized_target_word",
    "completed", "human_score", "comment",
)


def evaluation_options(model, language, limit, *, backend, batch_size=None, device=None,
                       concurrency=None, context_limit=None, dtype=None, bos=None,
                       reasoning_effort=None):
    if backend not in {"local", "api"}:
        raise ValueError(f"Unsupported backend: {backend}")
    if not model or not str(model).strip():
        raise ValueError("model must be a nonempty local directory or API model name")
    if language not in data.DATASET_SPECS:
        raise ValueError(f"Unsupported language: {language}")
    for name, value in {"limit": limit, "batch_size": batch_size,
                        "concurrency": concurrency, "context_limit": context_limit}.items():
        if value is not None and value < 1:
            raise ValueError(f"{name} must be positive")
    if backend == "local":
        if concurrency is not None or reasoning_effort is not None:
            raise ValueError("--concurrency and --reasoning_effort require --backend api")
        batch_size = 8 if batch_size is None else batch_size
        device, dtype, bos = device or "cuda", dtype or "auto", bos or "auto"
        if device not in {"cuda", "cpu"} or dtype not in {"auto", "float32", "float16", "bfloat16"} or bos not in {"auto", "always", "never"}:
            raise ValueError("Unsupported device, dtype or BOS policy")
        if device == "cpu" and limit is None:
            raise ValueError("CPU is for limited-sample debugging; full evaluation requires CUDA")
    else:
        if any(value is not None for value in (batch_size, device, context_limit, dtype, bos)):
            raise ValueError("Local model options require --backend local")
        if reasoning_effort is not None and not reasoning_effort.strip():
            raise ValueError("reasoning_effort must be nonempty when specified")
        concurrency = 4 if concurrency is None else concurrency
    return argparse.Namespace(backend=backend, model=str(model), language=language, limit=limit,
                              batch_size=batch_size, device=device, concurrency=concurrency,
                              context_limit=context_limit, dtype=dtype, bos=bos, reasoning_effort=reasoning_effort)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("local", "api"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--language", choices=data.DATASET_SPECS, required=True)
    parser.add_argument("--output_dir", required=True)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--limit", type=int, default=20)
    selection.add_argument("--all", action="store_true")
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--device", choices=("cuda", "cpu"))
    parser.add_argument("--context_limit", type=int)
    parser.add_argument("--dtype", choices=("auto", "float32", "float16", "bfloat16"))
    parser.add_argument("--bos", choices=("auto", "always", "never"))
    parser.add_argument("--concurrency", type=int)
    parser.add_argument("--reasoning_effort")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    args.limit = None if args.all else args.limit
    options = {key: value for key, value in vars(args).items()
               if key not in {"all", "output_dir", "overwrite"}}
    try:
        vars(args).update(vars(evaluation_options(**options)))
    except ValueError as error:
        parser.error(str(error))
    return args


def evaluate(model, language, limit=20, *, backend="local", batch_size=None, device=None,
             concurrency=None, context_limit=None, dtype=None, bos=None,
             reasoning_effort=None, on_record=None):
    options = evaluation_options(model, language, limit, backend=backend, batch_size=batch_size,
                                 device=device, concurrency=concurrency, context_limit=context_limit,
                                 dtype=dtype, bos=bos, reasoning_effort=reasoning_effort)
    return _evaluate(options, on_record)


def _evaluate(options, on_record=None):
    runner = None
    if options.backend == "local":
        runner = local.LocalModel.load(options.model, device=options.device, dtype=options.dtype,
                                       context_limit=options.context_limit, bos=options.bos)
    examples, dataset_metadata = data.load_lambada_examples(options.language)
    dataset_metadata = dict(dataset_metadata)
    sample_mapping = dataset_metadata.pop("sample_mapping", None)
    selected, prepared_inputs = [], []
    for index, text in enumerate(examples[:options.limit]):
        context, target_word = data.split_example(text, index)
        if runner is not None:
            try:
                prepared_inputs.append(runner.prepare(context))
            except ValueError as error:
                raise ValueError(f"Invalid context at dataset index {index}: {error}") from error
        source = sample_mapping[index] if options.language == "de" else {"dataset_index": index}
        sample_id = f"de:boldt:{source['dataset_index']}" if options.language == "de" else f"{options.language}:{index}"
        selected.append({"sample_id": sample_id, **source, "context": context, "target_word": target_word})
    if runner is not None:
        model_metadata = runner.metadata(prepared_inputs, options.batch_size, options.dtype)
        generated = runner.generate(prepared_inputs, options.batch_size)
    else:
        model_metadata = api.metadata(options.model, options.concurrency, options.reasoning_effort)
        generated = api.generate(selected, options.model, options.concurrency, options.reasoning_effort)
    samples = []
    with closing(generated):
        for example, result in tqdm(zip(selected, generated, strict=True), total=len(selected)):
            record = {**example, **result,
                      **scoring.score_word(result["prediction"], example["target_word"], result["completed"])}
            if on_record is not None:
                on_record(record)
            samples.append(record)
    return {
        **model_metadata, "backend": options.backend, "language": options.language, "limit": options.limit,
        "sample_selection": {"mode": "all" if options.limit is None else "limit", "limit": options.limit},
        **dataset_metadata, "oracle_target_token_length": False, "max_new_tokens": MAX_NEW_TOKENS,
        "prompt_template": "{context}", "word_boundary": "punctuation_first_word_v1",
        "scoring_policy": "unfinished_as_incorrect", "status": "complete",
        **scoring.summarize_greedy(samples),
    }, samples


def prepare_output_dir(output_dir: str, overwrite: bool) -> Path:
    output = Path(output_dir)
    if output.exists() and not overwrite and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Output exists: {output}; pass --overwrite explicitly")
    output.mkdir(parents=True, exist_ok=True)
    return output


def check_human_annotations(output):
    human_path = output / "human_eval.csv"
    if human_path.exists():
        with human_path.open(encoding="utf-8", newline="") as handle:
            if any(row.get("human_score", "").strip() or row.get("comment", "").strip()
                   for row in csv.DictReader(handle)):
                raise ValueError(f"Refusing to overwrite human annotations: {human_path}")


def write_greedy_results(output, payload, samples):
    check_human_annotations(output)
    (output / "samples.json").write_text(json.dumps(samples, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with (output / "human_eval.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=HUMAN_FIELDS)
        writer.writeheader()
        for item in samples:
            writer.writerow({name: item.get(name, "") for name in HUMAN_FIELDS})
    # Publish success only after every sample and review artifact has been saved.
    temporary = output / "metrics.json.tmp"
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(output / "metrics.json")


def main(argv=None):
    args = parse_args(argv)
    check_human_annotations(Path(args.output_dir))
    if args.backend == "local":
        local.resolve_device(args.device)
    output = prepare_output_dir(args.output_dir, args.overwrite)
    # An interrupted overwrite must not retain a previous success marker.
    for name in ("metrics.json", "metrics.json.tmp", "samples.json", "human_eval.csv"):
        (output / name).unlink(missing_ok=True)
    with (output / "samples.jsonl").open("w", encoding="utf-8") as journal:
        def save_record(record):
            journal.write(json.dumps(record, ensure_ascii=False) + "\n")
            journal.flush()
        payload, samples = _evaluate(args, on_record=save_record)
    write_greedy_results(output, payload, samples)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
