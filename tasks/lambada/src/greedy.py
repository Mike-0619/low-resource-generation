"""Free-greedy LAMBADA evaluation with local models or Responses APIs."""
import argparse
import json
import shutil
import sys
from contextlib import closing
from pathlib import Path

from . import MAX_NEW_TOKENS
from . import api, data, local, scoring

from tqdm import tqdm


def evaluation_options(model, language, limit, *, backend, batch_size=None, device=None,
                       concurrency=None, context_limit=None, dtype=None, bos=None,
                       reasoning_effort=None):
    if backend not in {"local", "api"}:
        raise ValueError(f"Unsupported backend: {backend}")
    if not model or not str(model).strip():
        raise ValueError("model must be a nonempty local directory or API model name")
    data.get_dataset_spec(language)
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
    parser.add_argument("--language", choices=data.LANGUAGES, required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--device", choices=("cuda", "cpu"))
    parser.add_argument("--context_limit", type=int)
    parser.add_argument("--dtype", choices=("auto", "float32", "float16", "bfloat16"))
    parser.add_argument("--bos", choices=("auto", "always", "never"))
    parser.add_argument("--concurrency", type=int)
    parser.add_argument("--reasoning_effort")
    args = parser.parse_args(argv)
    options = {key: value for key, value in vars(args).items()
               if key != "output_dir"}
    try:
        vars(args).update(vars(evaluation_options(**options)))
    except ValueError as error:
        parser.error(str(error))
    return args


def evaluate(model, language, limit=None, *, backend="local", batch_size=None, device=None,
             concurrency=None, context_limit=None, dtype=None, bos=None,
             reasoning_effort=None):
    options = evaluation_options(model, language, limit, backend=backend, batch_size=batch_size,
                                 device=device, concurrency=concurrency, context_limit=context_limit,
                                 dtype=dtype, bos=bos, reasoning_effort=reasoning_effort)
    return _evaluate(options)


def _evaluate(options):
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
            samples.append({
                **example, **result,
                **scoring.score_word(result["prediction"], example["target_word"], result["completed"]),
            })
    return {
        **model_metadata, "backend": options.backend, "language": options.language, "limit": options.limit,
        "sample_selection": {"mode": "all" if options.limit is None else "limit", "limit": options.limit},
        **dataset_metadata, "oracle_target_token_length": False, "max_new_tokens": MAX_NEW_TOKENS,
        "prompt_template": "{context}", "word_boundary": "punctuation_first_word_v1",
        "scoring_policy": "unfinished_as_incorrect", "status": "complete",
        **scoring.summarize_greedy(samples),
    }, samples


def prepare_output_dir(output_dir, limit=None):
    """Recover interrupted publication and protect full evaluation results."""
    output = Path(output_dir).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not output.is_dir():
        raise NotADirectoryError(output)
    previous = output.with_name(f".{output.name}.previous")
    if previous.exists():
        if output.exists():
            shutil.rmtree(previous)
        else:
            previous.rename(output)
    shutil.rmtree(output.with_name(f".{output.name}.staging"), ignore_errors=True)
    metrics = output / "metrics.json"
    if limit is not None and metrics.is_file():
        old = json.loads(metrics.read_text(encoding="utf-8"))
        if old.get("sample_selection", {}).get("mode") == "all" or old.get("limit") is None:
            raise ValueError("Limited evaluation cannot replace full results; use a different --output_dir")
    return output


def _publish_results(output, payload, samples):
    """Stage a complete result pair and roll back a failed publication."""
    staging = output.with_name(f".{output.name}.staging")
    previous = output.with_name(f".{output.name}.previous")
    try:
        if output.exists():
            shutil.copytree(output, staging, symlinks=True)
        else:
            staging.mkdir()
        for name, values in (("samples.json", samples), ("metrics.json", payload)):
            destination = staging / name
            destination.unlink(missing_ok=True)
            destination.write_text(json.dumps(values, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        if output.exists():
            output.rename(previous)
        try:
            staging.rename(output)
        except BaseException:
            if previous.exists():
                try:
                    previous.rename(output)
                except OSError as error:
                    raise RuntimeError(f"Could not restore old results; backup is preserved at {previous}") from error
            raise
        if previous.exists():
            try:
                shutil.rmtree(previous)
            except OSError as error:
                print(f"Results saved; could not remove backup {previous}: {error}", file=sys.stderr)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def write_greedy_results(output, payload, samples):
    output = prepare_output_dir(output, payload.get("limit"))
    _publish_results(output, payload, samples)


def main(argv=None):
    args = parse_args(argv)
    output = prepare_output_dir(args.output_dir, args.limit)
    if args.backend == "local":
        local.resolve_device(args.device)
    payload, samples = _evaluate(args)
    _publish_results(output, payload, samples)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
