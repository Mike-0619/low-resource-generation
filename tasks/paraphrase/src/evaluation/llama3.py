"""Generate Llama paraphrases and report the benchmark scores."""

import re
from datetime import datetime
from pathlib import Path

import torch
from transformers import GenerationConfig, set_seed

from ..data.formatter import build_chat_prompt
from ..data.loader import load_mrpc_test_positives
from ..models.builder import build_model_and_tokenizer
from .metrics import compute_copy_ratio, compute_generation_diagnostics, evaluate_bleu
from .parapluie import evaluate_parapluie
from .reports import prediction_details, require_new_file, write_report


def clean_generated_text(text):
    if not text:
        return ""

    clean_text = text.strip()

    if "Output:" in clean_text:
        clean_text = clean_text.split("Output:")[-1].strip()

    # Remove common chat role prefixes like "assistant" or ":assistant"
    clean_text = re.sub(r"^\s*:?[ \t]*assistant\b[\s:\-]*", "", clean_text, flags=re.IGNORECASE)
    clean_text = clean_text.strip()

    paragraphs = [p.strip() for p in re.split(r"\n+", clean_text) if p.strip()]
    if paragraphs:
        if len(paragraphs) > 1 and re.search(r"(here|sure|certainly|version|paraphrase|below).*:", paragraphs[0], flags=re.IGNORECASE):
            clean_text = paragraphs[1]
        else:
            clean_text = paragraphs[0]

    clean_text = re.sub(r"^(here'?s? (is )?(a )?paraphrased? version( of the sentence)?:?|sure,? here is.*:?|certainly:?)\s*", "", clean_text, flags=re.IGNORECASE).strip()
    clean_text = re.sub(r"\\assistant", "", clean_text, flags=re.IGNORECASE).strip()

    lower_text = clean_text.lower()
    stop_markers = [
        "note:",
        "(note:",
        "please provide",
        "let me know if",
        "can i help you",
        "if you meant to say",
        "i made some minor adjustments",
        "i cannot fulfill your request",
        "i am just an ai",
        "i'm just an ai",
    ]
    cut_positions = [lower_text.find(marker) for marker in stop_markers if lower_text.find(marker) != -1]
    if cut_positions:
        clean_text = clean_text[: min(cut_positions)].strip()

    clean_text = re.sub(r"\s+", " ", clean_text).strip()
    return clean_text

def generate_predictions(model, tokenizer, dataset, batch_size, max_input_length, max_new_tokens, prompt_type):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer.padding_side = "left"
    generation_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        min_new_tokens=0,
        do_sample=False,
        num_beams=4,
        no_repeat_ngram_size=0,
        repetition_penalty=1.1,
        length_penalty=1.0,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    predictions = []
    sources = []
    references = []

    current_bs = batch_size
    i = 0
    total = len(dataset)

    print(f"Generating on {device} with initial batch_size={batch_size}...")

    while i < total:
        batch_end = min(i + current_bs, total)
        batch = dataset[i:batch_end]

        chat_texts = [
            tokenizer.apply_chat_template(
                build_chat_prompt(s, prompt_type),
                tokenize=False,
                add_generation_prompt=True,
            )
            for s in batch["sentence1"]
        ]

        try:
            inputs = tokenizer(
                chat_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_input_length,
            )
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            with torch.no_grad():
                input_ids_len = inputs["input_ids"].shape[1]
                outputs = model.generate(
                    **inputs,
                    generation_config=generation_config,
                    max_length=None,
                )
            generated_ids = outputs[:, input_ids_len:]
            decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

            for pred, s1, s2 in zip(decoded, batch["sentence1"], batch["sentence2"]):
                clean_pred = clean_generated_text(pred)
                predictions.append(clean_pred)
                sources.append(s1)
                references.append(s2)

            i = batch_end

        except RuntimeError as e:
            msg = str(e).lower()
            if "out of memory" in msg and current_bs > 8:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                next_bs = max(8, current_bs // 2)
                print(
                    f"CUDA OOM at batch_size={current_bs}. Retrying same step with batch_size={next_bs}."
                )
                current_bs = next_bs
                continue
            raise

    sample_ids = [f"mrpc-test-{idx}" for idx in dataset["idx"]]
    if not (len(sample_ids) == len(sources) == len(predictions) == len(references)):
        raise RuntimeError("Generated output count does not match the MRPC test sample count")
    return sample_ids, sources, predictions, references

def run_llama_benchmark(args):
    """
    Modular execution function for Llama Zero-Shot / Few-Shot Paraphrasing Benchmark.
    """
    require_new_file(args.report_path)
    seed = getattr(args, 'seed', 42)
    set_seed(seed)
    print(f"--- Starting Llama Benchmark (Model: {args.model_id}) ---")

    # Load model and tokenizer using the builder
    model, tokenizer = build_model_and_tokenizer(args.model_id)
    model.eval()

    test_dataset = load_mrpc_test_positives()

    sample_ids, sources, predictions, references = generate_predictions(
        model=model,
        tokenizer=tokenizer,
        dataset=test_dataset,
        batch_size=args.batch_size,
        max_input_length=args.max_input_length,
        max_new_tokens=args.max_new_tokens,
        prompt_type=args.prompt_type,
    )

    bleu_results = evaluate_bleu(predictions, sources, references)
    predictions = bleu_results["predictions"]
    sources = bleu_results["sources"]
    references = bleu_results["references"]
    bleu_ref = bleu_results["bleu_ref"]
    bleu_src = bleu_results["bleu_src"]

    parapluie_results = None
    if args.with_parapluie:
        parapluie_results = evaluate_parapluie(
            predictions, sources, args.judge_model_id
        )

    copy_ratio = compute_copy_ratio(sources, predictions)
    diagnostics = compute_generation_diagnostics(predictions)

    print("\n" + "=" * 50)
    print(f"Model: {args.model_id}")
    print(f"Samples: {len(predictions)}")
    print(f"BLEU vs Reference (higher better): {bleu_ref:.4f}")
    print(f"BLEU vs Source (lower better):    {bleu_src:.4f}")
    print(f"Exact Copy Ratio (lower better):  {copy_ratio:.4f}")
    if parapluie_results is not None:
        print(
            "ParaPLUIE (higher better):        "
            f"{parapluie_results['parapluie']:.4f}"
        )
    print("=" * 50)

    report = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "model_id": str(Path(args.model_id).resolve()) if Path(args.model_id).is_dir() else args.model_id,
            "seed": seed,
            "prompt_type": args.prompt_type,
            "do_sample": False,
            "batch_size": args.batch_size,
            "max_input_length": args.max_input_length,
            "max_new_tokens": args.max_new_tokens,
            "with_parapluie": args.with_parapluie,
            "judge_model_id": args.judge_model_id if args.with_parapluie else None,
            "bleu": bleu_results["config"],
            "parapluie": (
                parapluie_results["config"] if parapluie_results else None
            ),
        },
        "metrics": {
            "bleu_ref": bleu_ref,
            "bleu_src": bleu_src,
            "exact_copy_ratio": copy_ratio,
            "parapluie": (
                parapluie_results["parapluie"] if parapluie_results else None
            ),
            "parapluie_median": (
                parapluie_results["parapluie_median"]
                if parapluie_results
                else None
            ),
            "parapluie_positive_ratio": (
                parapluie_results["parapluie_positive_ratio"]
                if parapluie_results
                else None
            ),
        },
        "diagnostics": diagnostics,
        "details": prediction_details(sample_ids, sources, predictions, references),
    }

    write_report(args.report_path, report)

    print(f"Report saved to {args.report_path}")
