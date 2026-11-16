"""Fine-tune GPT-2 and benchmark the fine-tuned model on MRPC."""

import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from transformers import Trainer, TrainingArguments, default_data_collator, set_seed

from ..data.formatter import PARAPHRASE_PROMPT_TEMPLATE, get_format_fn
from ..data.loader import get_augmented_dataset
from ..evaluation.metrics import compute_copy_ratio, compute_generation_diagnostics, evaluate_bleu
from ..evaluation.reports import prediction_details, require_new_file, write_report
from ..models.builder import build_model_and_tokenizer


def run_paraphrase_finetune(args):
    require_new_file(args.report_path)
    require_new_file(args.output_dir)
    set_seed(args.seed)
    print(f"--- Starting GPT-2 Finetune (Seed: {args.seed}) ---")
    model, tokenizer = build_model_and_tokenizer(args.model_path)
    train, validation, test = get_augmented_dataset(args.qqp_augment_size)
    trainer = finetune(args, model, tokenizer, train, validation)
    save_training_artifacts(args, model, tokenizer, trainer.state.log_history)

    test = test.filter(lambda item: item["label"] == 1)
    sample_ids = [f"mrpc-test-{idx}" for idx in test["idx"]]
    sources, references = test["sentence1"], test["sentence2"]
    predictions = generate_predictions(args, model, tokenizer, sources)
    report_predictions(args, sample_ids, sources, predictions, references)


def finetune(args, model, tokenizer, train_dataset_raw, val_dataset_raw):
    format_fn = get_format_fn(tokenizer, args.max_length)
    tokenized_train = train_dataset_raw.map(format_fn, batched=True, remove_columns=train_dataset_raw.column_names)
    tokenized_val = val_dataset_raw.map(format_fn, batched=True, remove_columns=val_dataset_raw.column_names)

    try:
        Path(args.output_dir).mkdir(parents=True)
    except FileExistsError as exc:
        raise FileExistsError(
            f"Output directory already exists: {args.output_dir}. Choose a new --output_dir."
        ) from exc

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        seed=args.seed,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        lr_scheduler_type=args.lr_scheduler_type,
        logging_steps=args.logging_steps,
        eval_strategy=args.eval_strategy,
        save_strategy=args.save_strategy,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=args.load_best_model_at_end,
        metric_for_best_model=args.metric_for_best_model,
        bf16=args.bf16,
        tf32=args.tf32,
        optim=args.optim,
        report_to=args.report_to
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=default_data_collator,
    )

    print("--- Starting Training ---")
    trainer.train()
    return trainer


def save_training_artifacts(args, model, tokenizer, history):
    print("\n" + "="*40)
    print("       TRAINING LOSS SUMMARY")
    print("="*40)
    print(f"{'Epoch':<10} | {'Step':<10} | {'Train Loss':<15} | {'Eval Loss':<15}")
    print("-" * 60)
    for log in history:
        epoch = log.get("epoch", "N/A")
        step = log.get("step", "N/A")
        train_loss = log.get("loss", "N/A")
        eval_loss = log.get("eval_loss", "N/A")
        if train_loss != "N/A" or eval_loss != "N/A":
            t_loss_str = f"{train_loss:.4f}" if isinstance(train_loss, float) else "---"
            e_loss_str = f"{eval_loss:.4f}" if isinstance(eval_loss, float) else "---"
            print(f"{epoch:<10.2f} | {step:<10} | {t_loss_str:<15} | {e_loss_str:<15}")
    print("="*60)

    # Save model and tokenizer
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"--- Model saved to {args.output_dir} ---")

    # Save history json
    history_path = Path(args.output_dir) / "training_history.json"
    with history_path.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    # Plot loss curve
    train_steps, train_losses = [], []
    eval_steps, eval_losses = [], []
    for log in history:
        step = log.get("step")
        if step is None:
            continue
        if "loss" in log and isinstance(log["loss"], (int, float)):
            train_steps.append(step)
            train_losses.append(log["loss"])
        if "eval_loss" in log and isinstance(log["eval_loss"], (int, float)):
            eval_steps.append(step)
            eval_losses.append(log["eval_loss"])

    if train_steps or eval_steps:
        plt.figure(figsize=(10, 6))
        if train_steps:
            plt.plot(train_steps, train_losses, label="Train Loss", marker="o", markersize=3)
        if eval_steps:
            plt.plot(eval_steps, eval_losses, label="Eval Loss", marker="s", markersize=4)
        plt.title("Training and Evaluation Loss")
        plt.xlabel("Global Step")
        plt.ylabel("Loss")
        plt.grid(True, linestyle="--", alpha=0.35)
        plt.legend()
        plt.tight_layout()
        loss_plot_path = Path(args.output_dir) / "training_loss_curve.png"
        plt.savefig(loss_plot_path, dpi=180)
        plt.close()
        print(f"Loss curve saved to {loss_plot_path}")


def generate_predictions(args, model, tokenizer, sources):
    print("\n--- Starting Stage 1 Evaluation ---")
    model.eval()

    # Left padding is required for batched decoder-only generation
    tokenizer.padding_side = "left"
    predictions = []

    for batch_idx, i in enumerate(range(0, len(sources), args.inference_batch_size)):
        batch_sentences = sources[i:i + args.inference_batch_size]
        prompts = [PARAPHRASE_PROMPT_TEMPLATE.format(sentence=s) for s in batch_sentences]

        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=args.max_length).to(model.device)

        # DEBUG: Print first batch info
        if batch_idx == 0:
            print(f"\n=== DEBUG Inference Batch 0 ===")
            print(f"  Prompt[0]: {prompts[0][:80]}...")
            input_ids_cpu = inputs["input_ids"][0].cpu().numpy()
            print(f"  Input IDs[0] (first 20): {list(input_ids_cpu[:20])}")
            print(f"  Attention mask[0] (first 20): {list(inputs['attention_mask'][0][:20].cpu().numpy())}")

        with torch.no_grad():
            input_ids_len = inputs["input_ids"].shape[1]
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                min_new_tokens=args.min_new_tokens,
                num_beams=args.num_beams,
                no_repeat_ngram_size=0,
                repetition_penalty=args.repetition_penalty,
                length_penalty=args.length_penalty,
                do_sample=args.do_sample,
                early_stopping=args.early_stopping,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id
            )

        generated_ids = outputs[:, input_ids_len:]
        decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        for sample_idx, dec in enumerate(decoded):
            res = dec.strip()
            predictions.append(res)

            if batch_idx == 0 and sample_idx == 0:
                print(f"  Full decoded[0]: {dec[:100]}...")
                print(f"  Extracted pred[0]: {res[:100]}...")
    return predictions


def report_predictions(args, sample_ids, sources, predictions, references):
    bleu_results = evaluate_bleu(predictions, sources, references)
    predictions_clean = bleu_results["predictions"]
    sources_clean = bleu_results["sources"]
    references_clean = bleu_results["references"]

    print(f"\nFinal Results:")
    print(f"BLEU vs Reference (S2 - Target High): {bleu_results['bleu_ref']:.4f}")
    print(f"BLEU vs Source (S1 - Target Low): {bleu_results['bleu_src']:.4f}")

    gen_diag = compute_generation_diagnostics(predictions_clean, include_repeats=False)
    copy_ratio = compute_copy_ratio(sources_clean, predictions_clean)

    paraphrase_report = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "model_path": str(Path(args.model_path).resolve()) if Path(args.model_path).is_dir() else args.model_path,
            "seed": args.seed,
            "training": {
                "num_train_epochs": args.num_train_epochs,
                "per_device_train_batch_size": args.per_device_train_batch_size,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "learning_rate": args.learning_rate,
            },
            "generation": {
                "max_new_tokens": args.max_new_tokens,
                "min_new_tokens": args.min_new_tokens,
                "num_beams": args.num_beams,
                "repetition_penalty": args.repetition_penalty,
                "length_penalty": args.length_penalty,
                "do_sample": args.do_sample,
                "early_stopping": args.early_stopping,
            },
            "max_length": args.max_length,
            "qqp_augment_size": args.qqp_augment_size,
            "prompt_template": PARAPHRASE_PROMPT_TEMPLATE,
            "bleu": bleu_results["config"],
        },
        "metrics": {
            "bleu_ref": bleu_results["bleu_ref"],
            "bleu_src": bleu_results["bleu_src"],
            "exact_copy_ratio": copy_ratio,
        },
        "diagnostics": gen_diag,
        "details": prediction_details(sample_ids, sources_clean, predictions_clean, references_clean),
    }

    write_report(args.report_path, paraphrase_report, ensure_ascii=True)
    print(f"Paraphrase report saved to {args.report_path}")
