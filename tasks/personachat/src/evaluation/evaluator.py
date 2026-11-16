# src/evaluation/evaluator.py
import os
import json
import subprocess
import torch
import torch.nn.functional as F
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import AutoPeftModelForCausalLM

from .metrics import calculate_ranking_metrics, aggregate_scores
from ..data.formatter import build_personachat_messages
from ..models.builder import build_model_and_tokenizer

# Same precision for every model so scores are comparable across the results table
EVAL_DTYPE = torch.bfloat16

def score_candidates(model, tokenizer, messages, candidates, device, max_length=512):
    """
    Compute the mean NLL of every candidate response given the conversation
    history, in one batched forward pass.

    Each candidate is appended as the final assistant turn and rendered by
    apply_chat_template, so the scored sequence is identical to what the model
    saw during training. Labels come from the template's own assistant mask
    rather than from counting prompt tokens by hand.
    """
    sequences = []
    for candidate in candidates:
        enc = tokenizer.apply_chat_template(
            messages + [{"role": "assistant", "content": candidate}],
            tokenize=True,
            return_dict=True,
            return_assistant_tokens_mask=True,
            add_generation_prompt=False,
        )
        input_ids = enc["input_ids"]
        assistant_mask = enc["assistant_masks"]

        if not any(assistant_mask):
            raise ValueError(
                "Empty assistant mask: the chat template is missing {% generation %}, "
                "so there is nothing to score. Check build_model_and_tokenizer()."
            )

        # Mask everything but the assistant turn so loss covers only the candidate
        # Every assistant turn is wrapped in {% generation %}, including the model's
        # earlier turns in the history. Score only the final span: the candidate.
        end = len(assistant_mask)
        while end > 0 and not assistant_mask[end - 1]:
            end -= 1
        start = end
        while start > 0 and assistant_mask[start - 1]:
            start -= 1
        labels = [tid if start <= i < end else -100 for i, tid in enumerate(input_ids)]


        # Left-truncate to fit the model's max input length (keeps the candidate intact)
        sequences.append((input_ids[-max_length:], labels[-max_length:]))

    # Right-pad: under causal attention real tokens never attend to trailing pads and
    # keep their original position ids, so scores match the unpadded sequences exactly
    batch_len = max(len(ids) for ids, _ in sequences)
    input_ids = torch.full((len(sequences), batch_len), tokenizer.pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros_like(input_ids)
    labels = torch.full_like(input_ids, -100)
    for row, (ids, labs) in enumerate(sequences):
        input_ids[row, :len(ids)] = torch.tensor(ids)
        attention_mask[row, :len(ids)] = 1
        labels[row, :len(labs)] = torch.tensor(labs)

    input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

    # outputs.loss would average over the whole batch; we need one mean NLL per candidate
    shift_logits = logits[:, :-1].float()
    shift_labels = labels[:, 1:]
    token_nll = F.cross_entropy(
        shift_logits.transpose(1, 2), shift_labels, ignore_index=-100, reduction="none"
    )
    num_tokens = (shift_labels != -100).sum(dim=1)

    return (token_nll.sum(dim=1) / num_tokens).tolist()


def get_git_state():
    """Return the current commit and whether the working tree has uncommitted changes."""
    repo_dir = os.path.dirname(os.path.abspath(__file__))
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_dir, text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=repo_dir, text=True, stderr=subprocess.DEVNULL
        ).strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"commit": None, "dirty": None}
    return {"commit": commit, "dirty": dirty}


def run_personachat_evaluate(args):
    """
    Main evaluation pipeline to compute Hits@1, Hits@5, and MRR metrics.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading Model: {args.model_path} on {device}...")

    # 1. Setup Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Smart Model Loading (Handles both Full Models and PEFT Adapters)
    if os.path.exists(os.path.join(args.model_path, "adapter_config.json")):
        print("Detected LoRA Adapter. Loading base model and merging weights...")
        peft_model = AutoPeftModelForCausalLM.from_pretrained(
            args.model_path, 
            dtype=EVAL_DTYPE
        ).to(device)
        model = peft_model.merge_and_unload().to(device)

        # To check if this is correct
        # from peft import PeftConfig
        # peft_conf = PeftConfig.from_pretrained(args.model_path)
        
        # # Load Base Model bằng Builder
        # model, tokenizer = build_model_and_tokenizer(peft_conf.base_model_name_or_path, stage="eval", load_dtype="bfloat16")
        
        # # Đắp LoRA lên và merge
        # model = PeftModel.from_pretrained(model, args.model_path)
        # model = model.merge_and_unload().to(device)
    else:
        print("Detected Standard Model. Loading full weights...")
        model, tokenizer = build_model_and_tokenizer(args.model_path, stage="eval", load_dtype="bfloat16")
        model = model.to(device)
        
    model.eval()

    # 3. Load Dataset
    print("Loading Evaluation Validation...")
    val_dataset = load_dataset("bavard/personachat_truecased", split="validation", revision="refs/convert/parquet")
    
    if args.limit:
        val_dataset = val_dataset.select(range(args.limit))
        print(f"TEST with {args.limit} samples.")

    total = len(val_dataset)
    results_detail = []

    # 4. Evaluation Loop
    print("Calculating (Hits@1, Hits@5, MRR)...")
    for i in tqdm(range(total), desc="Evaluating"):
        example = val_dataset[i]
        messages = build_personachat_messages(example)
        candidates = example['candidates']
        
        gold_index = len(candidates) - 1 
        
        # Calculate loss for every candidate
        losses = score_candidates(model, tokenizer, messages, candidates, device)

        # Delegate math to metrics.py
        gold_rank, pred_index = calculate_ranking_metrics(losses, gold_index)

        results_detail.append({
            "question_idx": i,
            "gold_index": gold_index,
            "pred_index": pred_index,
            "gold_rank": gold_rank,
            "losses": losses 
        })

    # 5. Aggregate and Print Results
    scores = aggregate_scores(results_detail, total)

    print("="*50)
    print(" EVALUATION RESULTS ")
    print("="*50)
    print(f" HITS@1: {scores['hits_at_1_pct']:.2f}% ({scores['hits_at_1_count']}/{total})")
    print(f" HITS@5: {scores['hits_at_5_pct']:.2f}% ({scores['hits_at_5_count']}/{total})")
    print(f" MRR   : {scores['mrr_score']:.4f} (Max: 1.0)")
    print("="*50)

    # 6. Save JSON Report
    model_name = os.path.basename(os.path.normpath(args.model_path))
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, f"{model_name}_eval_results.json")
    
    final_output = {
        "raw_model_name": getattr(args, "raw_model_name", args.model_path) or args.model_path,
        "model_path": args.model_path,
        "revision": getattr(args, "revision", "main"),
        "seed": getattr(args, "seed", 42),
        "total_evaluated": total,
        "hits_at_1": scores['hits_at_1_pct'],
        "hits_at_5": scores['hits_at_5_pct'],
        "mrr": scores['mrr_score'],
        # Provenance: which code, precision and end-of-turn token produced these numbers
        "eval_protocol": "chat_template_assistant_mask",
        "eval_dtype": str(EVAL_DTYPE),
        "eos_token": tokenizer.eos_token,
        "git": get_git_state(),
        "details": results_detail
    }
    
    with open(output_file, "w") as f:
        json.dump(final_output, f, indent=4)
        
    print(f"Results saved to: {output_file}")