# src/data/formatter.py

PARAPHRASE_INSTRUCTION = "Paraphrase the following sentence."
PARAPHRASE_PROMPT_TEMPLATE = (
    f"Instruction: {PARAPHRASE_INSTRUCTION}\nInput: {{sentence}}\nOutput:"
)

def get_format_fn(tokenizer, max_length, min_prompt_tokens=32):
    """
    Returns a formatter function closed over the tokenizer and max_length.
    Acts as a dataset mapper for training tokenization & label masking.
    """
    def format_fn(examples):
        input_ids_batch = []
        attention_masks_batch = []
        labels_batch = []

        for idx_sample, (s1, s2) in enumerate(zip(examples['sentence1'], examples['sentence2'])):
            prompt = PARAPHRASE_PROMPT_TEMPLATE.format(sentence=s1)
            target = f" {s2}{tokenizer.eos_token}"

            prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            target_ids = tokenizer(target, add_special_tokens=False)["input_ids"]

            max_target_tokens = max_length - min_prompt_tokens
            if max_target_tokens < 2:
                raise ValueError("max_length must leave at least two target tokens")
            if len(target_ids) > max_target_tokens:
                target_ids = target_ids[: max_target_tokens - 1] + [tokenizer.eos_token_id]
            prompt_budget = max_length - len(target_ids)
            if len(prompt_ids) > prompt_budget:
                prompt_ids = prompt_ids[-prompt_budget:]

            input_ids = prompt_ids + target_ids
            labels = ([-100] * len(prompt_ids)) + target_ids

            attention_mask = [1] * len(input_ids)
            pad_len = max_length - len(input_ids)
            if pad_len > 0:
                input_ids = input_ids + ([tokenizer.pad_token_id] * pad_len)
                attention_mask = attention_mask + ([0] * pad_len)
                labels = labels + ([-100] * pad_len)

            input_ids_batch.append(input_ids)
            attention_masks_batch.append(attention_mask)
            labels_batch.append(labels)

            # Print first few samples to console for visual inspection
            if idx_sample < 2:
                print(f"\n=== Preprocessing Sample {idx_sample} ===")
                print(f"  Prompt: {prompt[:80]}...")
                print(f"  Source (s1): {s1[:60]}...")
                print(f"  Target (s2): {s2[:60]}...")
                print(f"  Prompt token length: {len(prompt_ids)}")
                print(f"  Target token length: {len(target_ids)}")
                print(f"  Input IDs (first 20): {input_ids[:20]}")
                print(f"  Label mask (first 30): {labels[:30]}")

        return {
            "input_ids": input_ids_batch,
            "attention_mask": attention_masks_batch,
            "labels": labels_batch,
        }
    return format_fn

def build_chat_prompt(sentence, prompt_type="few_shot"):
    """
    Generates structured prompt templates for Llama zero-shot/few-shot.
    Preserves original prompt double-escaped newlines to maintain compatibility.
    """
    if prompt_type == "simple_zero_shot":
        return [
            {"role": "system", "content": "You are a helpful assistant. Paraphrase the following sentence."},
            {"role": "user", "content": f"Instruction: Paraphrase the following sentence.\\nInput: {sentence}\\nOutput:"}
        ]
    elif prompt_type == "zero_shot_preserve":
        return [
            {"role": "system", "content": "You are an AI assistant performing a paraphrase task. Make moderate edits: change synonyms and slight word order, but keep the overall structure similar. Preserve as many original words as possible."},
            {"role": "user", "content": f"Instruction: Paraphrase the following sentence while preserving most of the original words.\\nInput: {sentence}\\nOutput:"}
        ]
    else: # default: few_shot
        return [
            {"role": "system", "content": "You are an AI assistant performing a paraphrase task on the MRPC dataset. Your goal is to match the style of the reference sentences exactly. Make moderate edits: change synonyms and slight word order, but keep the overall structure similar."},
            {"role": "user", "content": "Instruction: Paraphrase the following sentence.\\nInput: The technological Nasdaq Composite Index fell 1.92 points to 1,496.90.\\nOutput:"},
            {"role": "assistant", "content": "The technology-focused Nasdaq Composite Index dropped 1.92 points to 1,496.90."},
            {"role": "user", "content": "Instruction: Paraphrase the following sentence.\\nInput: Amgen shares gained 93 cents to $66.83.\\nOutput:"},
            {"role": "assistant", "content": "Shares of Amgen rose 93 cents to $ 66.83."},
            {"role": "user", "content": f"Instruction: Paraphrase the following sentence.\\nInput: {sentence}\\nOutput:"}
        ]
