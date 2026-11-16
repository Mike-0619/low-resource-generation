# src/models/peft_setup.py
from peft import LoraConfig

def get_peft_config(args):
    """
    Generate the PEFT (LoRA) configuration based on pipeline arguments.
    Returns None if LoRA is disabled.
    """
    if not args.use_lora:
        return None
        
    print(f"Configuring LoRA (r={args.lora_r}, alpha={args.lora_alpha})...")
    return LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules="all-linear",
        modules_to_save=["embed_tokens", "lm_head", "embed_in", "embed_out", "wte", "wpe"],
        task_type="CAUSAL_LM",
        bias="none"
    )