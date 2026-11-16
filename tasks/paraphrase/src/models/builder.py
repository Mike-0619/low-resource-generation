# src/models/builder.py
from transformers import AutoTokenizer, AutoModelForCausalLM

def build_model_and_tokenizer(model_path: str, **kwargs):
    """
    Load causal language model and tokenizer from the specified path.
    Standardizes padding token setup.
    """
    print(f"Loading Model and Tokenizer from: {model_path}...")
    import torch
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Fix padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Default model kwargs for optimized GPU loading if CUDA is available
    if torch.cuda.is_available():
        if "device_map" not in kwargs:
            kwargs["device_map"] = "auto"
        if "torch_dtype" not in kwargs:
            kwargs["torch_dtype"] = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    model.config.pad_token_id = tokenizer.pad_token_id
    
    # Configure default tokenizer sides for SFT (these can be overridden if needed for inference)
    tokenizer.truncation_side = "left" 
    tokenizer.padding_side = "right"
    
    return model, tokenizer
