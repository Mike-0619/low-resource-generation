# src/models/builder.py
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os

# One chat format for every model, base or instruct. Native templates break this
# pipeline or confound comparisons: most lack {% generation %} markers, Mistral rejects
# histories that start with an assistant turn, and Llama 3 stamps today's date into the
# system prompt. Every assistant turn is wrapped in {% generation %} for assistant-only loss.
CHATML_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\n' }}"
    "{% if message['role'] == 'assistant' %}"
    "{% generation %}{{ message['content'] + '<|im_end|>\n' }}{% endgeneration %}"
    "{% else %}"
    "{{ message['content'] + '<|im_end|>\n' }}"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '<|im_start|>assistant\n' }}"
    "{% endif %}"
)
CHATML_TOKENS = ["<|im_start|>", "<|im_end|>"]


def init_chatml_embeddings(model, tokenizer, new_tokens, reference_eos_id, num_original_tokens):
    """
    Give newly added ChatML tokens embeddings that look like trained tokens.

    Neither pad nor the rows past len(tokenizer) are safe sources: in Pythia both
    are dead rows (norm ~0.007 vs ~0.79 for trained tokens). <|im_end|> copies the
    original eos, which already means "stop here"; any other new token gets the
    vocabulary mean rescaled to a typical norm.
    """
    embeddings = [model.get_input_embeddings()]
    output_embeddings = model.get_output_embeddings()
    if output_embeddings is not None and output_embeddings.weight is not embeddings[0].weight:
        embeddings.append(output_embeddings)

    with torch.no_grad():
        for emb in embeddings:
            weight = emb.weight
            original = weight[:num_original_tokens]
            mean = original.mean(dim=0)
            typical = mean / mean.norm() * original.norm(dim=1).median()
            for token in new_tokens:
                token_id = tokenizer.convert_tokens_to_ids(token)
                if token == "<|im_end|>" and reference_eos_id is not None:
                    weight[token_id] = weight[reference_eos_id]
                else:
                    weight[token_id] = typical

    weight = embeddings[0].weight
    norms = ", ".join(f"{t} {weight[tokenizer.convert_tokens_to_ids(t)].norm():.4f}" for t in new_tokens)
    print(f"Init ChatML embeddings: {norms}, "
          f"median trained norm {weight[:num_original_tokens].norm(dim=1).median():.4f}")


def build_model_and_tokenizer(model_path: str, stage: str, load_dtype: str = "float32", revision: str = None):
    """
    Load the causal LM and put its tokenizer on the pipeline's ChatML format.

    Every model is trained and evaluated with the same template, so score
    differences come from the model rather than from its chat format.
    """
    print(f"Loading Model and Tokenizer from: {model_path} (revision: {revision})...")
    actual_revision = revision if not os.path.isdir(model_path) else None
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, revision=actual_revision)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model in {load_dtype}...")
    # float32 + bf16=True in the Trainer = mixed precision with fp32 master weights.
    # Pure bfloat16 rounds small AdamW updates to zero; use it only to save memory.
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        revision=actual_revision,
        dtype=getattr(torch, load_dtype),
        low_cpu_mem_usage=True,
    )

    if stage == "dpo" and tokenizer.chat_template != CHATML_TEMPLATE:
        raise ValueError(
            f"DPO needs an SFT checkpoint produced by this pipeline, but '{model_path}' "
            "does not carry its ChatML template. Run the SFT stage first and pass its output dir."
        )

    tokenizer.chat_template = CHATML_TEMPLATE

    # Qwen already has both tokens; most other models need them added
    new_tokens = [t for t in CHATML_TOKENS if t not in tokenizer.get_vocab()]
    if new_tokens:
        print(f"Adding special tokens {new_tokens} to tokenizer and resizing embeddings...")
        original_eos_id = tokenizer.eos_token_id
        num_original_tokens = len(tokenizer)
        tokenizer.add_special_tokens({'additional_special_tokens': new_tokens})

        # Never shrink: Pythia pads its matrix to 50304, well past len(tokenizer)
        new_size = max(len(tokenizer), model.get_input_embeddings().weight.shape[0])
        model.resize_token_embeddings(new_size, mean_resizing=False)
        init_chatml_embeddings(model, tokenizer, new_tokens, original_eos_id, num_original_tokens)

    # The template always closes an assistant turn with <|im_end|>
    tokenizer.eos_token = "<|im_end|>"

    # Standardize padding/truncation for causal LLMs
    tokenizer.truncation_side = "left"
    tokenizer.padding_side = "right"

    return model, tokenizer
