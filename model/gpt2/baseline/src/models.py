"""GPT-2 model configuration, checkpoint saving and final export."""
import torch
from transformers import GPT2Config

from .utils import require_empty, staged_directory


def model_config(cfg):
    return GPT2Config(
        **cfg["model"], n_ctx=cfg["model"]["n_positions"],
        bos_token_id=1, eos_token_id=2, use_cache=True,
        reorder_and_upcast_attn=False, scale_attn_by_inverse_layer_idx=False,
        scale_attn_weights=True,
    )


def save_checkpoint(directory, model, optimizer, scheduler):
    require_empty(directory)
    with staged_directory(directory) as staging:
        torch.save(optimizer.state_dict(), staging / "latest_optimizer.pt")
        torch.save(scheduler.state_dict(), staging / "latest_scheduler.pt")
        torch.save(model.state_dict(), staging / "latest_student.pt")


def export_final(directory, model, tokenizer):
    require_empty(directory)
    with staged_directory(directory) as staging:
        model.save_pretrained(staging, safe_serialization=True)
        tokenizer.save_pretrained(staging)
