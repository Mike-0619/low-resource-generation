"""Save baseline checkpoints and export the final model."""

import torch

from .utils import staged_directory


def save_checkpoint(directory, model, optimizer, scheduler):
    with staged_directory(directory) as staging:
        torch.save(optimizer.state_dict(), staging / "latest_optimizer.pt")
        torch.save(scheduler.state_dict(), staging / "latest_scheduler.pt")
        torch.save(model.state_dict(), staging / "latest_student.pt")


def export_final(directory, model, tokenizer):
    with staged_directory(directory) as staging:
        model.save_pretrained(staging, safe_serialization=True)
        tokenizer.save_pretrained(staging)
