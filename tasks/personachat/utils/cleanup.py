# /src/utils/cleanup.py

import os
from transformers import TrainerCallback

class DynamicCleanupCallback(TrainerCallback):
    """Callback to clean up heavy optimizer files after each checkpoint save."""
    def on_save(self, args, state, control, **kwargs):
        checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        if os.path.exists(checkpoint_dir):
            print(f"\n[Callback] Cleaning up optimizer files at {checkpoint_dir}...")
            for file_name in ["optimizer.pt", "scheduler.pt", "rng_state.pth"]:
                file_path = os.path.join(checkpoint_dir, file_name)
                if os.path.exists(file_path):
                    os.remove(file_path)