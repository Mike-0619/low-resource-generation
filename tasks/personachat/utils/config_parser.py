# utils/config_parser.py
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class PipelineArguments:
    """
    Data class representing all arguments required for the pipeline.
    """
    # General Setup
    mode: str = field(metadata={"help": "Eval or Train mode. Choices: 'train', 'eval'"})
    stage: str = field(default="sft", metadata={"help": "Training stage. Choices: 'sft', 'dpo'"})
    seed: int = field(default=42, metadata={"help": "Random seed for reproducibility"})
    limit: Optional[int] = field(default=None, metadata={"help": "Limit dataset size for quick testing"})
    dataset: str = field(default="bavard/personachat_truecased", metadata={"help": "Dataset to use"})

    # Paths
    raw_model_name: str = field(default="", metadata={"help": "Name of the model (HuggingFace or Local)"})
    model_path: str = field(default="", metadata={"help": "Path to the model (HuggingFace or Local)"})
    output_dir: str = field(default="", metadata={"help": "Directory to save the trained model/results"})

    # Training Params
    epochs: int = field(default=1, metadata={"help": "Number of training epochs"})
    batch_size: int = field(default=32, metadata={"help": "Batch size per device"})
    lr: float = field(default=5e-5, metadata={"help": "Learning rate"})
    logging_steps: int = field(default=50, metadata={"help": "Number of steps between logging events"})
    use_fp16: bool = field(default=False, metadata={"help": "Enable FP16 mixed precision instead of BF16"})
    use_tf32: bool = field(default=True, metadata={"help": "Enable TF32 mixed precision instead of BF16"})
    load_dtype: str = field(default="float32", metadata={"help": "Dtype to load weights for training: 'float32' (mixed precision with fp32 master weights) or 'bfloat16' (pure bf16, lower memory)"})
    optim: str = field(default="adamw_torch_fused", metadata={"help": "Optimizer type"})
    revision: Optional[str] = field(default=None, metadata={"help": "Specific checkpoint revision to load from the model path (if applicable)"})

    # LoRA Config
    use_lora: bool = field(default=False, metadata={"help": "Enable LoRA fine-tuning"})
    lora_r: int = field(default=8, metadata={"help": "LoRA rank/dimension"})
    lora_alpha: int = field(default=16, metadata={"help": "LoRA alpha"})

    # W&B Config
    use_wandb: bool = field(default=False, metadata={"help": "Enable Weights & Biases logging"})
    wandb_project: str = field(default="babylm_finetune", metadata={"help": "W&B project name"})
    wandb_run_name: Optional[str] = field(default=None, metadata={"help": "Specific run name for W&B"})

    # DPO Config
    dpo_beta: float = field(default=0.1, metadata={"help": "Beta penalty for DPO"})
    dpo_loss_type: str = field(default="sigmoid", metadata={"help": "Loss type for DPO"})

    # Advanced Training Params
    gradient_accumulation_steps: int = field(default=4, metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."})
    gradient_checkpointing: bool = field(default=True, metadata={"help": "If True, use gradient checkpointing to save memory at the expense of slower backward pass."})
    sft_loss_type: str = field(default="chunked_nll", metadata={"help": "Loss function type for SFT. Use 'nll' for standard HF models or 'chunked_nll' for Unsloth."})