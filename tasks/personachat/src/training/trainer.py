# src/training/trainer.py
import os
import wandb
from transformers import TrainerCallback
from trl import SFTConfig, SFTTrainer, DPOConfig, DPOTrainer

from ..data.loader import get_prepared_datasets
from ..models.builder import build_model_and_tokenizer
from ..models.peft_setup import get_peft_config


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


def run_personachat_finetune(args):
    """
    Main orchestrator for SFT and DPO training stages.
    """

    if args.stage == "dpo":
        # 1. Check if model_path is a local directory (not a HuggingFace Hub ID)
        if not os.path.isdir(args.model_path):
            raise ValueError(
                "\n" + "!" * 70 + "\n"
                f" FATAL ERROR: DPO strictly requires a fine-tuned SFT model.\n"
                f" You provided: '{args.model_path}' (Looks like a Hub ID).\n"
                f" Please run the SFT stage first and pass its output directory here.\n"
                + "!" * 70
            )

        has_lora = os.path.exists(os.path.join(args.model_path, "adapter_config.json"))
        has_full = os.path.exists(os.path.join(args.model_path, "config.json"))

        if not (has_lora or has_full):
            raise ValueError(
                "\n" + "!" * 70 + "\n"
                f" ERROR: The directory '{args.model_path}' is invalid.\n"
                f" It lacks 'adapter_config.json' (LoRA) or 'config.json' (Full SFT).\n"
                f" Are you sure this is a successfully completed SFT output folder?\n"
                + "!" * 70
            )

    # 1. Build Model & Tokenizer
    model, tokenizer = build_model_and_tokenizer(args.model_path, args.stage, args.load_dtype, args.revision)

    # 2. Configure PEFT
    peft_config = get_peft_config(args)

    # 3. Setup Weights & Biases
    report_strategy = "none"
    if args.use_wandb:
        print("Initializing Weights & Biases logging...")
        run_name = args.wandb_run_name if args.wandb_run_name else f"personachat_{args.stage}_{os.path.basename(args.model_path)}"
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config=vars(args)
        )
        report_strategy = "wandb"

    # 4. Load Data (Notice: No tokenizer passed!)
    train_data, val_data = get_prepared_datasets(stage=args.stage, limit=args.limit, dataset_name=args.dataset)

    train_fp16 = args.use_fp16
    train_bf16 = not args.use_fp16
    train_tf32 = args.use_tf32
    optim = args.optim

    model.config.use_cache = False

    # 5. Route to specific Trainer
    if args.stage == "sft":
        print("--- STARTING SUPERVISED FINE-TUNING (SFT) ---")
        sft_lr = 2e-4 if args.use_lora else args.lr

        training_args = SFTConfig(
            output_dir=args.output_dir,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=sft_lr,
            fp16=train_fp16,
            bf16=train_bf16,
            tf32=train_tf32,
            save_strategy="epoch",
            logging_steps=args.logging_steps,
            dataloader_num_workers=8,
            eval_strategy="epoch",
            optim=optim,
            report_to=report_strategy,
            assistant_only_loss=True,
            max_length=512,
            loss_type=args.sft_loss_type,
            gradient_checkpointing=args.gradient_checkpointing,
            seed=args.seed,
            warmup_steps=0.03,
        )

        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_data,
            eval_dataset=val_data,
            processing_class=tokenizer,
            peft_config=peft_config,
            callbacks=[DynamicCleanupCallback()]
        )

    elif args.stage == "dpo":
        print("--- STARTING DIRECT PREFERENCE OPTIMIZATION (DPO) ---")
        dpo_lr = 1e-5 if args.use_lora else 1e-6
        
        training_args = DPOConfig(
            output_dir=args.output_dir,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=2,
            learning_rate=dpo_lr,
            fp16=train_fp16,
            bf16=train_bf16,
            tf32=train_tf32,
            save_strategy="epoch",
            logging_steps=args.logging_steps,
            dataloader_num_workers=8,
            eval_strategy="epoch",
            optim=optim,
            max_length=512,
            max_prompt_length=256,
            beta=args.dpo_beta,
            loss_type=args.dpo_loss_type,
            gradient_checkpointing=True,
            report_to=report_strategy,
            seed=args.seed,
        )
        
        trainer = DPOTrainer(
            model=model,
            args=training_args,
            train_dataset=train_data,
            eval_dataset=val_data,
            processing_class=tokenizer,
            peft_config=peft_config,
            callbacks=[DynamicCleanupCallback()]
        )
    else:
        raise ValueError(f"Unknown stage: {args.stage}. Choose 'sft' or 'dpo'.")

    # 6. Execute Training
    trainer.train()
    
    # 7. Save & Cleanup
    tokenizer.save_pretrained(args.output_dir)
    trainer.save_model(args.output_dir)

    if args.use_wandb:
        wandb.finish()
    
    print(f"FINISHED! Model saved to: {args.output_dir}")
