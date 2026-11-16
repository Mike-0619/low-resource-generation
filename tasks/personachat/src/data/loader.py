# src/data/loader.py
from datasets import load_dataset
from src.data.formatter import format_sft_personachat, format_dpo_personachat

def get_prepared_datasets(stage="sft", limit=None, dataset_name="bavard/personachat_truecased"):
    """
    Load, format, and return the PersonaChat dataset based on the training stage.
    """
    print(f"Loading Dataset {dataset_name} for {stage.upper()}...")
    
    raw_train = load_dataset(dataset_name, split="train", revision="refs/convert/parquet")
    raw_val = load_dataset(dataset_name, split="validation", revision="refs/convert/parquet")

    if limit:
        raw_train = raw_train.select(range(limit))
        raw_val = raw_val.select(range(limit // 10 if limit >= 10 else limit))

    print(f"Dataset not set, Using default dataset ({dataset_name}) for training ")
    print(f"Formatting Text for {stage.upper()}...")
    
    # Select the appropriate formatting function based on the stage
    if stage == "sft":
        formatter_fn = format_sft_personachat
    elif stage == "dpo":
        formatter_fn = format_dpo_personachat
    else:
        raise ValueError(f"Unknown training stage: {stage}")

    # Map the formatting function across the dataset using multiprocessing
    formatted_train = raw_train.map(
        formatter_fn, 
        num_proc=8,
        remove_columns=raw_train.column_names
    )
    formatted_val = raw_val.map(
        formatter_fn, 
        num_proc=8,
        remove_columns=raw_val.column_names
    )

    return formatted_train, formatted_val