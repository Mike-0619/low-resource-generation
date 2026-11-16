# src/data/loader.py
from datasets import load_dataset, concatenate_datasets

def get_augmented_dataset(qqp_size=5000):
    print(f"--- Loading MRPC and Augmenting with GLUE-QQP ({qqp_size} samples) ---")
    
    # 1. Load MRPC
    mrpc = load_dataset("nyu-mll/glue", "mrpc")
    mrpc_train = mrpc["train"].filter(lambda x: x['label'] == 1)
    mrpc_val = mrpc["validation"].filter(lambda x: x['label'] == 1)
    # Match schema with QQP by removing columns not shared by both datasets.
    mrpc_train = mrpc_train.remove_columns(['idx', 'label'])

    if qqp_size <= 0:
        print(f"--- MRPC positives: {len(mrpc_train)} | QQP augmentation disabled ---")
        return mrpc_train.shuffle(seed=42), mrpc_val, mrpc["test"]
    
    # 2. Load QQP from GLUE
    qqp = load_dataset("nyu-mll/glue", "qqp")["train"]
    qqp_dup = qqp.filter(lambda x: x['label'] == 1) # In QQP, label 1 means duplicate
    effective_qqp_size = min(qqp_size, len(qqp_dup))
    print(
        f"--- MRPC positives: {len(mrpc_train)} | QQP duplicates used: {effective_qqp_size} ---"
    )
    qqp_sampled = qqp_dup.shuffle(seed=42).select(range(effective_qqp_size))
    
    # 3. Transform QQP columns to match MRPC (sentence1, sentence2)
    def transform_qqp(example):
        return {
            "sentence1": example['question1'],
            "sentence2": example['question2']
        }
    
    qqp_final = qqp_sampled.map(transform_qqp, remove_columns=qqp_sampled.column_names)
    
    # 4. Combine datasets
    combined_train = concatenate_datasets([
        mrpc_train,
        qqp_final
    ])
    
    return combined_train.shuffle(seed=42), mrpc_val, mrpc["test"]

def load_mrpc_test_positives():
    print("Loading MRPC test split (label=1 only)...")
    mrpc = load_dataset("nyu-mll/glue", "mrpc")
    return mrpc["test"].filter(lambda x: x["label"] == 1)
