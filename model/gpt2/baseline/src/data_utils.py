"""Official full-source tokenization, chunking, padding and loss masks."""
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class BaselineDataset(Dataset):
    """Official full-source tokenization and 510-token chunks, including tails."""

    def __init__(self, directory, tokenizer, sequence_length):
        files = sorted(Path(directory).glob("*.train.txt"))
        if not files:
            raise FileNotFoundError(f"No cleaned training sources: {directory}")
        self.bos, self.eos = tokenizer.bos_token_id, tokenizer.eos_token_id
        self.data = []
        payload = sequence_length - 2
        for path in files:
            text = path.read_text(encoding="utf-8")
            tokens = tokenizer(text=[text], add_special_tokens=False)["input_ids"][0]
            self.data.extend(tokens[start:start + payload] for start in range(0, len(tokens), payload))
        if not self.data:
            raise ValueError("Empty training corpus")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return torch.tensor([self.bos] + self.data[index] + [self.eos], dtype=torch.long)


def collate_batch(batch, eos):
    tokens = pad_sequence(batch, padding_value=eos, batch_first=True)
    inputs, targets = tokens[:, :-1], tokens[:, 1:]
    mask = inputs != eos
    mask[:, 0] = True
    return inputs, targets, mask
