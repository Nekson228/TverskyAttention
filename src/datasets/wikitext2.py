import os

import torch
from torch.utils.data import DataLoader

from datasets import load_dataset
from transformers import GPT2TokenizerFast

from src.datasets import TextChunkDataset


def get_dataloaders(
    seq_len: int, batch_size: int, datasets_dir: str = "../data/processed"
) -> tuple[DataLoader, DataLoader, int]:
    os.makedirs(datasets_dir, exist_ok=True)

    train_cache_path = os.path.join(datasets_dir, "train_tokens.pt")
    val_cache_path = os.path.join(datasets_dir, "val_tokens.pt")

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    def get_tokens(split_name: str, dataset_path: str):
        if os.path.exists(dataset_path):
            print(f"Loading {split_name} tokens from {dataset_path}...")
            return torch.load(dataset_path)

        print(f"Tokenizing {split_name} split (this might take a minute)...")
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
        text = "\n".join(dataset[split_name]["text"])
        tokens = tokenizer.encode(text)

        print(f"Saving to {dataset_path}...")
        torch.save(tokens, dataset_path)
        return tokens

    train_tokens = get_tokens("train", train_cache_path)
    val_tokens = get_tokens("validation", val_cache_path)

    train_ds = TextChunkDataset(train_tokens, seq_len)
    val_ds = TextChunkDataset(val_tokens, seq_len)

    train_dl = DataLoader(train_ds, batch_size=batch_size)
    val_dl = DataLoader(val_ds, batch_size=batch_size)

    return train_dl, val_dl, tokenizer.vocab_size
