import torch
from torch.utils.data import IterableDataset


class TextChunkDataset(IterableDataset):
    def __init__(self, tokenized_text, seq_len: int):
        self.tokens = tokenized_text
        self.seq_len = seq_len

    def __len__(self) -> int:
        return (len(self.tokens) - self.seq_len) // self.seq_len

    def __iter__(self):
        for i in range(0, len(self.tokens) - self.seq_len, self.seq_len):
            chunk = self.tokens[i : i + self.seq_len + 1]
            x = torch.tensor(chunk[:-1], dtype=torch.long)
            y = torch.tensor(chunk[1:], dtype=torch.long)
            yield x, y
