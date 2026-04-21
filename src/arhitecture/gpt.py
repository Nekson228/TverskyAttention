import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention_dropin import TverskyMultiHeadAttentionDropIn
from src.enums import ModelType, IntersectionReductionType, DifferenceType


class TverskyBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_features: int,
        model_type: str | ModelType = ModelType.CONTRAST,
        intersection_reduction: str | IntersectionReductionType = IntersectionReductionType.PRODUCT,
        difference_type: str | DifferenceType = DifferenceType.SUBTRACT_MATCH,
    ):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = TverskyMultiHeadAttentionDropIn(
            d_model=d_model,
            num_heads=num_heads,
            num_features=num_features,
            model_type=model_type,
            intersection_reduction=intersection_reduction,
            difference_type=difference_type,
        )
        self.ln_2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(0.1),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x), mask=mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class TverskyGPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        num_features: int,
        max_seq_len: int,
        model_type: str | ModelType = ModelType.CONTRAST,
        intersection_reduction: str | IntersectionReductionType = IntersectionReductionType.PRODUCT,
        difference_type: str | DifferenceType = DifferenceType.SUBTRACT_MATCH,
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        self.blocks = nn.ModuleList(
            [
                TverskyBlock(
                    d_model,
                    num_heads,
                    num_features,
                    model_type,
                    intersection_reduction,
                    difference_type,
                )
                for _ in range(num_layers)
            ]
        )
        self.ln_f = nn.LayerNorm(d_model)

        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        self.lm_head.weight = self.token_emb.weight

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        _, T = idx.size()
        mask = torch.tril(torch.ones(T, T, device=idx.device)).view(1, 1, T, T)
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.token_emb(idx) + self.pos_emb(pos)

        for block in self.blocks:
            x = block(x, mask=mask)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

        return logits, loss
