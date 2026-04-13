import torch
import torch.nn as nn
import math
from typing import Optional

from src.arhitecture import TverskyProjection
from src.enums import ModelType, IntersectionReductionType, DifferenceType


class TverskyMultiHeadAttentionDropIn(nn.Module):
    """
    Multi-Head Self-Attention, где проекции Q, K, V и O реализованы через слои Тверски с общим банком признаков.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_features: int,
        shared_omega: Optional[nn.Parameter] = None,
        model_type: ModelType | str = ModelType.CONTRAST,
        intersection_reduction: IntersectionReductionType | str = IntersectionReductionType.PRODUCT,
        difference_type: DifferenceType | str = DifferenceType.SUBTRACT_MATCH,
    ) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model должно быть кратно num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        if shared_omega is None:
            self.omega = nn.Parameter(torch.empty(d_model, num_features))
            nn.init.xavier_uniform_(self.omega)
        else:
            self.omega = shared_omega

        self.q_proj = TverskyProjection(
            d_model, d_model, num_features, model_type, intersection_reduction, difference_type
        )
        self.k_proj = TverskyProjection(
            d_model, d_model, num_features, model_type, intersection_reduction, difference_type
        )
        self.v_proj = TverskyProjection(
            d_model, d_model, num_features, model_type, intersection_reduction, difference_type
        )
        self.o_proj = TverskyProjection(
            d_model, d_model, num_features, model_type, intersection_reduction, difference_type
        )

        self._tie_feature_banks()

    def _tie_feature_banks(self) -> None:
        self.q_proj.similarity.feature_bank = self.omega
        self.k_proj.similarity.feature_bank = self.omega
        self.v_proj.similarity.feature_bank = self.omega
        self.o_proj.similarity.feature_bank = self.omega

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x: Входной тензор формы (batch_size, seq_length, d_model)
            mask: Опциональная маска внимания формы (batch_size, 1, seq_length, seq_length)
        Returns:
            torch.Tensor: Выходной тензор формы (batch_size, seq_length, d_model)
        """
        batch_size, seq_length, _ = x.size()

        # (batch_size, seq_length, d_model)
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # (batch_size, num_heads, seq_length, d_k)
        Q = Q.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)

        # (batch_size, num_heads, seq_length, seq_length)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attention_weights = torch.softmax(scores, dim=-1)

        context = torch.matmul(attention_weights, V)

        # (batch_size, seq_length, d_model)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
        output = self.o_proj(context)

        return output
