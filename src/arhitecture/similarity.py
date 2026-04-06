from enum import StrEnum

import torch
import torch.nn as nn
import torch.nn.functional as F


class ModelType(StrEnum):
    CONTRAST = "contrast"
    RATIO = "ratio"


class IntersectionReduction(StrEnum):
    MIN = "min"
    MAX = "max"
    PRODUCT = "product"
    MEAN = "mean"
    GMEAN = "gmean"
    SOFTMIN = "softmin"


class DifferenceType(StrEnum):
    IGNORE_MATCH = "ignorematch"
    SUBTRACT_MATCH = "subtractmatch"


class TverskySimilarity(nn.Module):
    """
    Вычисляет дифференцируемую меру сходства Тверски между двумя наборами входных векторов на основе обучаемого банка признаков.
    """

    def __init__(
        self,
        dim: int,
        num_features: int,
        alpha: float = 0.5,
        beta: float = 0.5,
        model_type: ModelType | str = ModelType.RATIO,
        intersection_reduction: IntersectionReduction | str = IntersectionReduction.MIN,
        difference_type: DifferenceType | str = DifferenceType.SUBTRACT_MATCH,
    ) -> None:
        super().__init__()

        self.dim = dim
        self.num_features = num_features
        self.model_type = ModelType(model_type)
        self.intersection_reduction = IntersectionReduction(intersection_reduction)
        self.difference_type = DifferenceType(difference_type)

        self.feature_bank = nn.Parameter(torch.empty(dim, num_features))
        nn.init.xavier_uniform_(self.feature_bank)

        self.alpha = nn.Parameter(torch.tensor(alpha))
        self.beta = nn.Parameter(torch.tensor(beta))

        if self.model_type == ModelType.CONTRAST:
            self.theta = nn.Parameter(torch.tensor(1.0))
        else:
            self.theta = nn.Parameter(torch.tensor(1e-7))

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход для вычисления сходства Тверски.

        Args:
            a (torch.Tensor): Тензор запросов. Форма: (..., dim)
            b (torch.Tensor): Тензор ключей. Форма: (..., dim)

        Returns:
            torch.Tensor: Тензор мер сходства. Форма: (...)
        """
        a_proj = a @ self.feature_bank
        b_proj = b @ self.feature_bank

        a_pos_mask = (a_proj > 0).float()
        b_pos_mask = (b_proj > 0).float()

        intersect_mask = a_pos_mask * b_pos_mask
        inter_vals = self._reduce_intersection(a_proj, b_proj)
        f_intersect = torch.sum(inter_vals * intersect_mask, dim=-1)

        f_a_minus_b = self._compute_difference(a_proj, b_proj, a_pos_mask, b_pos_mask)
        f_b_minus_a = self._compute_difference(b_proj, a_proj, b_pos_mask, a_pos_mask)

        if self.model_type == ModelType.CONTRAST:
            return (
                self.theta * f_intersect
                - self.alpha * f_a_minus_b
                - self.beta * f_b_minus_a
            )
        elif self.model_type == ModelType.RATIO:
            denominator = (
                f_intersect
                + self.alpha * f_a_minus_b
                + self.beta * f_b_minus_a
                + self.theta
            )
            return f_intersect / denominator
        raise ValueError(f"Unknown model_type: {self.model_type}")

    def _reduce_intersection(
        self, a_proj: torch.Tensor, b_proj: torch.Tensor
    ) -> torch.Tensor:
        """Агрегация общих признаков на основе выбранного метода Ψ."""
        match self.intersection_reduction:
            case IntersectionReduction.MIN:
                return torch.minimum(a_proj, b_proj)
            case IntersectionReduction.MAX:
                return torch.maximum(a_proj, b_proj)
            case IntersectionReduction.PRODUCT:
                return a_proj * b_proj
            case IntersectionReduction.MEAN:
                return (a_proj + b_proj) / 2.0
            case IntersectionReduction.GMEAN:
                return torch.sqrt(torch.relu(a_proj * b_proj) + 1e-7)
            case IntersectionReduction.SOFTMIN:
                stacked = torch.stack([a_proj, b_proj], dim=-1)
                weights = F.softmin(stacked, dim=-1)
                return torch.sum(stacked * weights, dim=-1)

    def _compute_difference(
        self,
        x_proj: torch.Tensor,
        y_proj: torch.Tensor,
        x_pos_mask: torch.Tensor,
        y_pos_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Вычисление разности множеств (признаки, присутствующие в X, но отсутствующие или более слабые в Y).
        """
        # Строгое игнорирование: признак есть в X (>0), но нет в Y (<=0)
        ignorematch_mask = x_pos_mask * (y_proj <= 0).float()
        f_i = torch.sum(x_proj * ignorematch_mask, dim=-1)

        if self.difference_type == DifferenceType.IGNORE_MATCH:
            return f_i
        elif self.difference_type == DifferenceType.SUBTRACT_MATCH:
            # Вычитание совпадений: добавляем признаки, которые есть в обоих объектах, но в X они выражены сильнее
            subtract_mask = x_pos_mask * y_pos_mask * (x_proj > y_proj).float()
            f_s = f_i + torch.sum((x_proj - y_proj) * subtract_mask, dim=-1)
            return f_s
        raise ValueError(f"Unsupported difference type: {self.difference_type}")
