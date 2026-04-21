import torch
import torch.nn as nn
import torch.nn.functional as F

from src.enums import DifferenceType, IntersectionReductionType, ModelType


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
        feature_bank: nn.Parameter | None = None,
        model_type: ModelType | str = ModelType.RATIO,
        intersection_reduction: IntersectionReductionType | str = IntersectionReductionType.MIN,
        difference_type: DifferenceType | str = DifferenceType.SUBTRACT_MATCH,
    ) -> None:
        super().__init__()

        self.dim = dim
        self.num_features = num_features
        self.model_type = ModelType(model_type)
        self.intersection_reduction = IntersectionReductionType(intersection_reduction)
        self.difference_type = DifferenceType(difference_type)

        if feature_bank is not None:
            self.feature_bank = feature_bank
        else:
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

        return self.forward_from_projections(a_proj, b_proj)

    def forward_from_projections(self, a_proj: torch.Tensor, b_proj: torch.Tensor) -> torch.Tensor:
        """
        Альтернативный прямой проход, если входные данные уже проецированы на банк признаков.

        Args:
            a_proj (torch.Tensor): Проекция запросов. Форма: (..., num_features)
            b_proj (torch.Tensor): Проекция ключей. Форма: (..., num_features)

        Returns:
            torch.Tensor: Тензор мер сходства. Форма: (...)
        """
        a_pos_mask = (a_proj > 0).float()
        b_pos_mask = (b_proj > 0).float()

        intersect_mask = a_pos_mask * b_pos_mask
        inter_vals = self._reduce_intersection(a_proj, b_proj)
        f_intersect = torch.sum(inter_vals * intersect_mask, dim=-1)

        f_a_minus_b = self._compute_difference(a_proj, b_proj, a_pos_mask, b_pos_mask)
        f_b_minus_a = self._compute_difference(b_proj, a_proj, b_pos_mask, a_pos_mask)

        match self.model_type:
            case ModelType.CONTRAST:
                return self.theta * f_intersect - self.alpha * f_a_minus_b - self.beta * f_b_minus_a
            case ModelType.RATIO:
                denominator = (
                    f_intersect + self.alpha * f_a_minus_b + self.beta * f_b_minus_a + self.theta
                )
                return f_intersect / denominator

    def _reduce_intersection(self, a_proj: torch.Tensor, b_proj: torch.Tensor) -> torch.Tensor:
        """Агрегация общих признаков на основе выбранного метода Ψ."""
        match self.intersection_reduction:
            case IntersectionReductionType.MIN:
                return torch.minimum(a_proj, b_proj)
            case IntersectionReductionType.MAX:
                return torch.maximum(a_proj, b_proj)
            case IntersectionReductionType.PRODUCT:
                return a_proj * b_proj
            case IntersectionReductionType.MEAN:
                return (a_proj + b_proj) / 2.0
            case IntersectionReductionType.GMEAN:
                return torch.sqrt(torch.relu(a_proj * b_proj) + 1e-7)
            case IntersectionReductionType.SOFTMIN:
                stacked = torch.stack([a_proj, b_proj], dim=-1)
                weights = F.softmin(stacked, dim=-1)
                return torch.sum(stacked * weights, dim=-1)

    def _compute_difference(
        self,
        a_proj: torch.Tensor,
        b_proj: torch.Tensor,
        a_pos_mask: torch.Tensor,
        b_pos_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Вычисление разности множеств."""
        # Строгое игнорирование: признак есть в A, но нет в B
        ignorematch_mask = a_pos_mask * (b_proj <= 0).float()
        f_i = torch.sum(a_proj * ignorematch_mask, dim=-1)

        match self.difference_type:
            case DifferenceType.IGNORE_MATCH:
                return f_i
            case DifferenceType.SUBTRACT_MATCH:
                # Вычитание совпадений: признкак есть в обоих объектах, но в A он выражены сильнее
                subtract_mask = a_pos_mask * b_pos_mask * (a_proj > b_proj).float()
                f_s = f_i + torch.sum((a_proj - b_proj) * subtract_mask, dim=-1)
                return f_s
