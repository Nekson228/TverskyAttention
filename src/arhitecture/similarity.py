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
        f_intersect = self._compute_intersection(a_proj, b_proj)
        f_a_minus_b = self._compute_difference(a_proj, b_proj)
        f_b_minus_a = self._compute_difference(b_proj, a_proj)

        match self.model_type:
            case ModelType.CONTRAST:
                return self.theta * f_intersect - self.alpha * f_a_minus_b - self.beta * f_b_minus_a
            case ModelType.RATIO:
                denominator = (
                    f_intersect + self.alpha * f_a_minus_b + self.beta * f_b_minus_a + self.theta
                )
                return f_intersect / denominator

    def _compute_intersection(self, a_proj: torch.Tensor, b_proj: torch.Tensor) -> torch.Tensor:
        a_pos = F.relu(a_proj)
        b_pos = F.relu(b_proj)

        inter_vals = self._reduce_intersection(a_pos, b_pos)
        return torch.sum(inter_vals, dim=-1)

    def _reduce_intersection(self, a_pos: torch.Tensor, b_pos: torch.Tensor) -> torch.Tensor:
        """Агрегация общих признаков на основе выбранного метода Ψ."""
        match self.intersection_reduction:
            case IntersectionReductionType.MIN:
                return torch.minimum(a_pos, b_pos)
            case IntersectionReductionType.MAX:
                return torch.maximum(a_pos, b_pos)
            case IntersectionReductionType.PRODUCT:
                return a_pos * b_pos
            case IntersectionReductionType.MEAN:
                return (a_pos + b_pos) / 2.0
            case IntersectionReductionType.GMEAN:
                return torch.sqrt(torch.relu(a_pos * b_pos) + 1e-7)
            case IntersectionReductionType.SOFTMIN:
                stacked = torch.stack([a_pos, b_pos], dim=-1)
                weights = F.softmin(stacked, dim=-1)
                return torch.sum(stacked * weights, dim=-1)

    def _compute_difference(self, a_proj: torch.Tensor, b_proj: torch.Tensor) -> torch.Tensor:
        """
        Вычисление разности множеств (признаки, присутствующие в X, но отсутствующие или более слабые в Y).
        """
        # Строгое игнорирование: признак есть в X (>0), но нет в Y (<=0)
        a_pos_mask = a_proj > 0
        b_neg_mask = b_proj <= 0
        ignorematch_mask = a_pos_mask & b_neg_mask
        f_i = F.relu(a_proj) * ignorematch_mask.float()
        match self.difference_type:
            case DifferenceType.IGNORE_MATCH:
                result = f_i
            case DifferenceType.SUBTRACT_MATCH:
                # Вычитание совпадений: добавляем признаки, которые есть в обоих объектах, но в X они выражены сильнее
                diff = a_proj - b_proj
                both_positive_mask = a_pos_mask & (~b_neg_mask)
                result = f_i + F.relu(diff) * both_positive_mask.float()
        return result.sum(dim=-1)
