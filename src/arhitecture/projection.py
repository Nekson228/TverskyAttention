import torch
import torch.nn as nn

from src.enums import ModelType, IntersectionReductionType, DifferenceType
from src.arhitecture import TverskySimilarity


class TverskyProjection(nn.Module):
    """
    Проекционный слой Тверски.
    Сравнивает входные векторы с набором обучаемых прототипов.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_features: int,
        model_type: ModelType | str = ModelType.CONTRAST,
        intersection_reduction: IntersectionReductionType | str = IntersectionReductionType.MIN,
        difference_type: DifferenceType | str = DifferenceType.SUBTRACT_MATCH,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_features = num_features

        self.similarity = TverskySimilarity(
            dim=in_features,
            num_features=num_features,
            model_type=model_type,
            intersection_reduction=intersection_reduction,
            difference_type=difference_type,
        )

        self.prototypes = nn.Parameter(torch.empty(out_features, num_features))
        nn.init.xavier_uniform_(self.prototypes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Входной тензор формы (..., in_features)
        Returns:
            Тензор логитов сходства формы (..., out_features)
        """
        leading_dims = x.shape[:-1]

        x_proj = x @ self.similarity.feature_bank  # (..., num_features)

        # Расширяем размерности для попарного сравнения каждого объекта в батче с каждым прототипом
        # (..., 1, num_features) -> (..., out_features, num_features)
        x_proj_expanded = x_proj.unsqueeze(-2).expand(*leading_dims, self.out_features, self.num_features)

        # (..., out_features, num_features)
        prot_expanded = self.prototypes.unsqueeze(0).expand(*leading_dims, self.out_features, self.num_features)

        # (..., out_features)
        logits = self.similarity.forward_from_projections(x_proj_expanded, prot_expanded)
        return logits
