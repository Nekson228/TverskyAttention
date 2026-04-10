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

        self.similarity = TverskySimilarity(
            dim=in_features,
            num_features=num_features,
            model_type=model_type,
            intersection_reduction=intersection_reduction,
            difference_type=difference_type,
        )

        self.prototypes = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.xavier_uniform_(self.prototypes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Входной тензор формы (batch_size, in_features)
        Returns:
            Тензор логитов сходства формы (batch_size, out_features)
        """
        # Расширяем размерности для попарного сравнения каждого объекта в батче с каждым прототипом
        # (batch_size, out_features, in_features)
        x_expanded = x.unsqueeze(1).expand(-1, self.out_features, -1)

        # (batch_size, out_features, in_features)
        prot_expanded = self.prototypes.unsqueeze(0).expand(x.shape[0], -1, -1)

        # (batch_size, out_features)
        logits = self.similarity(x_expanded, prot_expanded)
        return logits
