from enum import StrEnum


class ModelType(StrEnum):
    CONTRAST = "contrast"
    RATIO = "ratio"


class IntersectionReductionType(StrEnum):
    MIN = "min"
    MAX = "max"
    PRODUCT = "product"
    MEAN = "mean"
    GMEAN = "gmean"
    SOFTMIN = "softmin"


class DifferenceType(StrEnum):
    IGNORE_MATCH = "ignorematch"
    SUBTRACT_MATCH = "subtractmatch"
