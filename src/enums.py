from enum import Enum


class ModelType(str, Enum):
    CONTRAST = "contrast"
    RATIO = "ratio"


class IntersectionReductionType(str, Enum):
    MIN = "min"
    MAX = "max"
    PRODUCT = "product"
    MEAN = "mean"
    GMEAN = "gmean"
    SOFTMIN = "softmin"


class DifferenceType(str, Enum):
    IGNORE_MATCH = "ignorematch"
    SUBTRACT_MATCH = "subtractmatch"
