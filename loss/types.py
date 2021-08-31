# for type hint
from typing import Union, Dict
from torch import Tensor
from plotly.graph_objects import Figure
from wandb import Histogram

from .utils import (bha_coeff, bha_coeff_distance, l2_distance)
from .loss import softmax_cross_entropy_loss, bha_coeff_loss, l2_dist_loss

LogDictType = Dict[str, Tensor]
PlotDictType = Dict[str, Union[Figure, Histogram]]
LossInfoType = Union[Dict[str, Union[LogDictType, PlotDictType]], LogDictType]

SimilarityType = Union[bha_coeff]
DistanceType = Union[bha_coeff_distance, l2_distance]
DistanceLossType = Union[softmax_cross_entropy_loss, l2_dist_loss, bha_coeff_loss]

__all__ = [
    "LogDictType",
    "PlotDictType",
    "LossInfoType",
    "SimilarityType",
    "DistanceType",
    "DistanceLossType",
]
