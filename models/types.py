from typing import Union, Iterable, Dict
from torch.nn import Parameter

from .optimization.types import LRSchedulerType
from .mixmatch.types import MixMatchFunctionType

ParametersType = Iterable[Parameter]
ParametersGroupType = Iterable[Dict[str, Union[Parameter, float, int]]]
OptimizerParametersType = Union[ParametersType, ParametersGroupType]

__all__ = [
    "LRSchedulerType",
    "MixMatchFunctionType",

    "ParametersType",
    "ParametersGroupType",
    "OptimizerParametersType",
]