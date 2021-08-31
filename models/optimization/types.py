from typing import Union, Iterable, Dict
from torch.optim.lr_scheduler import LambdaLR, StepLR
from torch.nn import Parameter

LRSchedulerType = Union[LambdaLR, StepLR]
ParametersType = Iterable[Parameter]
ParametersGroupType = Iterable[Dict[str, Union[Parameter, float, int]]]
OptimizerParametersType = Union[ParametersType, ParametersGroupType]

__all__ = [
    "LRSchedulerType",
    "ParametersType",
    "ParametersGroupType",
    "OptimizerParametersType",
]
