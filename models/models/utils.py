# for type hint
from typing import Union

from torch.nn import Module, DataParallel
from torch.nn.parallel import DistributedDataParallel

ModelType = Union[Module, DataParallel, DistributedDataParallel]


def unwrap_model(model: ModelType) -> Module:
    if hasattr(model, "module"):
        return model.module
    else:
        return model


__all__ = [
    "unwrap_model",
]
