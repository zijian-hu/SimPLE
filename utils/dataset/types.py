from .utils import BatchType, LoaderType, BatchGeneratorType

from typing import Dict, Union

DatasetDictType = Dict[str, Union[float, int]]

__all__ = [
    # types
    "BatchType",
    "LoaderType",
    "BatchGeneratorType",
    "DatasetDictType",
]
