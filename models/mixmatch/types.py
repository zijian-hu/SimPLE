from .mixmatch import mixmatch
from .simple_mixmatch import simple_mixmatch
from .mixmatch_enhanced import mixmatch_enhanced

from typing import Union

MixMatchFunctionType = Union[mixmatch, mixmatch_enhanced, simple_mixmatch]

__all__ = [
    "MixMatchFunctionType",
]
