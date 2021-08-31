from .mixmatch import MixMatch
from .simple_mixmatch import SimPLE
from .mixmatch_base import MixMatchBase as MixMatchEnhanced

from typing import Union

MixMatchFunctionType = Union[MixMatch, MixMatchEnhanced, SimPLE]

__all__ = [
    "MixMatchFunctionType",
]
