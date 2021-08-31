from abc import ABC, abstractmethod

# for type hint
from typing import Dict, Any, Set, Tuple, Optional


class RampUp(ABC):
    def __init__(self, length: int, current: int = 0):
        self.current = current
        self.length = length

    @abstractmethod
    def __call__(self, current: Optional[int] = None, is_step: bool = True) -> float:
        pass

    def state_dict(self) -> Dict[str, Any]:
        return dict(current=self.current, length=self.length)

    def load_state_dict(self, state_dict: Dict[str, Any], strict: bool = True):
        if strict:
            is_equal, incompatible_keys = self._verify_state_dict(state_dict)
            assert is_equal, f"loaded state dict contains incompatible keys: {incompatible_keys}"

        # for attr_name, attr_value in state_dict.items():
        #     if attr_name in self.__dict__:
        #         self.__dict__[attr_name] = attr_value

        self.current = state_dict["current"]
        self.length = state_dict["length"]

    def _verify_state_dict(self, state_dict: Dict[str, Any]) -> Tuple[bool, Set[str]]:
        self_keys = set(self.__dict__.keys())
        state_dict_keys = set(state_dict.keys())
        incompatible_keys = self_keys.union(state_dict_keys) - self_keys.intersection(state_dict_keys)
        is_equal = (len(incompatible_keys) == 0)

        return is_equal, incompatible_keys

    def _update_step(self, is_step: bool):
        if is_step:
            self.current += 1


class LinearRampUp(RampUp):
    def __call__(self, current: Optional[int] = None, is_step: bool = True) -> float:
        if current is not None:
            self.current = current

        if self.current >= self.length:
            ramp_up = 1.0
        else:
            ramp_up = self.current / self.length

        self._update_step(is_step)

        return ramp_up


def get_ramp_up(ramp_up_type: str, length: int, current: int = 0) -> RampUp:
    return {
        "linear": lambda: LinearRampUp(length, current),
    }[ramp_up_type]()
