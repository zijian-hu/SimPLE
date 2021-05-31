import numpy as np

from ..utils import detorch

# for type hint
from typing import List, Any, Dict, Optional, Union, FrozenSet


class LogAggregator:
    supported_reductions = {
        "mean": lambda x: np.mean(x).item(),
        "sum": lambda x: np.sum(x).item(),
    }

    def __init__(self):
        self.log_dict: Dict[str, Union[List[Any], Any]] = dict()
        self.plot_dict: Dict[str, Any] = dict()

    def __getitem__(self, k):
        return self.log_dict[k]

    @property
    def log_keys(self) -> FrozenSet[str]:
        return frozenset(self.log_dict.keys())

    @property
    def plot_keys(self) -> FrozenSet[str]:
        return frozenset(self.plot_dict.keys())

    def clear(self) -> None:
        self.log_dict.clear()
        self.plot_dict.clear()

    def add_log(self, log_info: Dict[str, Any]) -> None:
        conflict_keys = self.plot_keys.intersection(log_info.keys())
        assert len(conflict_keys) == 0, f"conflicting keys for plot and log data: {conflict_keys}"

        for k, v in log_info.items():
            if k not in self.log_dict:
                self.log_dict[k] = list()

            self.log_dict[k].append(detorch(v))

    def add_plot(self, plot_info: Dict[str, Any]) -> None:
        conflict_keys = self.log_keys.intersection(plot_info.keys())
        assert len(conflict_keys) == 0, f"conflicting keys for plot and log data: {conflict_keys}"

        for k, v in plot_info.items():
            self.plot_dict[k] = v

    def aggregate(self, reduction: str = "mean", key_mapping: Optional[Dict[Any, Any]] = None) -> Dict[str, Any]:
        assert reduction in self.supported_reductions, f"unsupported reduction method: {reduction}"

        if key_mapping is None:
            key_mapping = {}

        reduce_func = self.supported_reductions[reduction]

        output_log_dict = {key_mapping.get(k, k): reduce_func(v) for k, v in self.log_dict.items()}
        output_plot_dict = {key_mapping.get(k, k): v for k, v in self.plot_dict.items()}

        output_dict = output_log_dict
        output_plot_dict.update(output_plot_dict)

        return output_dict
