from enum import Enum

# for type hint
from typing import Dict, Union, Optional, KeysView, Set, Any


class MetricMode(Enum):
    MAX = 0
    MIN = 1


MetricDictType = Dict[str, Union[str, MetricMode, int, float]]


class MetricMonitor:
    def __init__(self, metrics: Optional[Dict[str, MetricDictType]] = None):
        self.metrics: Dict[str, MetricDictType] = dict()

        if metrics is not None:
            self.update(metrics)

    def __getitem__(self, key: str) -> MetricDictType:
        return self.metrics[key]

    def __setitem__(self, key: str, value: MetricDictType):
        self.track(key=key,
                   log_key=value["key"],
                   mode=value["mode"],
                   best_value=value["best_value"])

    def __contains__(self, key: str) -> bool:
        return key in self.metrics

    def track(self,
              key: str,
              best_value: Union[float, int],
              mode: MetricMode,
              log_key: Optional[str] = None,
              prefix: Optional[str] = None):
        if log_key is None:
            log_key = f"best_{key}"

        if prefix is not None:
            key = f"{prefix}/{key}"
            log_key = f"{prefix}/{log_key}"

        if key in self.metrics:
            # if key exist, update best_value
            curr_best_value = self.metrics[key]["best_value"]

            if mode == MetricMode.MIN:
                best_value = min(best_value, curr_best_value)
            else:
                best_value = max(best_value, curr_best_value)

        self.metrics[key] = {
            "key": log_key,
            "best_value": best_value,
            "mode": mode,
        }

    def keys(self) -> KeysView[str]:
        return self.metrics.keys()

    def update(self, metrics: Dict[str, MetricDictType]):
        for key, metric in metrics.items():
            self[key] = metric

    def mutual_keys(self, keys: Union[Set[str], KeysView[str]]) -> Set[str]:
        return set(keys).intersection(set(self.metrics.keys()))

    def update_metrics(self, log_info: Dict[str, Any]) -> Dict[str, Union[int, float]]:
        updated_dict = {}

        for mutual_key in self.mutual_keys(log_info.keys()):
            metric_dict = self[mutual_key]
            new_value = log_info[mutual_key]

            mode: MetricMode = metric_dict["mode"]
            best_value = metric_dict["best_value"]

            if (mode == MetricMode.MAX and new_value > best_value) or \
                    (mode == MetricMode.MIN and new_value < best_value):
                metric_dict["best_value"] = new_value

                # save updated key and value
                updated_dict[mutual_key] = new_value

        return updated_dict

    def state_dict(self) -> Dict[str, Any]:
        return {k: v["best_value"] for k, v in self.metrics.items()}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.update_metrics(state_dict)
