from .log_aggregator import LogAggregator
from ..utils import dict_add_prefix

# for type hint
from typing import Any, Dict, Optional, Union, Tuple
from argparse import Namespace

from torch.nn import Module
from plotly.graph_objects import Figure
from wandb import Histogram


class Logger:
    def __init__(self,
                 log_dir: str,
                 config: Union[Namespace, Dict[str, Any]],
                 *args,
                 log_info_key_map: Optional[Dict[str, str]] = None,
                 **kwargs):
        self.log_dir = log_dir
        self.config = config

        # hook functions
        self.log_hooks = []

        self.metric_smooth_record: Dict[str, Dict[str, Union[str, Optional[float]]]] = {
            "train/mean_acc": {
                "key": "train/smoothed_acc",
                "value": None,
            },
            "unlabeled/mean_acc": {
                "key": "unlabeled/smoothed_acc",
                "value": None,
            },
            "validation/mean_acc": {
                "key": "validation/smoothed_acc",
                "value": None,
            },
            "test/mean_acc": {
                "key": "test/smoothed_acc",
                "value": None,
            },
        }

        self.smoothing_weight = 0.9

        # init key map
        if log_info_key_map is None:
            log_info_key_map = dict()

        self.log_info_key_map = log_info_key_map

        # init log accumulator
        self.log_aggregator = LogAggregator()

    def log(self, log_info: Dict[str, Any], *args, **kwargs):
        pass

    def watch(self, model: Module, *args, **kwargs):
        pass

    def save(self, output_path: str):
        pass

    def process_log_info(self,
                         log_info: Dict[str, Any],
                         *args,
                         prefix: Optional[str] = None,
                         log_info_override: Optional[Dict[str, Any]] = None,
                         **kwargs) -> Dict[str, Any]:
        if log_info_override is None:
            log_info_override = {}

        # create shallow copy
        log_info = dict(log_info)

        # apply override
        log_info.update(log_info_override)

        # update keys based on key map
        for key in list(log_info.keys()):
            if key in self.log_info_key_map:
                new_key = self.log_info_key_map[key]
                log_info[new_key] = log_info.pop(key)

        if bool(prefix):
            # prepend prefix to info_dict keys
            log_info = dict_add_prefix(log_info, prefix)

        # apply smoothing
        smoothed_metrics = self.smooth_metrics(log_info)
        log_info.update(smoothed_metrics)

        return log_info

    def register_log_hook(self, func, *args, **kwargs):
        self.log_hooks.append([func, args, kwargs])

    def call_log_hooks(self, log_info: Dict[str, Any]):
        for (func, args, kwargs) in self.log_hooks:
            func(log_info=log_info, *args, **kwargs)

    @staticmethod
    def separate_plot(input_dict: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        log_info: Dict[str, Any] = dict()
        plot_info: Dict[str, Any] = dict()

        for k, v in input_dict.items():
            if isinstance(v, Figure) or isinstance(v, Histogram):
                plot_info[k] = v
            else:
                log_info[k] = v

        return log_info, plot_info

    def smooth_metrics(self, log_info: Dict[str, Any]) -> Dict[str, Any]:
        # see https://stackoverflow.com/a/49357445/5838091
        smoothed_metrics = dict()

        for key, value in log_info.items():
            if key not in self.metric_smooth_record:
                continue

            smoothed_metric_dict = self.metric_smooth_record[key]
            if smoothed_metric_dict["value"] is None:
                smoothed_metric_dict["value"] = value
            else:
                smoothed_metric_dict["value"] = smoothed_metric_dict["value"] * self.smoothing_weight + \
                                                (1 - self.smoothing_weight) * value

            smoothed_metrics[smoothed_metric_dict["key"]] = smoothed_metric_dict["value"]

        return smoothed_metrics

    def accumulate_log(self, log_info: Optional[Dict[str, Any]] = None, plot_info: Optional[Dict[str, Any]] = None):
        if log_info is not None:
            self.log_aggregator.add_log(log_info)

        if plot_info is not None:
            self.log_aggregator.add_plot(plot_info)

    def aggregate_log(self, reduction: str = "mean") -> Dict[str, Any]:
        return self.log_aggregator.aggregate(reduction=reduction)

    def reset_aggregator(self):
        self.log_aggregator.clear()
