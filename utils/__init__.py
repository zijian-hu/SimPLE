import random
import numpy as np
import torch
from torch.backends import cudnn

from .cli import get_arg_parser, get_args, update_args, args_to_logger_config
from .dataset import get_dataset, repeater
from .file_io import find_checkpoint_path, read_yaml, find_all_files

from .timing import timing
from .loggers import Logger, LogAggregator
from .utils import str_to_bool, get_device, dict_add_prefix, filter_dict, detorch

from . import dataset
from . import cli
from . import loggers
from . import metrics
from . import types

# for type hint
from typing import Optional
from argparse import Namespace


def set_random_seed(seed: Optional[int], is_cudnn_deterministic: bool) -> None:
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    if is_cudnn_deterministic:
        cudnn.deterministic = True
        cudnn.benchmark = False


def get_logger(args: Namespace) -> Logger:
    logger_type = args.logger
    config_dict = args_to_logger_config(args)

    if logger_type == "wandb":
        from .loggers import WandbLogger
        return WandbLogger(
            log_dir=args.log_dir,
            config=config_dict,
            **args.logger_config_dict)
    elif logger_type == "nop":
        return Logger(
            log_dir=args.log_dir,
            config=config_dict)
    else:
        from .loggers import PrintLogger
        return PrintLogger(
            log_dir=args.log_dir,
            config=config_dict,
            is_display_plots=args.is_display_plots)


__all__ = [
    # modules
    "dataset",
    "cli",
    "loggers",
    "metrics",
    "types",

    # classes
    "LogAggregator",
    "Logger",

    # functions
    "get_arg_parser",
    "get_args",
    "update_args",
    "args_to_logger_config",
    "get_dataset",
    "repeater",
    "find_checkpoint_path",
    "read_yaml",
    "find_all_files",
    "timing",
    "get_logger",
    "str_to_bool",
    "get_device",
    "dict_add_prefix",
    "filter_dict",
    "detorch",
    "set_random_seed",
]
