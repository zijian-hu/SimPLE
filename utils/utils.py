import torch
import numpy as np

import logging

# for type hint
from typing import Union, Dict, Any, Set, Optional
from torch import Tensor


def str_to_bool(input_str: Union[str]) -> Union[str, bool]:
    """
    If input_str is "True" or "False" (case insensitive and allows spaces on the side). Else, return the input_str

    Args:
        input_str:

    Returns: a boolean value if input_str is "True" or "False"; else, return the input string

    """
    comp_str = input_str.lower().strip()

    if comp_str == "true":
        return True
    elif comp_str == "false":
        return False
    else:
        return input_str


def get_device(device_id: str) -> torch.device:
    # update device
    if device_id != "cpu":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            logging.warning(f"device \"{device_id}\" is not available")
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")

    return device


def dict_add_prefix(input_dict: Dict[str, Any], prefix: str, separator: str = "/") -> Dict[str, Any]:
    return {f"{prefix}{separator}{key}": val for key, val in input_dict.items()}


def filter_dict(input_dict: Dict[str, Any], excluded_keys: Optional[Set[str]] = None,
                included_keys: Optional[Set[str]] = None) -> Dict[str, Any]:
    if excluded_keys is None:
        excluded_keys = set()

    if included_keys is not None:
        input_dict = {k: v for k, v in input_dict.items() if k in included_keys}

    return {k: v for k, v in input_dict.items() if k not in excluded_keys}


def detorch(inputs: Union[Tensor, np.ndarray, float, int, bool]) -> Union[np.ndarray, float, int, bool]:
    if isinstance(inputs, Tensor):
        outputs = inputs.detach().cpu().clone().numpy()
    else:
        outputs = inputs

    if isinstance(outputs, np.ndarray) and outputs.size == 1:
        outputs = outputs.item()

    return outputs
