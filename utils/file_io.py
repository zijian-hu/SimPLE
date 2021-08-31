import numpy as np
import yaml

from pathlib import Path
import re

# for type hint
from typing import Union, Pattern, Optional, Dict, Any, List


def find_checkpoint_path(checkpoint_dir: Union[str, Path], step_filter: Union[Pattern, str]) -> Optional[Path]:
    checkpoint_dir_path = Path(checkpoint_dir)
    output_file = None
    max_step_num = -np.inf

    for file_item in checkpoint_dir_path.iterdir():
        if not file_item.is_file():
            continue

        search_result = re.search(step_filter, file_item.name)
        if search_result is None:
            continue

        step_num = int(search_result.group(1))
        if step_num > max_step_num:
            max_step_num = step_num
            output_file = file_item

    return output_file


def read_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def find_all_files(checkpoint_dir: Union[str, Path], search_pattern: Union[Pattern, str]) -> List[Path]:
    checkpoint_dir_path = Path(checkpoint_dir)

    return [file_item for file_item in checkpoint_dir_path.iterdir()
            if file_item.is_file() and re.search(pattern=search_pattern, string=file_item.name) is not None]
