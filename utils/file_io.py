import numpy as np

from pathlib import Path
import re

# for type hint
from typing import Union, Optional, Pattern, List


def find_all_files(checkpoint_dir: Union[str, Path], search_pattern: Union[Pattern, str]) -> List[Path]:
    checkpoint_dir_path = Path(checkpoint_dir)

    return [file_item for file_item in checkpoint_dir_path.iterdir()
            if file_item.is_file() and re.search(pattern=search_pattern, string=file_item.name) is not None]


def find_checkpoint_path(checkpoint_dir: Union[str, Path],
                         step_filter: Union[Pattern, str],
                         return_full_path: bool = True) -> Optional[str]:
    checkpoint_dir_path = Path(checkpoint_dir)
    filename = None
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
            filename = file_item.name

    if return_full_path:
        if filename is not None:
            return str(checkpoint_dir_path / filename)
    return filename
