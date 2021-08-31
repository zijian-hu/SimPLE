import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm

from itertools import repeat, combinations
from pathlib import Path

# for type hint
from typing import Optional, Generator, Tuple, Union, List, Sequence, Any
from torch import Tensor
from PIL.Image import Image
from torchvision.datasets import VisionDataset

BatchType = Union[Tuple[Tuple[Tensor, Tensor], ...], Tuple[Tensor, Tensor]]
LoaderType = Union[DataLoader, Generator[BatchType, None, None]]
BatchGeneratorType = Generator[Tuple[int, BatchType], None, None]


def repeater(data_loader: DataLoader) -> Generator[Tuple[Tensor, Tensor], None, None]:
    for loader in repeat(data_loader):
        for data in loader:
            yield data


def get_batch(loaders: Sequence[LoaderType], max_iter: int, is_repeat: bool = False) \
        -> BatchGeneratorType:
    if is_repeat:
        loaders = [repeater(loader) for loader in loaders]

    if len(loaders) == 1:
        combined_loaders = loaders[0]
    else:
        combined_loaders = zip(*loaders)

    for idx, batch in enumerate(combined_loaders):
        if idx >= max_iter:
            return
        yield idx, batch


def get_targets(dataset: Union[Dataset, VisionDataset]) -> List[Any]:
    if hasattr(dataset, 'targets'):
        return dataset.targets

    # TODO: handle very large dataset
    return [target for (_, target) in dataset]


def get_class_indices(dataset: Dataset) -> List[np.ndarray]:
    targets = np.asarray(get_targets(dataset))
    num_classes = max(targets) + 1

    return [np.where(targets == i)[0] for i in range(num_classes)]


def to_indices_or_sections(subset_sizes: Sequence[int]) -> List[int]:
    outputs = [sum(subset_sizes[:i]) + size for i, size in enumerate(subset_sizes)]

    return outputs


def safe_round(inputs: np.ndarray, target_sums: np.ndarray, min_value: int = 0) -> np.ndarray:
    """
    Round array while maintaining the sum. The difference is adjusted based on value
    distribution in input array

    Args:
        inputs: input numpy array
        target_sums: target sum values
        min_value: each element in the output will be at least min_value

    Returns: numpy array where each element is at least min_value

    """
    assert len(inputs) == len(target_sums)
    rounded = np.around(inputs).astype(int)
    rounded = np.maximum(rounded, min_value)

    outputs = np.zeros_like(inputs, dtype=int)

    for i in range(len(inputs)):
        # TODO: use more efficient implementation
        rounded_row = rounded[i]
        row_target_sum = target_sums[i]

        # subtract by min_value so that rounding adjustment do not affect the min
        row_outputs = rounded_row - min_value
        adjusted_target_sum = row_target_sum - (len(rounded_row) * min_value)

        round_error = adjusted_target_sum - row_outputs.sum()

        if round_error != 0:
            # repeat index by its corresponding value; this will make sure the sampling follows value distribution
            extended_idx = np.repeat(np.arange(row_outputs.size), row_outputs)
            selected_idx = np.random.choice(extended_idx, abs(round_error))

            unique_idx, idx_counts = np.unique(selected_idx, return_counts=True)
            row_outputs[unique_idx] += np.copysign(idx_counts, round_error).astype(int)

        assert row_outputs.sum() == adjusted_target_sum

        # add back the subtracted min_value
        row_outputs += min_value

        assert row_outputs.sum() == row_target_sum
        outputs[i] = row_outputs

    assert np.all(outputs.sum(axis=1) == target_sums)
    assert np.all(outputs >= min_value)
    return outputs


def per_class_random_split_by_ratio(dataset: Dataset,
                                    ratios: Sequence[float],
                                    num_classes: int,
                                    uneven_split: bool = False,
                                    min_value: int = 1) -> List[Dataset]:
    """Split the dataset base on ratios.

    Args:
        dataset: dataset to split
        ratios: fraction of data in each subset
        num_classes: number of classes in dataset
        uneven_split: if True, will return len(ratios) + 1 subsets where the size for the last subset is interpolated
        min_value: min value in each class of each subset

    Returns: if is_uneven_split = False, returns len(ratios) subsets where the ith subset has size ratio[i] *
        len(dataset); if is_uneven_split = True, returns len(ratios) + 1 subsets where the size for the last
        subset is interpolated.

    """
    ratios = list(ratios)
    if uneven_split:
        ratios.append(1. - sum(ratios))

    assert sum(ratios) == 1.
    ratios = np.asarray(ratios)
    assert np.all(ratios >= 0.)

    # shape (num_classes, # samples in this class); each row is the indices of samples in that class
    class_indices = get_class_indices(dataset)
    assert len(class_indices) == num_classes

    # shape (num_classes,); each element is the number of samples in that class
    class_lengths = np.asarray([len(class_idx) for class_idx in class_indices])
    assert class_lengths.sum() == len(dataset)

    # shape (num_subset, num_classes); each row is a list of class sizes for the corresponding subset
    subset_class_lengths = ratios.reshape(-1, 1) * class_lengths

    subset_class_lengths = safe_round(subset_class_lengths.T, target_sums=class_lengths, min_value=min_value)
    subset_class_lengths = subset_class_lengths.transpose()

    return per_class_random_split_helper(dataset=dataset,
                                         class_indices=class_indices,
                                         subset_class_lengths=subset_class_lengths)


def per_class_random_split(dataset: Dataset, lengths: Sequence[int], num_classes: int, uneven_split: bool = False) \
        -> List[Dataset]:
    """Split the dataset evenly across all classes

    Args:
        dataset: dataset to split
        lengths: length for each subset
        num_classes: number of classes in dataset
        uneven_split: if True, will return len(lengths) + 1 subsets where the size for the last subset may not be the
            same for all classes

    Returns: len(lengths) subsets where the ith subset has size lengths[i]; if is_uneven_split = True, will return
        len(lengths) + 1 subsets where the size for the last subset may not be the same for all classes.

    """
    # see https://github.com/pytorch/vision/issues/168#issuecomment-319659360
    # and https://github.com/pytorch/vision/issues/168#issuecomment-398734285 for detail
    total_length = sum(lengths)

    if uneven_split:
        assert 0 < total_length <= len(dataset), f"Expecting 0 < length <= {len(dataset)} but get {total_length}"
    else:
        if len(lengths) <= 1:
            return [dataset]

        assert total_length == len(dataset), "Sum of input lengths does not equal the length of the input dataset"

    subset_num_per_class, remainders = np.divmod(lengths, num_classes)
    assert np.all(remainders == 0), f"Subset sizes is not divisible by the number of classes ({num_classes})"

    # shape (num_classes, # samples in this class); each row is the indices of samples in that class
    class_indices = get_class_indices(dataset)
    assert len(class_indices) == num_classes

    # shape (num_classes,); each element is the number of samples in that class
    class_lengths = np.asarray([len(class_idx) for class_idx in class_indices])

    # shape (num_subset, num_classes); each row is the class lengths of this subset
    subset_class_lengths = np.tile(subset_num_per_class.reshape(-1, 1), num_classes)
    if uneven_split:
        # interpolate last subset's class lengths
        last_subset_class_lengths = class_lengths - subset_class_lengths.sum(axis=0)
        subset_class_lengths = np.vstack((subset_class_lengths, last_subset_class_lengths))

    return per_class_random_split_helper(dataset=dataset,
                                         class_indices=class_indices,
                                         subset_class_lengths=subset_class_lengths)


def per_class_random_split_helper(dataset: Dataset,
                                  class_indices: List[np.ndarray],
                                  subset_class_lengths: np.ndarray) -> List[Dataset]:
    """

    Args:
        dataset:
        class_indices: shape (num_classes, # samples in this class);
            each row is the indices of samples in that class
        subset_class_lengths: shape (num_subset, num_classes);
            each row is a list of class sizes for that subset

    Returns:

    """
    class_lengths = np.asarray([len(class_idx) for class_idx in class_indices])
    assert np.all(subset_class_lengths.sum(axis=0) == class_lengths)

    num_subsets, num_classes = subset_class_lengths.shape

    subset_indices = [list() for _ in range(num_subsets)]

    for i in range(num_classes):
        # ith column contains the class length for each subset
        subset_num_per_class = subset_class_lengths[:, i]

        indices = class_indices[i]
        np.random.shuffle(indices)

        indices_or_sections = to_indices_or_sections(subset_num_per_class)[:-1]
        class_subset_indices = np.split(indices, indices_or_sections)
        [subset_indices[i].extend(idx) for i, idx in enumerate(class_subset_indices)]

    # check if index subsets matches the desired lengths
    subset_lengths = subset_class_lengths.sum(axis=1)
    assert all(len(subset_idx) == subset_lengths[i] for i, subset_idx in enumerate(subset_indices))

    # check if index subsets in subset_indices are unique
    subset_idx_sets = [set(subset_idx) for subset_idx in subset_indices]
    assert all(len(idx_set) == len(subset_indices[i]) for i, idx_set in enumerate(subset_idx_sets))

    # check if index subsets are mutually exclusive
    combos = combinations(subset_idx_sets, 2)
    assert all(combo[0].isdisjoint(combo[1]) for combo in combos)

    return [Subset(dataset, subset_idx) for subset_idx in subset_indices]


def get_data_shape(data_point: Union[np.ndarray, Tensor, Image]):
    data_shape = np.asarray(data_point).shape
    if isinstance(data_point, Tensor) and len(data_shape) >= 3:
        # torch tensor has channel (C, H, W, ...), swap channel to (H, W, ..., C)
        data_shape = np.roll(data_shape, -1)

    return data_shape


def get_directory_size(dirname: Union[Path, str]) -> int:
    return sum(f.stat().st_size for f in Path(dirname).rglob("*") if f.is_file())
