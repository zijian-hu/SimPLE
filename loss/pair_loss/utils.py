import torch

# for type hint
from torch import Tensor


def get_pair_indices(inputs: Tensor, ordered_pair: bool = False) -> Tensor:
    """
    Get pair indices between each element in input tensor

    Args:
        inputs: input tensor
        ordered_pair: if True, will return ordered pairs. (e.g. both inputs[i,j] and inputs[j,i] are included)

    Returns: a tensor of shape (K, 2) where K = choose(len(inputs),2) if ordered_pair is False.
        Else K = 2 * choose(len(inputs),2). Each row corresponds to two indices in inputs.

    """
    indices = torch.combinations(torch.tensor(range(len(inputs))), r=2)

    if ordered_pair:
        # make pairs ordered (e.g. both (0,1) and (1,0) are included)
        indices = torch.cat((indices, indices[:, [1, 0]]), dim=0)

    return indices
