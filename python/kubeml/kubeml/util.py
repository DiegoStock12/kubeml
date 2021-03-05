import math
from typing import List

import torch.nn as nn

# Number of datapoints in storage on average
STORAGE_SUBSET_SIZE = 64


def is_optimizable(layer: nn.Module) -> bool:
    """Should save layer returns just whether the layer is optimizable or not
    and thus if it should be sent to the parameter server"""
    t = str(type(layer))
    if 'conv' in t or 'linear' in t:
        return True
    return False


def split_minibatches(a: range, n: int) -> List[range]:
    """
    Based on the number of minibatches return the ones assigned to each
    function so that the count is approximately the same

    :arg a range with the list of minibatches
    :arg n number of functions to divide the minibatches across
    :return: list with all the ranges, indexed by the funcId
    """
    k, m = divmod(len(a), n)
    return [a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]


def get_subset_period(K: int, batch_size: int, assigned_subsets: range) -> int:
    """
    Calculates the number of subsets that will be evaluated per iteration
    to fulfill the K-avg sync.

    Calculates the number of datapoints before sync, and from there calculates
    the number of subsets for those

    :param K: Number of forward passes to be done
    :param batch_size: size of the batch
    :param assigned_subsets: the data subsets assigned to this function

    :return: the number of subsets that must be loaded per iteration
    """

    # if K is -1, it means that we process all the data
    # assigned to us before syncing after a full epoch, so
    # return the length of the subsets
    if K == -1:
        return len(assigned_subsets)

    # calculate number of datapoints in K passes and divide to get the number of subsets
    return int(math.ceil((batch_size * K) / STORAGE_SUBSET_SIZE))
