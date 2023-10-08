from typing import Callable

from torch import Tensor


def shaplit(
    f: Callable[[Tensor], Tensor],
    cond: Callable[[Tensor, set, int], Tensor],
    x: Tensor,
    j: int,
    C: set,
    K: int,
    M: int = 1,
):
    """
    A function that implements the SHAPley Local Independence Test (SHAPLIT), as described
    in Sec. 3.1 of the paper.

    Parameters:
    -----------
    f: Callable
        the model to test.
    cond: Callable
        the conditional distribution to mask features.
    x: Tensor
        the sample to test the model on.
    C: set
        the set of features to condition on.
    j: int
        the index of the feature to test.
    K: int
        the number of null statistics to compute.
    M: int
        the number of reference samples to draw.
    """
    # check that the set of features `C` does not
    # contain the feature `j`
    assert j not in C

    # add the feature `j` to the set of features `C`
    C.add(j)
    # compute the test statistic
    y = f(cond(x, C, M)).mean()

    # remove the feature `j` from the set of features `C`
    C.remove(j)
    # sample references
    X_tilde = cond(x, C, K * M)
    # compute null statistics
    y_tilde = f(X_tilde)
    y_tilde = y_tilde.view(K, M).mean(dim=1)
    # compute the one-sided p-value
    p = (1 + ((y - y_tilde) <= 0).sum()) / (K + 1)
    return p
