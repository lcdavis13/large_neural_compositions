import torch


def normed_log(x, eps=1e-6):
    """
    Given an epsilon smaller than most (or all) nonzero values in the dataset, this will return the data in logscale but with zero fixed at zero and one shifted slightly to 1+1/epsilon, and monotonically increasing.
    Using unnecessarily small epsilon can create numerical instability. It doesn't need to be strictly smaller than every value in the dataset. But if data points are comparable or smaller than epsilon, they won't benefit from the improved "resolution" of logscale and will trend linearly to zero.
    """
    return torch.log1p(x / eps) / -torch.log(torch.tensor(eps))

