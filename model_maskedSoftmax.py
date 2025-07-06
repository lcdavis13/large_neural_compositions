import torch
import torch.nn as nn

class MaskedSoftmax(nn.Module):
    def __init__(self, dim: int = -1, eps: float = 1e-13):
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x, mask_source):
        mask = (mask_source > 0)

        x_max = torch.max(x.masked_fill(~mask, float('-inf')), dim=self.dim, keepdim=True).values
        x_exp = torch.exp(x - x_max) * mask.float() # Subtract max for numerical stability, mask after exponentiation
        sum_exp = x_exp.sum(dim=self.dim, keepdim=True) + self.eps
        return x_exp / sum_exp


class WeightedSoftmax(nn.Module):
    """
    Softmax with multiplicative weights and numerical stabilization.

    Args:
        dim: Dimension along which to apply the softmax.
        eps: Small constant to prevent division by zero.

    Inputs:
        S: Tensor of any shape.
        x: Tensor of shape broadcastable to S, with the same size as S along `dim`.

    Returns:
        A: Weighted softmax result, same shape as S.
    """
    def __init__(self, dim: int = -1, eps: float = 1e-13):
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, S: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        dim = self.dim if self.dim >= 0 else S.ndim + self.dim

        if x.shape[dim] != S.shape[dim]:
            raise ValueError(
                f"x must match S along softmax dim={dim}. "
                f"x.shape[{dim}] = {x.shape[dim]}, S.shape[{dim}] = {S.shape[dim]}"
            )

        # Stabilized exponentiation
        max_S = S.max(dim=dim, keepdim=True).values
        S_stable = S - max_S
        exp_S = torch.exp(S_stable)

        numerator = exp_S * x
        denominator = numerator.sum(dim=dim, keepdim=True) + self.eps

        return numerator / denominator

 