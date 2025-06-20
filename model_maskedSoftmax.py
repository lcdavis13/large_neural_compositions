import torch

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