import math
import torch.nn as nn

class LinearKH(nn.Module):
    """Wrapper for pytorch Linear that uses Kaiming-He initialization"""
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.lin = nn.Linear(in_features, out_features, bias=bias)
        nn.init.kaiming_uniform_(self.lin.weight, nonlinearity="relu")
        if self.lin.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.lin.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.lin.bias, -bound, bound)

    def forward(self, x):
        return self.lin(x)
