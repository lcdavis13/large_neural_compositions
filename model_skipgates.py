import torch
import torch.nn as nn
    
class ZeroGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(0.0))

    def forward(self, h):
        return self.a * h

class SkipGateBase(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, h, x0):
        raise NotImplementedError("Subclasses must implement the forward method.")

class StaticSkip(SkipGateBase):
    def __init__(self):
        super().__init__()

    def forward(self, h, x0):
        return h + x0
    
# class StaticBlendSkip(SkipGateBase):
#     def __init__(self):
#         super().__init__()

#     def forward(self, h, x0):
#         return (h + x0)*0.5

class GateSkip(SkipGateBase):
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(0.0))
        self.b = nn.Parameter(torch.tensor(1.0))

    def forward(self, h, x0):
        return self.a * h + self.b * x0

class BlendSkip(SkipGateBase):
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(0.0))

    def forward(self, h, x0):
        return self.a * h + (1.0 - self.a) * x0

class RezeroSkip(SkipGateBase):
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(0.0))

    def forward(self, h, x0):
        return self.a * h + x0
