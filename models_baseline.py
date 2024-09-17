import torch
import torch.nn as nn
from torchdiffeq import odeint
# from torchdiffeq import odeint_adjoint as odeint  # tiny memory footprint but it is intractible for large models such as cNODE2 with Waimea data


class SingleLayerPerceptron(nn.Module):
    def __init__(self, N):
        super(SingleLayerPerceptron, self).__init__()
        self.f = nn.Linear(N, N)
    
    def forward(self, t, x):
        return self.f(x)


class SingleLayerMultiplied(nn.Module):
    def __init__(self, N):
        super(SingleLayerMultiplied, self).__init__()
        self.f = nn.Linear(N, N)
    
    def forward(self, t, x): # x' = x*f(x)
        fx = self.f(x)  # B x N
        
        y = torch.mul(x, fx)  # B x N
        
        return y  # B x N


class SingleLayerSummed(nn.Module):
    def __init__(self, N):
        super(SingleLayerSummed, self).__init__()
        self.f = nn.Linear(N, N)
    
    def forward(self, t, x): # x' = x + f(x)
        fx = self.f(x)  # B x N
        
        y = x + fx  # B x N
        
        return y  # B x N


class SingleLayerMultipliedSummed(nn.Module):
    def __init__(self, N):
        super(SingleLayerMultipliedSummed, self).__init__()
        self.f = nn.Linear(N, N)
    
    def forward(self, t, x): # x' = x + x*f(x)
        fx = self.f(x)  # B x N
        
        y = torch.mul(x, fx)  # B x N
        
        return x + y  # B x N


class SingleLayerReplicator(nn.Module):
    def __init__(self, N):
        super(SingleLayerReplicator, self).__init__()
        self.f = nn.Linear(N, N)
    
    def forward(self, t, x): # x' = x + x*(f(x) - mean(f(x)))
        fx = self.f(x)  # B x N
        
        xT_fx = torch.sum(x * fx, dim=-1).unsqueeze(1)  # B x 1 (batched dot product)
        diff = fx - xT_fx  # B x N
        dxdt = torch.mul(x, diff)  # B x N
        
        return x + dxdt  # B x N


class ConstReplicator(nn.Module):
    def __init__(self, N):
        super(ConstReplicator, self).__init__()
        self.f = nn.Parameter(torch.randn(N))
    
    def forward(self, t, x):  # x' = x + x*(f(x) - mean(f(x)))
        fx = self.f.expand(x.size(0), -1)  # B x N
        
        xT_fx = torch.sum(x * fx, dim=-1).unsqueeze(1)  # B x 1 (batched dot product)
        diff = fx - xT_fx  # B x N
        dxdt = torch.mul(x, diff)  # B x N
        
        return x + dxdt  # B x N


class ODEFunc_cNODE0(nn.Module):  # cNODE where "F(x)" does not depend on x
    def __init__(self, N):
        super(ODEFunc_cNODE0, self).__init__()
        self.f = nn.Parameter(torch.randn(N))
    
    def forward(self, t, x):
        fx = self.f.expand(x.size(0), -1)  # B x N
        
        xT_fx = torch.sum(x * fx, dim=-1).unsqueeze(1)  # B x 1 (batched dot product)
        diff = fx - xT_fx  # B x N
        dxdt = torch.mul(x, diff)  # B x N
        
        return dxdt  # B x N


class cNODE0(nn.Module):
    def __init__(self, N):
        super(cNODE0, self).__init__()
        self.func = ODEFunc_cNODE0(N)
    
    def forward(self, t, x):
        y = odeint(self.func, x, t)[-1]
        return y
