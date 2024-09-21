import torch
import torch.nn as nn
from torchdiffeq import odeint

from models import cNODE1


# from torchdiffeq import odeint_adjoint as odeint  # tiny memory footprint but it is intractible for large models such as cNODE2 with Waimea data


class ReturnInput(nn.Module):
    # This model returns the input as the output. Since our inputs are fixed value for all nonzero features, this model is equivalent to returning a uniform distribution of the species in the assemblage.
    def __init__(self, N):
        super().__init__()
    
    def forward(self, t, x):
        return x


class ConstOutput(nn.Module):
    def __init__(self, N):
        super().__init__()
        self.f = nn.Parameter(torch.randn(N))
    
    def forward(self, t, x):
        return self.f


class ConstOutputFilteredNormalized(nn.Module):
    # This learns a vector for the relative distribution of each species in the dataset. It masks that to match the zero pattern of the input, then normalizes it to sum to 1.
    def __init__(self, N):
        super().__init__()
        self.f = nn.Parameter(torch.randn(N))
    
    def forward(self, t, x):
        f = self.f
        
        mask = x != 0
        f_selected = f[mask]
        f_normalized = f_selected / f_selected.sum()
        
        # Create the output y, fill zeros where masked, fill with f_normalized where unmasked
        y = torch.zeros_like(f)
        y[mask] = f_normalized
        
        return y
    
    
class SingleLayerPerceptron(nn.Module):
    def __init__(self, N):
        super().__init__()
        self.f = nn.Linear(N, N)
    
    def forward(self, t, x):
        return self.f(x)


class SingleLayerMultiplied(nn.Module):
    def __init__(self, N):
        super().__init__()
        self.f = nn.Linear(N, N)
    
    def forward(self, t, x): # x' = x*f(x)
        fx = self.f(x)  # B x N
        
        y = torch.mul(x, fx)  # B x N
        
        return y  # B x N


class SingleLayerSummed(nn.Module):
    def __init__(self, N):
        super().__init__()
        self.f = nn.Linear(N, N)
    
    def forward(self, t, x): # x' = x + f(x)
        fx = self.f(x)  # B x N
        
        y = x + fx  # B x N
        
        return y  # B x N


class SingleLayerMultipliedSummed(nn.Module):
    def __init__(self, N):
        super().__init__()
        self.f = nn.Linear(N, N)
    
    def forward(self, t, x): # x' = x + x*f(x)
        fx = self.f(x)  # B x N
        
        y = torch.mul(x, fx)  # B x N
        
        return x + y  # B x N


class cNODE1_singlestep(nn.Module):
    # cNODE1, but instead of solving the ODE fixed point, it takes one single step of the replicator equation.
    
    def __init__(self, N):
        super().__init__()
        self.func = ODEFunc_cNODE0(N)
    
    def forward(self, t, x):
        dxdt = self.func(t, x)
        return x + dxdt
    



class ODEFunc_cNODE0(nn.Module):
    # identical to ConstReplicator, except in ODE form; it returns the derivative instead of the next state.
    def __init__(self, N):
        super().__init__()
        self.f = nn.Parameter(torch.randn(N))
    
    def forward(self, t, x):
        fx = self.f.expand(x.size(0), -1)  # B x N
        
        xT_fx = torch.sum(x * fx, dim=-1).unsqueeze(1)  # B x 1 (batched dot product)
        diff = fx - xT_fx  # B x N
        dxdt = torch.mul(x, diff)  # B x N
        
        return dxdt  # B x N


class cNODE0(nn.Module):
    # cNODE where "F(x)" does not depend on x. In other words, it learns a fixed fitness value for each species regardless of which species are actually present.
    def __init__(self, N):
        super().__init__()
        self.func = ODEFunc_cNODE0(N)
    
    def forward(self, t, x):
        y = odeint(self.func, x, t)[-1]
        return y


class cNODE0_singlestep(nn.Module):
    # Identical to cNODE0 but instead of solving the ODE fixed point, it takes one single step of the replicator equation.
    def __init__(self, N):
        super().__init__()
        self.func = ODEFunc_cNODE0(N)
    
    def forward(self, t, x):
        dxdt = self.func(t, x)
        return x + dxdt