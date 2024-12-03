import torch
import torch.nn as nn

import models
from ode_solver import odeint


class ReturnInput(nn.Module):
    # This model returns the input as the output. Since our inputs are fixed value for all nonzero features, this model is equivalent to returning a uniform distribution of the species in the assemblage.
    def __init__(self):
        super().__init__()
    
    def forward(self, t, x):
        return x

class SingleConst(nn.Module):
    # This model returns the input as the output. Since our inputs are fixed value for all nonzero features, this model is equivalent to returning a uniform distribution of the species in the assemblage.
    def __init__(self):
        super().__init__()
        self.f = nn.Parameter(torch.randn(1))
    
    def forward(self, t, x):
        return self.f.expand(x.shape)


class SingleConstFilteredNormalized(nn.Module):
    # This learns a vector for the relative distribution of each species in the dataset. It masks that to match the zero pattern of the input, then normalizes it to sum to 1.
    def __init__(self):
        super().__init__()
        self.f = nn.Parameter(torch.randn(1))
    
    def forward(self, t, x):
        f = self.f.expand(x.shape)
        
        # Apply mask: we need to do this for each batch element separately
        mask = x != 0  # Shape: (batch_size, input_dim)
        
        # Masking and normalization per batch element
        y = torch.zeros_like(f)  # This will hold the output
        
        for i in range(x.shape[0]):
            f_selected = f[i, mask[i]]  # Select only the unmasked values for this batch element
            if f_selected.numel() > 0:  # If there are any unmasked elements
                f_normalized = f_selected / f_selected.sum()  # Normalize
                y[i, mask[i]] = f_normalized  # Assign normalized values back
        
        return y


class ConstOutput(nn.Module):
    def __init__(self, N):
        super().__init__()
        self.f = nn.Parameter(torch.randn(N))
    
    def forward(self, t, x):
        # x is assumed to be batched with shape (batch_size, input_dim)
        batch_size = x.shape[0]
        f_repeated = self.f.unsqueeze(0).expand(batch_size, -1)  # Repeat f for each batch element
        return f_repeated


class ConstOutputFilteredNormalized(nn.Module):
    # This learns a vector for the relative distribution of each species in the dataset. It masks that to match the zero pattern of the input, then normalizes it to sum to 1.
    def __init__(self, N):
        super().__init__()
        self.f = nn.Parameter(torch.randn(N))
    
    def forward(self, t, x):
        # x is assumed to be batched with shape (batch_size, input_dim)
        batch_size = x.shape[0]
        input_dim = x.shape[1]
        
        f = self.f.unsqueeze(0).expand(batch_size, -1)  # Repeat f for each batch element
        
        # Apply mask: we need to do this for each batch element separately
        mask = x != 0  # Shape: (batch_size, input_dim)
        
        # Masking and normalization per batch element
        y = torch.zeros_like(f)  # This will hold the output
        
        for i in range(batch_size):
            f_selected = f[i, mask[i]]  # Select only the unmasked values for this batch element
            if f_selected.numel() > 0:  # If there are any unmasked elements
                f_normalized = f_selected / f_selected.sum()  # Normalize
                y[i, mask[i]] = f_normalized  # Assign normalized values back
        
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


class SingleLayerFiltered(nn.Module):
    def __init__(self, N):
        super().__init__()
        self.f = nn.Linear(N, N)
    
    def forward(self, t, x):  # x' = x*f(x)
        fx = self.f(x)  # B x N

        ones = torch.zeros_like(x)
        y = torch.mul(ones, fx)  # B x N
        
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


class SingleLayerFilteredSummed(nn.Module):
    def __init__(self, N):
        super().__init__()
        self.f = nn.Linear(N, N)
    
    def forward(self, t, x): # x' = x + x*f(x)
        fx = self.f(x)  # B x N
        
        ones = torch.zeros_like(x)
        y = torch.mul(ones, fx)  # B x N
        
        return x + y  # B x N


class cNODE1_singlestep(nn.Module):
    # cNODE1, but instead of solving the ODE fixed point, it takes one single step of the replicator equation.
    
    def __init__(self, N):
        super().__init__()
        self.func = models.ODEFunc_cNODE1(N)
    
    def forward(self, t, x):
        dxdt = self.func(t, x)
        return x + dxdt


class ODEFunc_SLPODE(nn.Module):
    # use odeint to train a single layer perceptron's fixed point
    def __init__(self, N):
        super().__init__()
        self.f = nn.Linear(N, N)
    
    def forward(self, t, x):
        fx = self.f(x)  # B x N
        
        return fx  # B x N


class SLPODE(nn.Module):
    # use odeint to train a single layer perceptron's fixed point
    def __init__(self, N):
        super().__init__()
        self.func = ODEFunc_SLPODE(N)
    
    def forward(self, t, x):
        y = odeint(self.func, x, t)[-1]
        return y


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