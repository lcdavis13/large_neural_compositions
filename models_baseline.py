import torch
import torch.nn as nn

import models_cnode
from ode_solver import odeint


def masked_softmax(y, x):
    # Mask out elements where pos is 0
    mask = (x > 0.0)
    masked_y = y.masked_fill(~mask, float('-inf'))
    # Normalize the output to sum to 1 (excluding masked elements)
    y = nn.functional.softmax(masked_y, dim=-1)
    # Set masked elements to 0
    y = y * mask.float()
    return y


class ReturnInput(nn.Module):
    # This model returns the input as the output. Since our inputs are fixed value for all nonzero features, this model is equivalent to returning a uniform distribution of the species in the assemblage.
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x

class SingleConst(nn.Module):
    # This model returns the input as the output. Since our inputs are fixed value for all nonzero features, this model is equivalent to returning a uniform distribution of the species in the assemblage.
    def __init__(self):
        super().__init__()
        self.f = nn.Parameter(torch.randn(1))
    
    def forward(self, x):
        return self.f.expand(x.shape)


class SingleConstFilteredNormalized(nn.Module):
    # This learns a vector for the relative distribution of each species in the dataset. It masks that to match the zero pattern of the input, then normalizes it to sum to 1.
    def __init__(self):
        super().__init__()
        self.f = nn.Parameter(torch.randn(1))
    
    def forward(self, x):
        f = self.f.expand(x.shape)
        
        y = masked_softmax(f, x)  # Apply mask: we need to do this for each batch element separately
        
        return y


class ConstOutput(nn.Module):
    def __init__(self, N):
        super().__init__()
        self.f = nn.Parameter(torch.randn(N))
    
    def forward(self, x):
        # x is assumed to be batched with shape (batch_size, input_dim)
        batch_size = x.shape[0]
        f_repeated = self.f.unsqueeze(0).expand(batch_size, -1)  # Repeat f for each batch element
        return f_repeated


class ConstOutputFilteredNormalized(nn.Module):
    # This learns a vector for the relative distribution of each species in the dataset. It masks that to match the zero pattern of the input, then normalizes it to sum to 1.
    def __init__(self, N, identity_gate):
        super().__init__()
        self.f = nn.Parameter(torch.randn(N))

        if identity_gate:
            self.gateA = nn.Parameter(torch.tensor(0.0))
            self.gateB = nn.Parameter(torch.tensor(1.0))
        else:
            self.register_buffer('gateA', torch.tensor(1.0))
            self.register_buffer('gateB', torch.tensor(0.0))

    def forward(self, x):
        # x is assumed to be batched with shape (batch_size, input_dim)
        batch_size = x.shape[0]
        
        f = self.f.unsqueeze(0).expand(batch_size, -1)  # Repeat f for each batch element
        
        gated_f = self.gateA*f + self.gateB*x
        
        y = masked_softmax(gated_f, x) 

        return y


class SingleLayerPerceptron(nn.Module):
    def __init__(self, N, M):
        super().__init__()
        self.f1 = nn.Linear(N, M)
        self.relu = nn.ReLU()
        self.f2 = nn.Linear(M, N)
    
    def forward(self, t, x):
        h = self.f1(x)
        h = self.relu(h)
        h = self.f2(h)
        
        return h
    
    
class SLPFilteredNormalized(nn.Module):
    # This learns a vector for the relative distribution of each species in the dataset. It masks that to match the zero pattern of the input, then normalizes it to sum to 1.
    def __init__(self, N, M):
        super().__init__()
        self.f1 = nn.Linear(N, M)
        self.relu = nn.ReLU()
        self.f2 = nn.Linear(M, N)
    
    def forward(self, x):
        f = self.f1(x)
        f = self.relu(f)
        f = self.f2(f)
        
        y = masked_softmax(f, x)
        
        return y


class SLPSumFilteredNormalized(nn.Module):
    # This learns a vector for the relative distribution of each species in the dataset. It masks that to match the zero pattern of the input, then normalizes it to sum to 1.
    def __init__(self, N, M):
        super().__init__()
        self.f1 = nn.Linear(N, M)
        self.relu = nn.ReLU()
        self.f2 = nn.Linear(M, N)
    
    def forward(self, x):
        f = self.f1(x)
        f = self.relu(f)
        f = self.f2(f)
        f = x + f
        
        y = masked_softmax(f, x)
        
        return y


class SLPMultFilteredNormalized(nn.Module):
    # This learns a vector for the relative distribution of each species in the dataset. It masks that to match the zero pattern of the input, then normalizes it to sum to 1.
    def __init__(self, N, M, identity_gate):
        super().__init__()
        self.f1 = nn.Linear(N, M)
        self.relu = nn.ReLU()
        self.f2 = nn.Linear(M, N)

        if identity_gate:
            self.gateA = nn.Parameter(torch.tensor(0.0))
            self.gateB = nn.Parameter(torch.tensor(1.0))
        else:
            self.register_buffer('gateA', torch.tensor(1.0))
            self.register_buffer('gateB', torch.tensor(0.0))
    
    def forward(self, x):
        f = self.f1(x)
        f = self.relu(f)
        f = self.f2(f)
        f = x * f

        gated_f = self.gateA*f + self.gateB*x
        
        y = masked_softmax(gated_f, x)
        
        return y


class SLPMultSumFilteredNormalized(nn.Module):
    # This learns a vector for the relative distribution of each species in the dataset. It masks that to match the zero pattern of the input, then normalizes it to sum to 1.
    def __init__(self, N, M):
        super().__init__()
        self.f1 = nn.Linear(N, M)
        self.relu = nn.ReLU()
        self.f2 = nn.Linear(M, N)
    
    def forward(self, x):
        f = self.f1(x)
        f = self.relu(f)
        f = self.f2(f)
        f = (x * f) + x
        
        y = masked_softmax(f, x)
        
        return y
    
    
class SingleLayer(nn.Module):
    def __init__(self, N):
        super().__init__()
        self.f = nn.Linear(N, N)
    
    def forward(self, x):
        return self.f(x)


class SingleLayerMultiplied(nn.Module):
    def __init__(self, N):
        super().__init__()
        self.f = nn.Linear(N, N)
    
    def forward(self, x): # x' = x*f(x)
        fx = self.f(x)  # B x N
        
        y = torch.mul(x, fx)  # B x N
        
        return y  # B x N


class SingleLayerFiltered(nn.Module):
    def __init__(self, N):
        super().__init__()
        self.f = nn.Linear(N, N)
    
    def forward(self, x):  # x' = x*f(x)
        fx = self.f(x)  # B x N

        ones = torch.zeros_like(x)
        y = torch.mul(ones, fx)  # B x N
        
        return y  # B x N


class SingleLayerSummed(nn.Module):
    def __init__(self, N):
        super().__init__()
        self.f = nn.Linear(N, N)
    
    def forward(self, x): # x' = x + f(x)
        fx = self.f(x)  # B x N
        
        y = x + fx  # B x N
        
        return y  # B x N


class SingleLayerMultipliedSummed(nn.Module):
    def __init__(self, N):
        super().__init__()
        self.f = nn.Linear(N, N)
    
    def forward(self, x): # x' = x + x*f(x)
        fx = self.f(x)  # B x N
        
        y = torch.mul(x, fx)  # B x N
        
        return x + y  # B x N


class SingleLayerFilteredSummed(nn.Module):
    def __init__(self, N):
        super().__init__()
        self.f = nn.Linear(N, N)
    
    def forward(self, x): # x' = x + x*f(x)
        fx = self.f(x)  # B x N
        
        ones = torch.zeros_like(x)
        y = torch.mul(ones, fx)  # B x N
        
        return x + y  # B x N


class cNODE1_singlestep(nn.Module):
    # cNODE1, but instead of solving the ODE fixed point, it takes one single step of the replicator equation.
    
    def __init__(self, N, bias, init_zero, identity_gate):
        super().__init__()
        self.func = models_cnode.ODEFunc_cNODE1(N, bias, init_zero=init_zero, identity_gate=identity_gate)
    
    def forward(self, x):
        dxdt = self.func([0.0], x)
        return x + dxdt


class SLPODE(nn.Module):
    # use odeint to train a single layer perceptron's fixed point
    def __init__(self, N, M):
        self.USES_ODEINT = True
        
        super().__init__()
        self.func = SingleLayerPerceptron(N, M)
    
    def forward(self, t, x):
        y = odeint(self.func, x, t)
        return y


class ODEFunc_cNODE0(nn.Module):
    # identical to ConstReplicator, except in ODE form; it returns the derivative instead of the next state.
    def __init__(self, N, init_zero=True, identity_gate=False):
        super().__init__()
        self.f = nn.Parameter(torch.rand(N))
        
        if not identity_gate:
            self.register_buffer('gate', torch.tensor(1.0)) # Make sure this has the same name as the parameter version
        
        if init_zero:
            # Initialize weights and biases to zero (this is the original approach from the paper)
            nn.init.zeros_(self.f)

            if identity_gate:
                self.gate = nn.Parameter(torch.tensor(1.0)) # modified identity_gate for when init_zero is used. Otherwise there would be symmetry and model couldn't learn. In the hypothetical where identity_gate gate provides some benefit beyond the initialization to zero, this can capture that benefit when init_zero is used.
        else:
            if identity_gate:
                self.gate = nn.Parameter(torch.tensor(0.0)) # regular identity_gate, when init_zero isn't used
    

    def forward(self, t, x):
        fx = self.f.expand(x.size(0), -1)  # B x N
        
        xT_fx = torch.sum(x * fx, dim=-1).unsqueeze(1)  # B x 1 (batched dot product)
        diff = fx - xT_fx  # B x N
        dxdt = torch.mul(x, diff)  # B x N
        
        return self.gate*dxdt  # B x N


class cNODE0(nn.Module):
    # cNODE where "F(x)" does not depend on x. In other words, it learns a fixed fitness value for each species regardless of which species are actually present.
    def __init__(self, N, init_zero, identity_gate):
        self.USES_ODEINT = True
        
        super().__init__()
        self.func = ODEFunc_cNODE0(N, init_zero=init_zero, identity_gate=identity_gate)
    
    def forward(self, t, x):
        y = odeint(self.func, x, t)
        return y


class cNODE0_singlestep(nn.Module):
    # Identical to cNODE0 but instead of solving the ODE fixed point, it takes one single step of the replicator equation.
    def __init__(self, N, init_zero, identity_gate):
        super().__init__()
        self.func = ODEFunc_cNODE0(N, init_zero=init_zero, identity_gate=identity_gate)
    
    def forward(self, x):
        dxdt = self.func([0.0], x)
        return x + dxdt