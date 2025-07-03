import torch
import torch.nn as nn
from ode_solver import odeint

class ODEFunc_cNODE1(nn.Module):  # optimized implementation of cNODE2
    def __init__(self, data_dim):
        super().__init__()

        # assert False
        self.fcc1 = nn.Linear(in_features=data_dim, out_features=data_dim, bias=False)
        nn.init.zeros_(self.fcc1.weight)

    
    def forward(self, t, x):
        fx = self.fcc1(x)  # B x N
        
        xT_fx = torch.sum(x * fx, dim=-1).unsqueeze(1)  # B x 1 (batched dot product)
        diff = fx - xT_fx  # B x N
        dxdt = torch.mul(x, diff)  # B x N
        
        return dxdt


class cNODE1(nn.Module):
    def __init__(self, data_dim):
        self.USES_ODEINT = True
        super().__init__()
        
        self.func = ODEFunc_cNODE1(data_dim)
    
    def forward(self, t, x):
        y = odeint(self.func, x, t)
        return y