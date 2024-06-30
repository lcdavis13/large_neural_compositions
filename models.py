import torch
import torch.nn as nn
from torchdiffeq import odeint
# from torchdiffeq import odeint_adjoint as odeint  # tiny memory footprint but it is intractible for large models such as cNODE2 with Waimea data


class ODEFunc_cNODE2(nn.Module): # optimized implementation of cNODE2
    def __init__(self, N):
        super(ODEFunc_cNODE2, self).__init__()
        self.fcc1 = nn.Linear(N, N)
        # self.bn1 = nn.BatchNorm1d(N)
        self.fcc2 = nn.Linear(N, N)
    
    def forward(self, t, x):
        fx = self.fcc1(x)  # B x N
        # fx = self.bn1(fx)
        fx = self.fcc2(fx)  # B x N
        
        xT_fx = torch.sum( x *fx, dim=-1).unsqueeze(1) # B x 1 (batched dot product)
        diff = fx - xT_fx # B x N
        dxdt = torch.mul(x, diff)  # B x N
        
        return dxdt # B x N
    
    
class cNODE2(nn.Module):
    def __init__(self, N):
        super(cNODE2, self).__init__()
        self.func = ODEFunc_cNODE2(N)
    
    def forward(self, t, x):
        x = odeint(self.func, x, t)[-1]
        return x


class ODEFunc_cNODE1(nn.Module):  # optimized implementation of cNODE2
    def __init__(self, N):
        super(ODEFunc_cNODE1, self).__init__()
        self.fcc1 = nn.Linear(N, N)
    
    def forward(self, t, x):
        fx = self.fcc1(x)  # B x N
        
        xT_fx = torch.sum(x * fx, dim=-1).unsqueeze(1)  # B x 1 (batched dot product)
        diff = fx - xT_fx  # B x N
        dxdt = torch.mul(x, diff)  # B x N
        
        return dxdt  # B x N


class cNODE1(nn.Module):
    def __init__(self, N):
        super(cNODE1, self).__init__()
        self.func = ODEFunc_cNODE1(N)
    
    def forward(self, t, x):
        x = odeint(self.func, x, t)[-1]
        return x


class Embedded_cNODE2(nn.Module):
    # This model doesn't work.
    # With softmax layers, it has the same score at the start of training as at the end. It doesn't learn.
    # Without any softmax layers, it starts off producing non-distributions, with huge loss. The model eventually beats the performance of the softmax version, but doesn't learn to produce a distribution.
    # Without softmax layers and WITH added penalties for non-distributions, it performs slightly better and becomes closer to producing a distribution, but it's only a partial success.
    # From just theoretical perspective, the ODE expects a few-hot encoded species assemblage summing to 1. It doesn't make sense to discard the distribution information in the input before it ever reaches the ODE. We should instead use two channels - embedded IDs for each species, and a small dense list of their abundances.
    def __init__(self, N, M):
        super(Embedded_cNODE2, self).__init__()
        self.embed = nn.Linear(N, M)  # can't use a proper embedding matrix because there are multiple active channels, not one-hot encoded
        # self.softmax = nn.Softmax(dim=-1)
        self.func = ODEFunc_cNODE2(M)
        self.unembed = nn.Linear(M, N)
        # self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, t, x):
        x = self.embed(x)
        # x = self.softmax(x)
        x = odeint(self.func, x, t)[-1]
        x = self.unembed(x)
        # x = self.softmax(x)
        return x


class ODEFunc_cNODE_Gen(nn.Module):  # cNODE2 with generalized f(x), specified at construction
    def __init__(self, f_constr):
        super(ODEFunc_cNODE_Gen, self).__init__()
        self.f = f_constr()
    
    def forward(self, t, x):
        fx = self.f(x)  # B x N
        
        xT_fx = torch.sum(x * fx, dim=-1).unsqueeze(1)  # B x 1 (batched dot product)
        diff = fx - xT_fx  # B x N
        dxdt = torch.mul(x, diff)  # B x N
        
        return dxdt  # B x N
    
class cNODE_Gen(nn.Module):
    def __init__(self, f_constr):
        super(cNODE_Gen, self).__init__()
        self.func = ODEFunc_cNODE_Gen(f_constr)
    
    def forward(self, t, x):
        x = odeint(self.func, x, t)[-1]
        return x


class ODEFunc_cNODE_GenRun(nn.Module):  # cNODE2 with generalized f(x), computed by calling context at runtime
    def __init__(self):
        super(ODEFunc_cNODE_GenRun, self).__init__()
    
    def forward(self, t, x, fx):
        xT_fx = torch.sum(x * fx, dim=-1).unsqueeze(1)  # B x 1 (batched dot product)
        diff = fx - xT_fx  # B x N
        dxdt = torch.mul(x, diff)  # B x N
        
        return dxdt  # B x N


class cNODE2_GenRun(nn.Module):
    def __init__(self, N):
        super(cNODE2_GenRun, self).__init__()
        self.func = ODEFunc_cNODE_GenRun()
        self.fcc1 = nn.Linear(N, N)
        self.fcc2 = nn.Linear(N, N)
    
    def forward(self, t, x):
        fx = self.fcc1(x)
        fx = self.fcc2(fx)
        dxdt = odeint(lambda x,t: self.func(x,t,fx), x, t)[-1]
        return dxdt


# class ODEFunc_cNODE2_DKI_unbatched(nn.Module):  # original DKI implementation of cNODE2, but will crash if you send batched data
#     def __init__(self, N):
#         super(ODEFunc_cNODE2_DKI_unbatched, self).__init__()
#         self.fcc1 = nn.Linear(N, N)
#         self.fcc2 = nn.Linear(N, N)
#
#     def forward(self, t, y):
#         out = self.fcc1(y)
#         out = self.fcc2(out)
#         f = torch.matmul(torch.matmul(torch.ones(y.size(dim=1), 1).to(y.device), y), torch.transpose(out, 0, 1))
#         return torch.mul(y, out - torch.transpose(f, 0, 1))

class ODEFunc_cNODE2_DKI(nn.Module): # DKI implementation of cNODE2 modified to allow batches
    def __init__(self, N):
        super(ODEFunc_cNODE2_DKI, self).__init__()
        self.fcc1 = nn.Linear(N, N)
        self.fcc2 = nn.Linear(N, N)

    def forward(self, t, y):
        y = y.unsqueeze(1)  # B x 1 x N
        out = self.fcc1(y)
        out = self.fcc2(out)
        f = torch.matmul(torch.matmul(torch.ones(y.size(dim=-1), 1).to(y.device), y), torch.transpose(out, -2, -1))
        dydt = torch.mul(y, out - torch.transpose(f, -2, -1))
        return dydt.squeeze(1)  # B x N

class cNODE2_DKI(nn.Module):
    def __init__(self, N):
        super(cNODE2_DKI, self).__init__()
        self.func = ODEFunc_cNODE2_DKI(N)

    def forward(self, t, x):
        x = odeint(self.func, x, t)[-1]
        return x
