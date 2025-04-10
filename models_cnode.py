import math

import torch
import torch.nn as nn
from ode_solver import odeint

class ODEFunc_cNODE2(nn.Module): # optimized implementation of cNODE2
    def __init__(self, N, bias):
        super().__init__()
        self.fcc1 = nn.Linear(N, N, bias=bias)
        # self.bn1 = nn.BatchNorm1d(N)
        self.fcc2 = nn.Linear(N, N, bias=bias)
    
    def forward(self, t, x):
        fx = self.fcc1(x)  # B x N
        # fx = self.bn1(fx)
        fx = self.fcc2(fx)  # B x N
        
        xT_fx = torch.sum( x *fx, dim=-1).unsqueeze(1) # B x 1 (batched dot product)
        diff = fx - xT_fx # B x N
        dxdt = torch.mul(x, diff)  # B x N
        
        return dxdt # B x N
    
    
class cNODE2(nn.Module):
    def __init__(self, N, bias):
        self.USES_ODEINT = True
        
        super().__init__()
        self.func = ODEFunc_cNODE2(N, bias=bias)
    
    def forward(self, t, x):
        y = odeint(self.func, x, t)
        return y


class ODEFunc_cNODE1(nn.Module):  # optimized implementation of cNODE2
    def __init__(self, N, bias):
        super().__init__()
        self.fcc1 = nn.Linear(N, N, bias=bias)
        
        # Initialize weights and biases to zero
        nn.init.zeros_(self.fcc1.weight)
        if bias:
            nn.init.zeros_(self.fcc1.bias)
    
    def forward(self, t, x):
        fx = self.fcc1(x)  # B x N
        
        xT_fx = torch.sum(x * fx, dim=-1).unsqueeze(1)  # B x 1 (batched dot product)
        # print(f"shape of x: {x.shape}")
        # print(f"shape of fx: {fx.shape}")
        # print(f"shape of xT_fx: {xT_fx.shape}")
        diff = fx - xT_fx  # B x N
        dxdt = torch.mul(x, diff)  # B x N
        
        return dxdt  # B x N


class cNODE1(nn.Module):
    def __init__(self, N, bias):
        self.USES_ODEINT = True
        
        super().__init__()
        self.func = ODEFunc_cNODE1(N, bias=bias)
    
    def forward(self, t, x):
        y = odeint(self.func, x, t)
        return y


class ODEFunc_cNODEGen_ConstructedFitness(nn.Module):  # cNODE2 with generalized f(x), specified via a constructor for the f(x) object passed during construction
    def __init__(self, f_constr):
        super().__init__()
        self.f = f_constr()
    
    def forward(self, t, x):
        fx = self.f(x)  # B x N
        
        xT_fx = torch.sum(x * fx, dim=-1).unsqueeze(1)  # B x 1 (batched dot product)
        diff = fx - xT_fx  # B x N
        dxdt = torch.mul(x, diff)  # B x N
        
        return dxdt  # B x N
    
    
class cNODEGen_ConstructedFitness(nn.Module):
    def __init__(self, f_constr):
        self.USES_ODEINT = True
        
        super().__init__()
        self.func = ODEFunc_cNODEGen_ConstructedFitness(f_constr)
    
    def forward(self, t, x):
        y = odeint(self.func, x, t) #, rtol=1e-10, atol=1e-12)[-1] # can increase tolerance when model is too stiff (underflows)
        return y


class cNODE_HourglassFitness(nn.Module):
    def __init__(self, data_dim, hidden_dim, depth, bias=True):
        self.USES_ODEINT = True
        super().__init__()
        
        self.func = ODEFunc_cNODEGen_ConstructedFitness(
            lambda: self.construct_fitness(data_dim, hidden_dim, depth, bias)
        )
    
    def forward(self, t, x):
        y = odeint(self.func, x, t)
        return y
    
    @staticmethod
    def construct_fitness(data_dim, hidden_dim, depth, bias):
        depth = depth + 1 # Add 1 to account for the input layer

        if depth < 3:
            raise ValueError("Depth must be at least 2 for a valid autoencoder structure.")
        
        layers = []
        half_depth = (depth - 1) // 2  # Exclude the input and output layers in the interpolation

        # Interpolate dimensions downwards (encoder) and upwards (decoder)
        down_dims = [
            int(data_dim * (hidden_dim / data_dim) ** (i / half_depth))
            for i in range(half_depth + 1)
        ]

        # Avoid repeating the middle dimension for odd depths
        is_odd = depth % 2 == 1
        if is_odd:
            reverse_depth = half_depth
        else:
            reverse_depth = half_depth + 1

        up_dims = list(reversed(down_dims[:reverse_depth])) 
        dims = down_dims + up_dims

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1], bias=bias))
            if i < len(dims) - 2:  # Add ReLU for all but the last layer
                layers.append(nn.ReLU())

        print(f"Constructed cNODE-HourglassFit with dimensions: {layers}")

        return nn.Sequential(*layers)


class ODEFunc_cNODEGen_FnFitness(nn.Module):  # cNODE2 with generalized f(x), specified at construction, but constructed externally (if necessary)
    def __init__(self, f):
        super().__init__()
        self.f = f
    
    def forward(self, t, x):
        fx = self.f(x)  # B x N
        
        xT_fx = torch.sum(x * fx, dim=-1).unsqueeze(1)  # B x 1 (batched dot product)
        diff = fx - xT_fx  # B x N
        dxdt = torch.mul(x, diff)  # B x N
        
        return dxdt  # B x N
    
class cNODEGen_FnFitness(nn.Module):
    def __init__(self, f):
        self.USES_ODEINT = True
        
        super().__init__()
        self.func = ODEFunc_cNODEGen_FnFitness(f)
    
    def forward(self, t, x):
        y = odeint(self.func, x, t)
        return y


class ODEFunc_cNODEGen_FnFitness_Args(nn.Module):  # cNODE2 with generalized f(x), specified at construction, but constructed externally (if necessary)
    def __init__(self, f):
        super().__init__()
        self.f = f
    
    def forward(self, t, x, args):
        fx = self.f(x, args)  # B x N
        
        xT_fx = torch.sum(x * fx, dim=-1).unsqueeze(1)  # B x 1 (batched dot product)
        diff = fx - xT_fx  # B x N
        dxdt = torch.mul(x, diff)  # B x N
        
        return dxdt  # B x N

    
class cNODE2_FnFitness(nn.Module):
    def __init__(self, N, bias):
        self.USES_ODEINT = True
        
        super().__init__()
        f = nn.Sequential(
            nn.Linear(N, N, bias=bias),
            nn.Linear(N, N, bias=bias)
        )
        self.func = ODEFunc_cNODEGen_FnFitness(f)
    
    def forward(self, t, x):
        y = odeint(self.func, x, t)
        return y


class ODEFunc_cNODEGen_ExternalFitness(nn.Module):  # cNODE2 with generalized f(x0), computed by calling context at runtime
    # Note that, while f(x0) can be a function of the initial vector x0, it cannot be a function of the evolving value x
    # as the ODE is evaluated, unlike normal implementations of cNODE.
    def __init__(self):
       super(ODEFunc_cNODEGen_ExternalFitness, self).__init__()
    
    def forward(self, t, x, fx0):
        xT_fx = torch.sum(x * fx0, dim=-1).unsqueeze(1)  # B x 1 (batched dot product)
        diff = fx0 - xT_fx  # B x N
        dxdt = torch.mul(x, diff)  # B x N
        
        return dxdt  # B x N


class cNODE2_ExternalFitness(nn.Module):
    def __init__(self, N, bias):
        self.USES_ODEINT = True
        
        super().__init__()
        self.func = ODEFunc_cNODEGen_ExternalFitness()
        self.fcc1 = nn.Linear(N, N, bias=bias)
        self.fcc2 = nn.Linear(N, N, bias=bias)
    
    def forward(self, t, x):
        fx = self.fcc1(x)
        fx = self.fcc2(fx)
        y = odeint(lambda x,t: self.func(x,t,fx), x, t)
        return y


class ODEFunc_cNODEGen_ExternalFitnessFn(nn.Module):
    # cNODE2 with generalized f(x0), computed by calling context at runtime
    # Note that, while f(x0) can be a function of the initial vector x0, it cannot be a function of the evolving value x
    # as the ODE is evaluated, unlike normal implementations of cNODE.
    def __init__(self):
        super().__init__()
    
    def forward(self, t, x, Fn):
        fitness = Fn(x)
        xT_fx = torch.sum(x * fitness, dim=-1).unsqueeze(1)  # B x 1 (batched dot product)
        diff = fitness - xT_fx  # B x N
        dxdt = torch.mul(x, diff)  # B x N
        
        return dxdt  # B x N


class cNODE2_ExternalFitnessFn(nn.Module):
    def __init__(self, N, bias):
        self.USES_ODEINT = True
        
        super().__init__()
        self.func = ODEFunc_cNODEGen_ExternalFitnessFn()
        self.Fn = nn.Linear(N, N, bias=bias)
    
    def forward(self, t, x):
        y = odeint(lambda x,t: self.func(x,t,self.Fn), x, t)
        return y


# class ODEFunc_cNODE2_DKI_unbatched(nn.Module):  # original DKI implementation of cNODE2, but will crash if you send batched data
#     def __init__(self, N, bias):
#         super(ODEFunc_cNODE2_DKI_unbatched, self).__init__()
#         self.fcc1 = nn.Linear(N, N, bias=bias)
#         self.fcc2 = nn.Linear(N, N, bias=bias)
#
#     def forward(self, t, y):
#         out = self.fcc1(y)
#         out = self.fcc2(out)
#         f = torch.matmul(torch.matmul(torch.ones(y.size(dim=1), 1).to(y.device), y), torch.transpose(out, 0, 1))
#         return torch.mul(y, out - torch.transpose(f, 0, 1))

class ODEFunc_cNODE2_DKI(nn.Module): # DKI implementation of cNODE2 modified to allow batches
    def __init__(self, N, bias):
        super().__init__()
        self.fcc1 = nn.Linear(N, N, bias=bias)
        self.fcc2 = nn.Linear(N, N, bias=bias)

    def forward(self, t, y):
        y = y.unsqueeze(1)  # B x 1 x N
        out = self.fcc1(y)
        out = self.fcc2(out)
        f = torch.matmul(torch.matmul(torch.ones(y.size(dim=-1), 1).to(y.device), y), torch.transpose(out, -2, -1))
        dydt = torch.mul(y, out - torch.transpose(f, -2, -1))
        return dydt.squeeze(1)  # B x N

#
# class canODE_concat_cNODE(nn.Module):
#     '''
#     Version of canODE that learns to directly generate a fitness interaction matrix
#
#     1. Create condensed embedding
#     2. Enrich embeddings with transformer encoder
#     3. Create a fitness interaction matrix using embedded scaled dot product similarity (a subcomponent of attention)
#     4. Retrieve fitness biases from a learnable parameter
#     5. Compose the fitness function as F(x) = M * x + b
#     6. Run cNODE1 using the computed fitness function
#     '''
#
#     def __init__(self, data_dim, id_embed_dim, num_heads, depth, ffn_dim_multiplier, fitness_qk_dim, dropout):
#         self.USES_ODEINT = True
#         super().__init__()
#
#         self.data_dim = data_dim
#         self.embed = nn.Embedding(data_dim + 1, id_embed_dim)  # Add 1 to account for placeholder ID
#
#         # define the transformer
#         encoder_layer = nn.TransformerEncoderLayer(d_model=id_embed_dim, nhead=num_heads,
#                                                    dim_feedforward=math.ceil(id_embed_dim * ffn_dim_multiplier),
#                                                    activation="gelu", batch_first=True, dropout=dropout)
#         self.transform = nn.TransformerEncoder(encoder_layer, num_layers=depth)
#
#         # define the fitness function generators
#         self.fitmatrix_Q = nn.Linear(id_embed_dim, fitness_qk_dim)
#         self.fitmatrix_K = nn.Linear(id_embed_dim, fitness_qk_dim)
#         self.fitmatrix_scalefactor = fitness_qk_dim ** -0.5
#         self.fitbias = nn.Parameter(torch.zeros(data_dim + 1))  # Add 1 to account for placeholder ID
#
#         # define the ODE function
#         self.ode_func = ODEFunc_cNODEGen_ExternalFitnessFn()
#
#     def forward(self, t, x):
#         val = x
#         pos = torch.arange(1, x.size(1) + 1, device=x.device)
#         pos = pos * (x != 0).long()
#
#         # modify v
#         id_embed = self.embed(pos)
#         id_embed = self.transform(id_embed)
#
#         q = self.fitmatrix_Q(id_embed)
#         k = self.fitmatrix_K(id_embed)
#         fitmatrix = torch.einsum('...id , ...jd -> ...ij', q, k) * self.fitmatrix_scalefactor
#         fitbias = self.fitbias[pos]
#         fitnessFn = lambda h: torch.einsum('...ij, ...i -> ...j', fitmatrix, h) + fitbias
#
#         y = odeint(lambda xo, to: self.ode_func(xo, to, fitnessFn), val, t)
#
#         return y

class cNODE2_DKI(nn.Module):
    def __init__(self, N):
        self.USES_ODEINT = True
        
        super().__init__()
        self.func = ODEFunc_cNODE2_DKI(N)

    def forward(self, t, x):
        y = odeint(self.func, x, t)
        return y
