import torch
import torch.nn as nn
import torch.nn.functional as F
from ode_solver import odeint
import model_linearKH as lin

class ODEFunc_cNODE1(nn.Module):
    def __init__(self, data_dim):
        super().__init__()

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
    

class ODEFunc_cNODE2(nn.Module): 
    def __init__(self, data_dim):
        super().__init__()

        self.fcc1 = lin.LinearKH(in_features=data_dim, out_features=data_dim, bias=False)
        self.fcc2 = lin.LinearKH(in_features=data_dim, out_features=data_dim, bias=False)
    
    def forward(self, t, x):
        fx = self.fcc1(x)  # B x N
        fx = self.fcc2(fx)  # B x N
        
        xT_fx = torch.sum(x * fx, dim=-1).unsqueeze(1)  # B x 1 (batched dot product)
        diff = fx - xT_fx  # B x N
        dxdt = torch.mul(x, diff)  # B x N
        
        return dxdt


class cNODE2(nn.Module):
    def __init__(self, data_dim):
        self.USES_ODEINT = True
        super().__init__()

        self.func = ODEFunc_cNODE2(data_dim)
    
    def forward(self, t, x):
        y = odeint(self.func, x, t)
        return y


class ODEFunc_cNODE2_NonLinear(nn.Module): 
    def __init__(self, data_dim):
        super().__init__()

        self.fcc1 = lin.LinearKH(in_features=data_dim, out_features=data_dim)
        self.fcc2 = lin.LinearKH(in_features=data_dim, out_features=data_dim)

    
    def forward(self, t, x):
        fx = self.fcc1(x)  # B x N
        fx = F.gelu(fx)
        fx = self.fcc2(fx) # B x N
        
        xT_fx = torch.sum(x * fx, dim=-1).unsqueeze(1)  # B x 1 (batched dot product)
        diff = fx - xT_fx  # B x N
        dxdt = torch.mul(x, diff)  # B x N
        
        return dxdt


class cNODE2_NonLinear_Biased(nn.Module):
    def __init__(self, data_dim):
        self.USES_ODEINT = True
        super().__init__()
        
        self.func = ODEFunc_cNODE2_NonLinear(data_dim)
    
    def forward(self, t, x):
        y = odeint(self.func, x, t)
        return y

# # The following models (all of which are variations of cNODE with an extra dimension added) have been temporarily removed until I add support for returning the derivative

# class ODEFunc_glv1NODE(nn.Module):
#     def __init__(self, data_dim):
#         super().__init__()
#         self.data_dim = data_dim

#         # Trainable matrix A: (N x (N+1))
#         self.A = nn.Parameter(torch.zeros(data_dim, data_dim + 1))

#         # Non-trainable zero vector: (N+1,)
#         self.zeros = nn.Parameter(torch.zeros(data_dim + 1), requires_grad=False)

#     def forward(self, t, x_aug):
#         # x_aug: (B, N+1)
#         B = x_aug.shape[0]

#         # Build full (N+1 x N+1) matrix: top = A, bottom = r
#         top = self.A                      # (N x N+1)
#         bottom = self.zeros.view(1, -1)       # (1 x N+1)
#         A_tilde = torch.cat([top, bottom], dim=0)  # (N+1 x N+1)

#         fx = torch.matmul(x_aug, A_tilde.T)             # (B x N+1)
#         xT_fx = torch.sum(x_aug * fx, dim=1, keepdim=True)  # (B x 1)
#         dxdt = x_aug * (fx - xT_fx)                     # (B x N+1)

#         return dxdt


# class glv1NODE(nn.Module):
#     def __init__(self, data_dim, env_scale):
#         super().__init__()
#         self.USES_ODEINT = True
#         self.data_dim = data_dim
#         self.env_scale = env_scale
#         self.func = ODEFunc_glv1NODE(data_dim)

#     def forward(self, t, x):
#         # x: (B, N), already on simplex (sum = 1)

#         # Add extra component (any small constant, 1 if we want the terms to have the same scale as in our generative gLV model that starts on simplex) and renormalize

#         x_extra = torch.full((x.shape[0], 1), fill_value=self.env_scale, device=x.device)
#         x_aug = torch.cat([x, x_extra], dim=1)  # (B, N+1)
#         x_aug = x_aug / x_aug.sum(dim=1, keepdim=True)  # Renormalize to simplex

#         # Solve ODE
#         y_aug, dydt_aug = odeint(self.func, x_aug, t)  # (T, B, N+1)

#         # Remove extra dimension and renormalize remaining components
#         y = y_aug[..., :self.data_dim]                    # (T, B, N)
#         y = y / y.sum(dim=-1, keepdim=True)               # Renormalize to simplex

#         dydt = dydt_aug[..., :self.data_dim]               # (T, B, N)
#         dydt = dydt / dydt.sum(dim=-1, keepdim=True)       # Renormalize to simplex

#         return y, dydt


# class ODEFunc_glv2NODE(nn.Module):
#     def __init__(self, data_dim):
#         super().__init__()
#         self.data_dim = data_dim

#         # Trainable matrix A: ((N+1) x N)
#         self.A = nn.Parameter(torch.zeros(data_dim + 1, data_dim))

#         # Non-trainable zero vector: (N+1,)
#         self.zeros = nn.Parameter(torch.zeros(data_dim + 1), requires_grad=False)

#     def forward(self, t, x_aug):
#         # x_aug: (B, N+1)
#         B = x_aug.shape[0]

#         # Build full (N+1 x N+1) matrix: left = A, bottom = r
#         left = self.A                      # (N+1 x N)
#         right = self.zeros.unsqueeze(1)       # (N+1 x 1)
#         A_tilde = torch.cat([left, right], dim=1)  # (N+1 x N+1)

#         fx = torch.matmul(x_aug, A_tilde.T)             # (B x N+1)
#         xT_fx = torch.sum(x_aug * fx, dim=1, keepdim=True)  # (B x 1)
#         dxdt = x_aug * (fx - xT_fx)                     # (B x N+1)

#         return dxdt


# class glv2NODE(nn.Module):
#     def __init__(self, data_dim, env_scale):
#         super().__init__()
#         self.USES_ODEINT = True
#         self.data_dim = data_dim
#         self.env_scale = env_scale
#         self.func = ODEFunc_glv2NODE(data_dim)

#     def forward(self, t, x):
#         # x: (B, N), already on simplex (sum = 1)

#         # Add extra component (any small constant, 1 if we want the terms to have the same scale as in our generative gLV model that starts on simplex) and renormalize

#         x_extra = torch.full((x.shape[0], 1), fill_value=self.env_scale, device=x.device)
#         x_aug = torch.cat([x, x_extra], dim=1)  # (B, N+1)
#         x_aug = x_aug / x_aug.sum(dim=1, keepdim=True)  # Renormalize to simplex

#         # Solve ODE
#         y_aug = odeint(self.func, x_aug, t)  # (T, B, N+1)

#         # Remove extra dimension and renormalize remaining components
#         y = y_aug[..., :self.data_dim]                    # (T, B, N)
#         y = y / y.sum(dim=-1, keepdim=True)               # Renormalize to simplex

#         return y



# class ODEFunc_envNODE(nn.Module):
#     def __init__(self, data_dim, env_dim):
#         super().__init__()
#         self.data_dim = data_dim

#         self.A = nn.Parameter(torch.zeros(data_dim + env_dim, data_dim + env_dim))

#         if env_dim > 1:
#             # Slightly perturb the environmental effects on species to break symmetry
#             eps = 1e-4
#             with torch.no_grad():
#                 self.A[:data_dim, data_dim:] += eps * torch.randn(data_dim, env_dim)

#     def forward(self, t, x_aug):
#         # x_aug: (B, N+1)
#         B = x_aug.shape[0]

#         fx = torch.matmul(x_aug, self.A.T)             # (B x N+1)
#         xT_fx = torch.sum(x_aug * fx, dim=1, keepdim=True)  # (B x 1)
#         dxdt = x_aug * (fx - xT_fx)                     # (B x N+1)

#         return dxdt


# class envNODE(nn.Module):
#     def __init__(self, data_dim, env_dim, env_scale):
#         super().__init__()
#         self.USES_ODEINT = True
#         self.data_dim = data_dim
#         self.env_dim = env_dim
#         self.env_scale = env_scale
#         self.func = ODEFunc_envNODE(data_dim, env_dim)

#     def forward(self, t, x):
#         # x: (B, N), already on simplex (sum = 1)

#         # Add extra component (any small constant, 1 if we want the terms to have the same scale as in our generative gLV model that starts on simplex) and renormalize
#         # if self.env_scale <= 0.0:
#         #     # 1 / number of nonzero elements per row
#         #     x_extra = 1.0 / (x != 0).sum(dim=1, keepdim=True)
#         # else:
#         x_extra = torch.full((x.shape[0], self.env_dim), fill_value=self.env_scale, device=x.device)
#         x_aug = torch.cat([x, x_extra], dim=1)  # (B, N+1)
#         x_aug = x_aug / x_aug.sum(dim=1, keepdim=True)  # Renormalize to simplex

#         # Solve ODE
#         y_aug = odeint(self.func, x_aug, t)  # (T, B, N+1)

#         # Remove extra dimension and renormalize remaining components
#         y = y_aug[..., :self.data_dim]                    # (T, B, N)
#         y = y / y.sum(dim=-1, keepdim=True)               # Renormalize to simplex

#         return y
