import torch
import torch.nn as nn
from torchdiffeq import odeint
import models


def condense(x):
    device = x.device
    batch_size = x.size(0)
    max_len = max(torch.sum(x[i] != 0).item() for i in range(batch_size))
    
    padded_values = torch.zeros(batch_size, max_len, dtype=x.dtype, device=device)
    padded_positions = torch.zeros(batch_size, max_len, dtype=torch.long, device=device)
    
    for i in range(batch_size):
        positions = torch.nonzero(x[i], as_tuple=True)[0]
        values = x[i][positions]
        len_values = len(values)
        
        padded_values[i, :len_values] = values
        padded_positions[i, :len_values] = positions + 1  # Adjust positions by 1
    
    return padded_values, padded_positions


def decondense(values, positions, size):
    device = values.device
    batch_size = values.size(0)
    y = torch.zeros(batch_size, size, dtype=values.dtype, device=device)
    
    for i in range(batch_size):
        valid_positions = positions[i]  # Compensate for the adjustment
        valid_positions = valid_positions[valid_positions > 0]  # Remove any zero positions
        valid_positions -= 1  # Adjust positions back to zero-based indexing
        len_valid = len(valid_positions)
        
        y[i][valid_positions[:len_valid]] = values[i][:len_valid]
    
    return y


class cAttend_simple(nn.Module):
    def __init__(self, data_dim, id_embed_dim, qk_dim):
        super(cAttend_simple, self).__init__()
        self.data_dim = data_dim
        self.embed = nn.Embedding(data_dim+1, id_embed_dim)  # Add 1 to account for placeholder ID
        self.w_q = nn.Linear(id_embed_dim, qk_dim)
        self.w_k = nn.Linear(id_embed_dim, qk_dim)
    
    def forward(self, t, x):
        v,p = condense(x)
        
        # modify v
        id_embed = self.embed(p)
        
        q = self.w_q(id_embed)
        k = self.w_k(id_embed)
        soft_weights = torch.einsum('...id , ...jd -> ...ij', q, k) * self.scale_factor
        fx = torch.einsum('...ij, ...i -> ...j', soft_weights, v)
        
        dv = torch.mul(v, fitness)
        
        v = torch.add(v, dv) # Can also just do v += fitness but this adds a bit of ODE-like inductive bias
        
        y = decondense(v, p, self.data_dim)
        return y


class canODE_simple(nn.Module): # compositional attention nODE
    def __init__(self, data_dim, id_embed_dim, qk_dim):
        super(canODE_simple, self).__init__()
        self.data_dim = data_dim
        self.embed = nn.Embedding(data_dim + 1, id_embed_dim)  # Add 1 to account for placeholder ID
        self.w_q = nn.Linear(id_embed_dim, qk_dim)
        self.w_k = nn.Linear(id_embed_dim, qk_dim)
        self.func = models.ODEFunc_cNODE_GenRun()
        self.scale_factor = qk_dim ** -0.5
    
    def forward(self, t, x):
        v, p = condense(x)
        
        # modify v
        id_embed = self.embed(p)
        
        q = self.w_q(id_embed)
        k = self.w_k(id_embed)
        soft_weights = torch.einsum('...id , ...jd -> ...ij', q, k) * self.scale_factor
        fx = torch.einsum('...ij, ...i -> ...j', soft_weights, v)
        # intentionally not computing v from attention mechanism, since with a single head it could not produce both benefits and harms to fitness.
        # Without softmax and without v, the dot product of Q+K can be either positive or negative, increasing or decreasing fitness.
        # But multihead attention could solve that in a more flexible way.
        
        y = odeint(lambda xo,to: self.func(xo,to,fx), v, t)[-1]
        
        y = decondense(y, p, self.data_dim)
        return y
    
# nn.functional.scaled_dot_product_attention
# nn.MultiheadAttention
# nn.TransformerEncoder + nn.TransformerEncoderLayer

class ctnODE(nn.Module):
    def __init__(self, data_dim, id_embed_dim, qk_dim):
        encoder_layer = nn.TransformerEncoderLayer(d_model=qk_dim, nhead=8, dim_feedforward=id_embed_dim,
            activation="gelu", batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
    
    
if __name__ == '__main__':
    B = 5
    
    # random test data
    # N = 10
    # x = torch.randn(B, N)
    # num_zeros = 30
    # zero_indices_flat = torch.randperm(B * N)[:num_zeros]
    # zero_indices = torch.stack([zero_indices_flat // N, zero_indices_flat % N], dim=-1)
    # indices = zero_indices.unbind(dim=-1)
    # x[indices[0], indices[1]] = 0.0
    
    # real test data
    import data
    x_all,y_all = data.load_data('data/cNODE-paper-drosophila_train.csv', None)
    x,y,_ = data.get_batch(x_all, y_all, B, 0)
    _, N = x.shape
    
    print(y)
    print(x)

    t = torch.linspace(0, 1, 2)
    E_dim = 3
    QK_dim = 4
    model = canODE_simple(N, E_dim, QK_dim)
    y_pred = model(t, x)
    print(y_pred)
    
    print(y_pred - y)