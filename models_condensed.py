import math

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
        val, pos = condense(x)
        
        # modify v
        id_embed = self.embed(pos)
        
        q = self.w_q(id_embed)
        k = self.w_k(id_embed)
        soft_weights = torch.einsum('...id , ...jd -> ...ij', q, k) * self.scale_factor
        fx = torch.einsum('...ij, ...i -> ...j', soft_weights, val)
        
        dv = torch.mul(val, fx)
        val = torch.add(val, dv) # Can also just do v += fitness but this adds a bit of ODE-like inductive bias
        
        y = decondense(val, pos, self.data_dim)
        return y


class canODE_attentionNoValue(nn.Module): # compositional attention nODE
    def __init__(self, data_dim, id_embed_dim, qk_dim):
        super(canODE_attentionNoValue, self).__init__()
        self.data_dim = data_dim
        self.embed = nn.Embedding(data_dim + 1, id_embed_dim)  # Add 1 to account for placeholder ID
        self.w_q = nn.Linear(id_embed_dim, qk_dim)
        self.w_k = nn.Linear(id_embed_dim, qk_dim)
        self.func = models.ODEFunc_cNODE_GenRun()
        self.scale_factor = qk_dim ** -0.5
    
    def forward(self, t, x):
        val, pos = condense(x)
        
        # modify v
        id_embed = self.embed(pos)
        
        q = self.w_q(id_embed)
        k = self.w_k(id_embed)
        soft_weights = torch.einsum('...id , ...jd -> ...ij', q, k) * self.scale_factor
        fx = torch.einsum('...ij, ...i -> ...j', soft_weights, val)
        # intentionally not computing v from attention mechanism, since with a single head it could not produce both benefits and harms to fitness.
        # Without softmax and without v, the dot product of Q+K can be either positive or negative, increasing or decreasing fitness.
        # But multihead attention could solve that in a more flexible way.
        
        y = odeint(lambda xo,to: self.func(xo,to,fx), val, t)[-1]
        
        y = decondense(y, pos, self.data_dim)
        return y


class canODE_attention(nn.Module): # compositional attention nODE
    def __init__(self, data_dim, id_embed_dim, qk_dim):
        super(canODE_attention, self).__init__()
        self.data_dim = data_dim
        self.embed = nn.Embedding(data_dim + 1, id_embed_dim)  # Add 1 to account for placeholder ID
        self.w_q = nn.Linear(id_embed_dim, qk_dim)
        self.w_k = nn.Linear(id_embed_dim, qk_dim)
        self.w_v = nn.Linear(id_embed_dim, 1)
        self.func = models.ODEFunc_cNODE_GenRun()
    
    def forward(self, t, x):
        val, pos = condense(x)
        
        # modify v
        id_embed = self.embed(pos)
        
        q = self.w_q(id_embed)
        k = self.w_k(id_embed)
        v = self.w_v(id_embed)
        fx = nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=0.0).squeeze()  # TODO: try dropout on this and other models
        y = odeint(lambda xo,to: self.func(xo,to,fx), val, t)[-1]
        
        y = decondense(y, pos, self.data_dim)
        return y

class canODE_attentionMultihead(nn.Module): # compositional attention nODE
    def __init__(self, data_dim, id_embed_dim, num_heads):
        # As is standard in multihead attention, the Q+K embedding dimensions are equal to the full embed dim divided by number of attention heads. This contrasts with the above models where it was explicitly parameterized.
        super(canODE_attentionMultihead, self).__init__()
        self.data_dim = data_dim
        self.embed = nn.Embedding(data_dim + 1, id_embed_dim)  # Add 1 to account for placeholder ID
        self.attend = nn.MultiheadAttention(id_embed_dim, num_heads, batch_first=True, dropout=0.0)
        self.decode = nn.Linear(id_embed_dim, 1) # because pytorch's implementation doesn't support using a different embedding dim for V+Output than for Q+K
        self.func = models.ODEFunc_cNODE_GenRun()
    
    def forward(self, t, x):
        val, pos = condense(x)
        
        # modify v
        id_embed = self.embed(pos)
        
        mha = self.attend(id_embed, id_embed, id_embed, need_weights=False)[0]
        fx = self.decode(mha).squeeze();
        
        y = odeint(lambda xo,to: self.func(xo,to,fx), val, t)[-1]
        
        y = decondense(y, pos, self.data_dim)
        return y
    
# nn.TransformerEncoder + nn.TransformerEncoderLayer

class canODE_transformer(nn.Module):  # compositional attention nODE
    def __init__(self, data_dim, id_embed_dim, num_heads, depth=6, ffn_dim_multiplier=4):
        # As is standard in multihead attention, the Q+K embedding dimensions are equal to the full embed dim divided by number of attention heads. This contrasts with the above models where it was explicitly parameterized.
        super(canODE_transformer, self).__init__()
        self.data_dim = data_dim
        self.embed = nn.Embedding(data_dim + 1, id_embed_dim)  # Add 1 to account for placeholder ID
        encoder_layer = nn.TransformerEncoderLayer(d_model=id_embed_dim, nhead=num_heads, dim_feedforward=math.ceil(id_embed_dim*ffn_dim_multiplier),
            activation="gelu", batch_first=True, dropout=0.0)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.attend = nn.MultiheadAttention(id_embed_dim, num_heads, batch_first=True, dropout=0.0)
        self.decode = nn.Linear(id_embed_dim,1)
        self.func = models.ODEFunc_cNODE_GenRun()
    
    def forward(self, t, x):
        val, pos = condense(x)
        
        # modify v
        id_embed = self.embed(pos)
        
        h = self.transformer(id_embed)
        fx = self.decode(h).squeeze();
        
        y = odeint(lambda xo, to: self.func(xo, to, fx), val, t)[-1]
        
        y = decondense(y, pos, self.data_dim)
        return y
    
    
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
    E_dim = 5
    QK_dim = 4
    num_heads = 2
    # model = canODE_attention(N, E_dim, QK_dim)
    # model = canODE_attentionMultihead(N, QK_dim, num_heads)
    model = canODE_transformer(N, QK_dim, num_heads, depth=6, ffn_dim_multiplier=4)
    y_pred = model(t, x)
    print(y_pred)
    
    print(y_pred - y)