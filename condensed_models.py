import torch
import torch.nn as nn
from torchdiffeq import odeint


def condense(x):
    batch_size = x.size(0)
    max_len = max(torch.sum(x[i] != 0).item() for i in range(batch_size))
    
    padded_values = torch.zeros(batch_size, max_len, dtype=x.dtype)
    padded_positions = torch.zeros(batch_size, max_len, dtype=torch.long)
    
    for i in range(batch_size):
        positions = torch.nonzero(x[i], as_tuple=True)[0]
        values = x[i][positions]
        len_values = len(values)
        
        padded_values[i, :len_values] = values
        padded_positions[i, :len_values] = positions + 1  # Adjust positions by 1
    
    return padded_values, padded_positions


def decondense(values, positions, size):
    batch_size = values.size(0)
    y = torch.zeros(batch_size, size, dtype=values.dtype)
    
    for i in range(batch_size):
        valid_positions = positions[i]  # Compensate for the adjustment
        valid_positions = valid_positions[valid_positions > 0]  # Remove any zero positions
        valid_positions -= 1  # Adjust positions back to zero-based indexing
        len_valid = len(valid_positions)
        
        y[i][valid_positions[:len_valid]] = values[i][:len_valid]
    
    return y


class cNODE_condensed(nn.Module):
    def __init__(self, data_dim, id_embed_dim):
        super(cNODE_condensed, self).__init__()
        self.data_dim = data_dim
        self.embed = nn.Embedding(data_dim+1, id_embed_dim)  # Add 1 to account for placeholder ID
    
    def forward(self, t, x):
        v,p = condense(x)
        
        # modify v
        id_embed = self.embed(p)
        
        y = decondense(v, p, self.data_dim)
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
    H = 3
    model = cNODE_condensed(N, H)
    y_pred = model(t, x)
    print(y_pred)
    
    print(y_pred - y)