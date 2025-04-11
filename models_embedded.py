import math
import torch
import torch.nn as nn
# from rezero.transformer import RZTXEncoderLayer
from ode_solver import odeint
import models_cnode

# # TODO: Putting a setting here is a hacky shortcut. But ideally this should be addressed at the data preprocessing level instead of doing it dynamically, since it's expensive to do at runtime.
# #  The complication is that we then need to dynamically select from condensed or non-condensed versions of the dataset depending on the model being trained.
# #  We'd also have to use a fixed condensed dimension with padding to equalize them all, so it might be worth it to dynimcally trim off the last dimensions from batches even after moving the rest of the condensiing process to preprocessing.
# needs_condense = False
#

# def condense(x):
#     if needs_condense:
#
#         device = x.device
#         batch_size = x.size(0)
#         max_len = max(torch.sum(x[i] != 0).item() for i in range(batch_size))
#
#         padded_values = torch.zeros(batch_size, max_len, dtype=x.dtype, device=device)
#         padded_positions = torch.zeros(batch_size, max_len, dtype=torch.long, device=device)
#
#         for i in range(batch_size):
#             positions = torch.nonzero(x[i], as_tuple=True)[0]
#             values = x[i][positions]
#             len_values = len(values)
#
#             padded_values[i, :len_values] = values
#             padded_positions[i, :len_values] = positions + 1  # Adjust positions by 1
#
#         return padded_values, padded_positions
#
#     else:
#         pos = torch.arange(1, x.size(1) + 1, device=x.device)
#         # multiply by 0 or 1 depending on if the value is 0 or not
#         pos = pos * (x != 0).long()
#
#         return x, pos
#
#
# def decondense(values, positions, size):
#     """
#     Decondenses the values into a larger tensor according to the specified positions.
#     This function supports multiple batch dimensions, with only the last dimension
#     being a non-batch dimension.
#
#     Args:
#         values (torch.Tensor): Tensor of values to be decondensed. Shape: (..., N)
#         positions (torch.Tensor): Tensor of positions where the values should be placed. Shape: (..., N) or (extra_batch, ..., N)
#         size (int): The size of the last dimension of the output tensor.
#
#     Returns:
#         torch.Tensor: Decondensed tensor with shape (..., size).
#     """
#
#     if needs_condense:
#         device = values.device
#         batch_dims_values = values.shape[:-1]  # All batch dimensions of values except the last
#         batch_dims_positions = positions.shape[:-1]  # All batch dimensions of positions except the last
#
#         # Ensure positions has the same batch dimensions as values
#         if batch_dims_positions != batch_dims_values:
#             extra_batch_dims = batch_dims_positions[:-len(batch_dims_values)]
#             positions = positions.reshape(*extra_batch_dims, *batch_dims_values, positions.shape[-1])
#             positions = positions.expand(*batch_dims_values, positions.shape[-1])
#
#         y_shape = (*batch_dims_values, size)
#         y = torch.zeros(y_shape, dtype=values.dtype, device=device)  # Initialize with zeros
#
#         # Reshape to flatten batch dimensions
#         batch_size = values.numel() // values.shape[-1]  # Total batch size, treating multiple dimensions as one
#         values_flat = values.reshape(batch_size, -1)
#         positions_flat = positions.reshape(batch_size, -1)
#         y_flat = y.reshape(batch_size, size)
#
#         for i in range(batch_size):
#             valid_positions = positions_flat[i]
#             valid_positions = valid_positions[
#                 valid_positions >= 0]  # Remove invalid positions (assume non-negative are valid)
#             valid_positions = valid_positions.clamp(max=size - 1)  # Clamp positions within range [0, size-1]
#             y_flat[i][valid_positions] = values_flat[i][
#                                          :len(valid_positions)]  # Place the values at the valid positions
#
#         # Reshape y back to original batch dimensions
#         y = y_flat.view(*y_shape)
#         return y
#
#     else:
#         return values


class cAttend_simple(nn.Module):
    """
    No ODE, simplest model that only generates a fitness matrix from Q and K and tries to directly predict from it.
    """
    def __init__(self, data_dim, id_embed_dim, qk_dim):
        self.USES_CONDENSED = True
        super().__init__()
        self.data_dim = data_dim
        self.embed = nn.Embedding(data_dim+1, id_embed_dim)  # Add 1 to account for placeholder ID
        self.w_q = nn.Linear(id_embed_dim, qk_dim)
        self.w_k = nn.Linear(id_embed_dim, qk_dim)
        self.scale_factor = qk_dim ** -0.5
    
    def forward(self, t, val, pos):
        # val, pos = condense(x)
        
        # modify v
        id_embed = self.embed(pos)
        
        q = self.w_q(id_embed)
        k = self.w_k(id_embed)
        soft_weights = torch.einsum('...id , ...jd -> ...ij', q, k) * self.scale_factor
        fx = torch.einsum('...ij, ...i -> ...j', soft_weights, val)
        
        dv = torch.mul(val, fx)
        val = torch.add(val, dv) # Can also just do v += fitness but this adds a bit of ODE-like inductive bias
        
        # y = decondense(val, pos, self.data_dim)
        return val
    
    
class cAttentionNoValue(nn.Module):
    def __init__(self, id_embed_dim, qk_dim):
        super().__init__()
        self.w_q = nn.Linear(id_embed_dim, qk_dim)
        self.w_k = nn.Linear(id_embed_dim, qk_dim)
        self.scale_factor = qk_dim ** -0.5

    def forward(self, val, id_embed):
        h0 = torch.concat((val.unsqueeze(-1), id_embed), -1) # TODO: concat val onto id_embed. Adjust w_q and w_k to be +1 input dim.
        
        q = self.w_q(h0)
        k = self.w_k(h0)
        f_matrix = torch.einsum('...id , ...jd -> ...ij', q, k) * self.scale_factor
        # intentionally not computing v from attention mechanism, since with a single head it could not produce both benefits and harms to fitness.
        # Without softmax and without v, the dot product of Q+K can be either positive or negative, increasing or decreasing fitness.
        # But multihead attention could solve that in a more flexible way.
        
        fx = torch.einsum('...ij, ...j -> ...i', f_matrix, val)
        return fx
    
    
class cAttention(nn.Module):
    def __init__(self, id_embed_dim, qk_dim):
        super().__init__()
        self.w_q = nn.Linear(id_embed_dim, qk_dim)
        self.w_k = nn.Linear(id_embed_dim, qk_dim)
        self.w_v = nn.Linear(id_embed_dim, 1)

    def forward(self, val, id_embed):
        h0 = torch.concat((val.unsqueeze(-1), id_embed), -1) # TODO: concat val onto id_embed. Adjust w_q and w_k to be +1 input dim.
        
        q = self.w_q(h0)
        k = self.w_k(h0)
        v = self.w_v(h0)
        fx = nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=0.0).squeeze()  # TODO: try dropout on this and other models
    
        return fx


class canODE_attentionNoValue(nn.Module): # compositional attention nODE
    def __init__(self, data_dim, id_embed_dim, qk_dim, identity_gate):
        self.USES_ODEINT = True
        self.USES_CONDENSED = True
        
        super().__init__()
        self.data_dim = data_dim
        self.embed = nn.Embedding(data_dim + 1, id_embed_dim - 1)  # Add 1 to account for placeholder ID, subtract one to account for value concat while maintaining divisibility by num_heads
        attend = cAttentionNoValue(id_embed_dim, qk_dim)
        self.func = models_cnode.ODEFunc_cNODEGen_FnFitness_Args(attend, identity_gate=identity_gate)
    
    def forward(self, t, val, pos):
        # val, pos = condense(x)
        
        # modify v
        id_embed = self.embed(pos)
        
        y = odeint(lambda xo,to: self.func(xo, to, id_embed), val, t)
        
        # y = decondense(y, pos, self.data_dim)
        return y


class canODE_attentionNoValue_static(nn.Module): # compositional attention nODE
    def __init__(self, data_dim, id_embed_dim, qk_dim, identity_gate):
        self.USES_ODEINT = True
        self.USES_CONDENSED = True
        
        super().__init__()
        self.data_dim = data_dim
        self.embed = nn.Embedding(data_dim + 1, id_embed_dim)  # Add 1 to account for placeholder ID
        self.w_q = nn.Linear(id_embed_dim, qk_dim)
        self.w_k = nn.Linear(id_embed_dim, qk_dim)
        self.func = models_cnode.ODEFunc_cNODEGen_ExternalFitness(identity_gate=identity_gate)
        self.scale_factor = qk_dim ** -0.5
    
    def forward(self, t, val, pos):
        # val, pos = condense(x)
        
        # modify v
        id_embed = self.embed(pos)
        
        q = self.w_q(id_embed)
        k = self.w_k(id_embed)
        soft_weights = torch.einsum('...id , ...jd -> ...ij', q, k) * self.scale_factor
        fx = torch.einsum('...ij, ...i -> ...j', soft_weights, val)
        # intentionally not computing v from attention mechanism, since with a single head it could not produce both benefits and harms to fitness.
        # Without softmax and without v, the dot product of Q+K can be either positive or negative, increasing or decreasing fitness.
        # But multihead attention could solve that in a more flexible way.
        
        y = odeint(lambda xo,to: self.func(xo,to,fx), val, t)
        
        # y = decondense(y, pos, self.data_dim)
        return y


class canODE_attention(nn.Module): # compositional attention nODE
    def __init__(self, data_dim, id_embed_dim, qk_dim, identity_gate):
        self.USES_ODEINT = True
        self.USES_CONDENSED = True
        
        super().__init__()
        self.data_dim = data_dim
        self.embed = nn.Embedding(data_dim + 1, id_embed_dim - 1)  # Add 1 to account for placeholder ID, subtract one to account for value concat while maintaining divisibility by num_heads
        attend = cAttention(id_embed_dim, qk_dim)
        self.func = models_cnode.ODEFunc_cNODEGen_FnFitness_Args(attend, identity_gate=identity_gate)
    
    def forward(self, t, val, pos):
        # val, pos = condense(x)
        
        # modify v
        id_embed = self.embed(pos)
        
        y = odeint(lambda xo,to: self.func(xo, to, id_embed), val, t)
        
        # y = decondense(y, pos, self.data_dim)
        return y


class canODE_attention_static(nn.Module): # compositional attention nODE
    def __init__(self, data_dim, id_embed_dim, qk_dim, identity_gate):
        self.USES_ODEINT = True
        self.USES_CONDENSED = True
        
        super().__init__()
        self.data_dim = data_dim
        self.embed = nn.Embedding(data_dim + 1, id_embed_dim)  # Add 1 to account for placeholder ID
        self.w_q = nn.Linear(id_embed_dim, qk_dim)
        self.w_k = nn.Linear(id_embed_dim, qk_dim)
        self.w_v = nn.Linear(id_embed_dim, 1)
        self.func = models_cnode.ODEFunc_cNODEGen_ExternalFitness(identity_gate=identity_gate)
    
    def forward(self, t, val, pos):
        # val, pos = condense(x)
        
        # modify v
        id_embed = self.embed(pos)
        
        q = self.w_q(id_embed)
        k = self.w_k(id_embed)
        v = self.w_v(id_embed)
        fx = nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=0.0).squeeze()  # TODO: try dropout on this and other models
        y = odeint(lambda xo,to: self.func(xo,to,fx), val, t)
        
        # y = decondense(y, pos, self.data_dim)
        return y
    
    
class cAttentionMultihead(nn.Module):
    def __init__(self, id_embed_dim, num_heads):
        super().__init__()
        self.attend = nn.MultiheadAttention(id_embed_dim, num_heads, batch_first=True, dropout=0.0)
        self.decode = nn.Linear(id_embed_dim,1)  # because pytorch's implementation doesn't support using a different embedding dim for V+Output than for Q+K
    
    def forward(self, val, id_embed):
        h0 = torch.concat((val.unsqueeze(-1), id_embed), -1) # TODO: concat val onto id_embed. Adjust w_q and w_k to be +1 input dim.
        
        h1 = self.attend(h0, h0, h0, need_weights=False)[0]
        fx = self.decode(h1).squeeze()
        
        return fx


class canODE_attentionMultihead(nn.Module): # compositional attention nODE
    def __init__(self, data_dim, id_embed_dim, num_heads, identity_gate):
        self.USES_ODEINT = True
        self.USES_CONDENSED = True
        
        # As is standard in multihead attention, the Q+K embedding dimensions are equal to the full embed dim divided by number of attention heads. This contrasts with the above models where it was explicitly parameterized.
        super().__init__()
        self.data_dim = data_dim
        self.embed = nn.Embedding(data_dim + 1, id_embed_dim - 1)  # Add 1 to account for placeholder ID, subtract one to account for value concat while maintaining divisibility by num_heads
        attend = cAttentionMultihead(id_embed_dim, num_heads)
        self.func = models_cnode.ODEFunc_cNODEGen_FnFitness_Args(attend, identity_gate=identity_gate)
    
    def forward(self, t, val, pos):
        # val, pos = condense(x)
        
        # modify v
        id_embed = self.embed(pos)
        
        y = odeint(lambda xo,to: self.func(xo, to, id_embed), val, t)
        
        # y = decondense(y, pos, self.data_dim)
        return y


class canODE_attentionMultihead_static(nn.Module):  # compositional attention nODE
    def __init__(self, data_dim, id_embed_dim, num_heads, identity_gate):
        self.USES_ODEINT = True
        self.USES_CONDENSED = True
        
        # As is standard in multihead attention, the Q+K embedding dimensions are equal to the full embed dim divided by number of attention heads. This contrasts with the above models where it was explicitly parameterized.
        super().__init__()
        self.data_dim = data_dim
        self.embed = nn.Embedding(data_dim + 1, id_embed_dim)  # Add 1 to account for placeholder ID
        self.attend = nn.MultiheadAttention(id_embed_dim, num_heads, batch_first=True, dropout=0.0)
        self.decode = nn.Linear(id_embed_dim,
                                1)  # because pytorch's implementation doesn't support using a different embedding dim for V+Output than for Q+K
        self.func = models_cnode.ODEFunc_cNODEGen_ExternalFitness(identity_gate=identity_gate)
    
    def forward(self, t, val, pos):
        # val, pos = condense(x)
        
        # modify v
        id_embed = self.embed(pos)
        
        mha = self.attend(id_embed, id_embed, id_embed, need_weights=False)[0]
        fx = self.decode(mha).squeeze();
        
        y = odeint(lambda xo, to: self.func(xo, to, fx), val, t)
        
        # y = decondense(y, pos, self.data_dim)
        return y
    
    
class cAttentionTransformer(nn.Module):
    def __init__(self, id_embed_dim, num_heads, depth=6, ffn_dim_multiplier=4):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=id_embed_dim, nhead=num_heads,
                                                   dim_feedforward=math.ceil(id_embed_dim * ffn_dim_multiplier),
                                                   activation="gelu", batch_first=True, dropout=0.0)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.decode = nn.Linear(id_embed_dim, 1)
    
    def forward(self, val, id_embed):
        h0 = torch.concat((val.unsqueeze(-1), id_embed), -1) # TODO: concat val onto id_embed. Adjust w_q and w_k to be +1 input dim.
        
        h = self.transformer(h0)
        fx = self.decode(h).squeeze()
        
        return fx


class canODE_transformer(nn.Module):  # compositional attention nODE
    def __init__(self, data_dim, id_embed_dim, num_heads, depth=6, ffn_dim_multiplier=4, identity_gate=True):
        self.USES_ODEINT = True
        self.USES_CONDENSED = True
        
        # As is standard in multihead attention, the Q+K embedding dimensions are equal to the full embed dim divided by number of attention heads. This contrasts with the above models where it was explicitly parameterized.
        super().__init__()
        self.data_dim = data_dim
        self.embed = nn.Embedding(data_dim + 1, id_embed_dim - 1)  # Add 1 to account for placeholder ID, subtract one to account for value concat while maintaining divisibility by num_heads
        attend = cAttentionTransformer(id_embed_dim, num_heads, depth, ffn_dim_multiplier)
        self.func = models_cnode.ODEFunc_cNODEGen_FnFitness_Args(attend, identity_gate=identity_gate)
    
    def forward(self, t, val, pos):
        # val, pos = condense(x)
        
        # modify v
        id_embed = self.embed(pos)
        
        y = odeint(lambda xo,to: self.func(xo, to, id_embed), val, t)
        
        # y = decondense(y, pos, self.data_dim)
        return y


class canODE_transformer_static(nn.Module):  # compositional attention nODE
    def __init__(self, data_dim, id_embed_dim, num_heads, depth=6, ffn_dim_multiplier=4, identity_gate=True):
        self.USES_ODEINT = True
        self.USES_CONDENSED = True
        
        # As is standard in multihead attention, the Q+K embedding dimensions are equal to the full embed dim divided by number of attention heads. This contrasts with the above models where it was explicitly parameterized.
        super().__init__()
        self.data_dim = data_dim
        self.embed = nn.Embedding(data_dim + 1, id_embed_dim)  # Add 1 to account for placeholder ID
        encoder_layer = nn.TransformerEncoderLayer(d_model=id_embed_dim, nhead=num_heads,
                                                   dim_feedforward=math.ceil(id_embed_dim * ffn_dim_multiplier),
                                                   activation="gelu", batch_first=True, dropout=0.0)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.decode = nn.Linear(id_embed_dim, 1)
        self.func = models_cnode.ODEFunc_cNODEGen_ExternalFitness(identity_gate=identity_gate)
    
    def forward(self, t, val, pos):
        # val, pos = condense(x)
        
        # modify v
        id_embed = self.embed(pos)
        
        h = self.transformer(id_embed)
        fx = self.decode(h).squeeze()
        
        y = odeint(lambda xo, to: self.func(xo, to, fx), val, t)
        
        # y = decondense(y, pos, self.data_dim)
        return y


class canODE_GenerateFitMat(nn.Module):
    '''
    Version of canODE that learns to directly generate a fitness interaction matrix
    
    1. Create condensed embedding
    2. Enrich embeddings with transformer encoder
    3. Create a fitness interaction matrix using embedded scaled dot product similarity (a subcomponent of attention)
    4. Retrieve fitness biases from a learnable parameter
    5. Compose the fitness function as F(x) = M * x + b
    6. Run cNODE1 using the computed fitness function
    '''
    def __init__(self, data_dim, id_embed_dim, num_heads, depth, ffn_dim_multiplier, fitness_qk_dim, dropout, bias, identity_gate):
        self.USES_ODEINT = True
        self.USES_CONDENSED = True
        super().__init__()
        
        self.data_dim = data_dim
        self.bias = bias  # Store bias setting
        self.embed = nn.Embedding(data_dim + 1, id_embed_dim)  # Add 1 to account for placeholder ID
        
        # define the transformer
        encoder_layer = nn.TransformerEncoderLayer(d_model=id_embed_dim, nhead=num_heads,
                                                   dim_feedforward=math.ceil(id_embed_dim * ffn_dim_multiplier),
                                                   activation="gelu", batch_first=True, dropout=dropout)
        self.transform = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        # define the fitness function generators
        self.fitmatrix_Q = nn.Linear(id_embed_dim, fitness_qk_dim)
        self.fitmatrix_K = nn.Linear(id_embed_dim, fitness_qk_dim)
        self.fitmatrix_scalefactor = fitness_qk_dim ** -0.5
        
        if bias:
            self.fitbias = nn.Parameter(torch.zeros(data_dim + 1))  # Add 1 to account for placeholder ID
        else:
            self.fitbias = None  # No bias parameter if bias=False
        
        # define the ODE function
        self.ode_func = models_cnode.ODEFunc_cNODEGen_ExternalFitnessFn(identity_gate=identity_gate)
    
    def forward(self, t, val, pos):
        # val, pos = condense(x)
        
        # modify v
        id_embed = self.embed(pos)
        id_embed = self.transform(id_embed)

        q = self.fitmatrix_Q(id_embed)
        k = self.fitmatrix_K(id_embed)
        fitmatrix = torch.einsum('...id , ...jd -> ...ij', q, k) * self.fitmatrix_scalefactor
        
        # Conditionally include bias term
        if self.bias:
            fitbias = self.fitbias[pos]
            fitnessFn = lambda h: torch.einsum('...ij, ...i -> ...j', fitmatrix, h) + fitbias
        else:
            fitnessFn = lambda h: torch.einsum('...ij, ...i -> ...j', fitmatrix, h)
        
        y = odeint(lambda xo, to: self.ode_func(xo, to, fitnessFn), val, t)
        
        # y = decondense(y, pos, self.data_dim)
        return y


class canODE_ReplicatorAttendFit(nn.Module):
    '''
    Version of canODE that encodes the OTUs with a transformer, then uses an attention layer as the fitness function for the replicator ODE.
    
    1. Create condensed embedding
    2. Enrich embeddings with transformer encoder
    3. Define the fitness function as an attention mechanism with the encoded embeddings concatenated onto the abundances
    4. Run the Replicator ODE using the computed fitness function
    '''
    def __init__(self, data_dim, id_embed_dim, num_heads, depth, ffn_dim_multiplier, fitness_qk_dim, dropout, identity_gate):
        self.USES_ODEINT = True
        self.USES_CONDENSED = True
        super().__init__()
        
        self.data_dim = data_dim
        self.embed = nn.Embedding(data_dim + 1, id_embed_dim)  # Add 1 to account for placeholder ID
        
        # define the transformer
        encoder_layer = nn.TransformerEncoderLayer(d_model=id_embed_dim, nhead=num_heads,
                                                   dim_feedforward=math.ceil(id_embed_dim * ffn_dim_multiplier),
                                                   activation="gelu", batch_first=True, dropout=dropout)
        self.transform = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        self.compress = nn.Linear(id_embed_dim, fitness_qk_dim - 1) # minus 1 to account for value concat while maintaining divisibility by num_heads
        
        attend = cAttentionMultihead(fitness_qk_dim, num_heads)
        self.func = models_cnode.ODEFunc_cNODEGen_FnFitness_Args(attend, identity_gate=identity_gate)
    
    def forward(self, t, val, pos):
        # val, pos = condense(x)
        
        # modify v
        id_embed = self.embed(pos)
        id_embed = self.transform(id_embed)
        
        # compress transformer embeddings for the ODE attention mechanism
        id_embed = self.compress(id_embed)
        
        y = odeint(lambda xo, to: self.func(xo, to, id_embed), val, t)
        
        # y = decondense(y, pos, self.data_dim)
        return y


# class canODE_AttendODE(nn.Module):
#     '''
#     Version of canODE that encodes the OTUs with a transformer, then uses an attention layer as the ODE
#
#     1. Create condensed embedding
# 2. Enrich embeddings with transformer encoder
#     3. Solve an ODE of the attention mechanism and linear layer with the encoded embeddings concatenated onto the abundances
#     (intuition for step 3 being that finding the fixed point of the final layer will act as an inductive bias toward replicator-equation-like dynamics, even though we don't explicitly use the replicator equation)
#     '''


class JustATransformer(nn.Module):
    '''
    Use a transformer to directly predict the final relative abundances from the input.
    The only thing unconventional for transformers is basics of working with our data structure:
    - We concatenate the value onto the ID embedding as a separate channel, and extract a single channel to use as the predicted values
    - We condense & decondense the sequence if applicable
    '''
    def __init__(self, data_dim, id_embed_dim, num_heads, depth, ffn_dim_multiplier, dropout):
        self.USES_CONDENSED = True
        super().__init__()
        
        self.data_dim = data_dim
        self.embed = nn.Embedding(data_dim + 1, id_embed_dim - 1)  # Add 1 to account for placeholder ID, subtract one to account for value concat while maintaining divisibility by num_heads
        
        # define the transformer
        encoder_layer = nn.TransformerEncoderLayer(d_model=id_embed_dim, nhead=num_heads,
                                                   dim_feedforward=math.ceil(id_embed_dim * ffn_dim_multiplier),
                                                   activation="gelu", batch_first=True, dropout=dropout)
        self.transform = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
    
    def forward(self, val, pos):
        # val, pos = condense(x)
        
        # modify v
        id_embed = self.embed(pos)
        
        # concatenate the value onto the id embedding
        h0 = torch.cat((val.unsqueeze(-1), id_embed), -1)
        
        h = self.transform(h0)
        
        # extract the value from the transformer output
        y = h[..., 0]
        
        # y = decondense(y, pos, self.data_dim)
        return y
    
class TransformerNormalized(nn.Module):
    '''
    Use a transformer to directly predict the final relative abundances from the input.
    This model normalizes the output to sum to 1 in addition to the following:
    - We concatenate the value onto the ID embedding as a separate channel, and extract a single channel to use as the predicted values
    - We condense & decondense the sequence if applicable
    '''
    def __init__(self, data_dim, id_embed_dim, num_heads, depth, ffn_dim_multiplier, dropout, identity_gate):
        self.USES_CONDENSED = True
        super().__init__()
        
        self.data_dim = data_dim
        self.embed = nn.Embedding(data_dim + 1, id_embed_dim - 1)  # Add 1 to account for placeholder ID, subtract one to account for value concat while maintaining divisibility by num_heads
        
        # define the transformer
        encoder_layer = nn.TransformerEncoderLayer(d_model=id_embed_dim, nhead=num_heads,
                                                   dim_feedforward=math.ceil(id_embed_dim * ffn_dim_multiplier),
                                                   activation="gelu", batch_first=True, dropout=dropout)
        self.transform = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        if identity_gate:
            self.gateA = nn.Parameter(torch.tensor(0.0))
            self.gateB = nn.Parameter(torch.tensor(1.0))
        else:
            self.register_buffer('gateA', torch.tensor(1.0))
            self.register_buffer('gateB', torch.tensor(0.0))
    
    def forward(self, val, pos):
        # val, pos = condense(x)
        
        # modify v
        id_embed = self.embed(pos)
        
        # concatenate the value onto the id embedding
        h0 = torch.cat((val.unsqueeze(-1), id_embed), -1)
        
        h = self.transform(h0)
        
        # extract the value from the transformer output
        y_raw = h[..., 0]

        
        gated_y = self.gateA*y_raw + self.gateB*val # TODO: Proper rezero implementation, this is a cheap hack to get the Identity initialization without any internal layer-wise benefits
        
        # softmax
        y = self.masked_softmax(pos, gated_y)
        
        # y = decondense(y, pos, self.data_dim)
        return y
    
    def masked_softmax(self, pos, y_raw):
        # Mask out elements where pos is 0
        mask = (pos != 0)
        masked_y_raw = y_raw.masked_fill(~mask, float('-inf'))
        # Normalize the output to sum to 1 (excluding masked elements)
        y = nn.functional.softmax(masked_y_raw, dim=-1)
        # Set masked elements to 0
        y = y * mask.float()
        return y
    
class RZTransformerNormalized(nn.Module):
    '''
    Use a re-zero transformer to directly predict the final relative abundances from the input.
    This model normalizes the output to sum to 1 in addition to the following:
    - We concatenate the value onto the ID embedding as a separate channel, and extract a single channel to use as the predicted values
    - We condense & decondense the sequence if applicable
    '''
    def __init__(self, data_dim, id_embed_dim, num_heads, depth, ffn_dim_multiplier, dropout):
        self.USES_CONDENSED = True
        super().__init__()
        
        self.data_dim = data_dim
        self.embed = nn.Embedding(data_dim + 1, id_embed_dim - 1)  # Add 1 to account for placeholder ID, subtract one to account for value concat while maintaining divisibility by num_heads
        
        # define the transformer
        encoder_layer = RZTXEncoderLayer(d_model=id_embed_dim, nhead=num_heads,
                                                   dim_feedforward=math.ceil(id_embed_dim * ffn_dim_multiplier),
                                                   activation="gelu", batch_first=True, dropout=dropout)
        self.transform = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
    
    def forward(self, val, pos):
        # val, pos = condense(x)
        
        # modify v
        id_embed = self.embed(pos)
        
        # concatenate the value onto the id embedding
        h0 = torch.cat((val.unsqueeze(-1), id_embed), -1)
        
        h = self.transform(h0)
        
        # extract the value from the transformer output
        y_raw = h[..., 0]
        
        # softmax
        y = self.masked_softmax(pos, y_raw)
        
        # y = decondense(y, pos, self.data_dim)
        return y
    
    def masked_softmax(self, pos, y_raw):
        # Mask out elements where pos is 0
        mask = (pos != 0)
        masked_y_raw = y_raw.masked_fill(~mask, float('-inf'))
        # Normalize the output to sum to 1 (excluding masked elements)
        y = nn.functional.softmax(masked_y_raw, dim=-1)
        # Set masked elements to 0
        y = y * mask.float()
        return y


class TransformerSoftmax(nn.Module):
    '''
    Use a transformer to directly predict the final relative abundances from the input.
    This model normalizes the output to sum to 1 in addition to the following:
    - We concatenate the value onto the ID embedding as a separate channel, and extract a single channel to use as the predicted values
    - We condense & decondense the sequence if applicable
    '''
    def __init__(self, data_dim, id_embed_dim, num_heads, depth, ffn_dim_multiplier, dropout):
        self.USES_CONDENSED = True
        super().__init__()
        
        self.data_dim = data_dim
        self.embed = nn.Embedding(data_dim + 1, id_embed_dim - 1)  # Add 1 to account for placeholder ID, subtract one to account for value concat while maintaining divisibility by num_heads
        
        # define the transformer
        encoder_layer = nn.TransformerEncoderLayer(d_model=id_embed_dim, nhead=num_heads,
                                                   dim_feedforward=math.ceil(id_embed_dim * ffn_dim_multiplier),
                                                   activation="gelu", batch_first=True, dropout=dropout)
        self.transform = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
    
    def forward(self, val, pos):
        # val, pos = condense(x)
        
        # modify v
        id_embed = self.embed(pos)
        
        # concatenate the value onto the id embedding
        h0 = torch.cat((val.unsqueeze(-1), id_embed), -1)
        
        h = self.transform(h0)
        
        # extract the value from the transformer output
        y = h[..., 0]
        
        # normalize the output to sum to 1
        y = nn.functional.softmax(y, dim=-1)
        
        # y = decondense(y, pos, self.data_dim)
        return y
    
    
if __name__ == '__main__': # test
    B = 6
    
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
    x,y,_ = data.get_batch_raw(x_all, y_all, B, 0)
    _, N = x.shape
    
    print(y)
    print(x)

    t = torch.linspace(0, 1, 2)
    E_dim = 5
    QK_dim = 4
    num_heads = 2
    # model = canODE_attentionNoValue(N, E_dim, QK_dim)
    # model = canODE_attentionNoValue_static(N, E_dim, QK_dim)
    model = canODE_attention(N, E_dim, QK_dim)
    # model = canODE_attentionMultihead(N, QK_dim, num_heads)
    # model = canODE_transformer(N, QK_dim, num_heads, depth=6, ffn_dim_multiplier=4)
    y_pred = model(t, x)
    print(y_pred)
    
    print(y_pred - y)