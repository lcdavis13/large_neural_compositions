import torch
import torch.nn as nn
import math
from model_commonblocks import ResidualMLPBlock
from model_maskedSoftmax import MaskedSoftmax,WeightedSoftmax
from model_normedLog import normed_log
import model_skipgates as skips


class PopulationAttentionDispersed(nn.Module):
    def __init__(self, dim_qk: int, dropout: float):
        super().__init__()
        self.scale = dim_qk ** 0.5
        self.dropout = nn.Dropout(dropout)
        self.wsoftmax = WeightedSoftmax(dim=-2)

    def forward(self, Q, K, V, x):
        """
        Q, K: (B, H, L, d_k)
        V: (B, H, L, d_v)
        x: (B, L)
        Returns:
            O: (B, H, L, d_v)
        """
        B, H, L, _ = K.shape
        x_shaped = x.unsqueeze(1).unsqueeze(-1)  # (B, 1, L, 1)

        # Compute attention scores: S = Q @ K^T / sqrt(d_k)
        S = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (B, H, L, L)

        # Column-wise weighted softmax. x is not exponentiated to avoid issues with zeroes.
        A = self.wsoftmax(S, x_shaped)  # (B, H, L, L)

        # Apply dropout to attention weights
        A = self.dropout(A)  # This is the standard place to apply dropout in attention mechanisms, although it can cause the attention weights to deviate from the simplex, especially if sparse / high entropy.

        # Multiply V by x_j (elementwise): broadcast x to (B, 1, L, 1)
        V_scaled = x_shaped * V  # (B, H, L, d_v)

        # Weighted sum: (B, H, L, L) @ (B, H, L, d_v) → (B, H, L, d_v)
        O = torch.matmul(A, V_scaled)

        return O


class PopulationAttention(nn.Module):
    def __init__(self, dim_qk: int, dropout):
        super().__init__()
        self.scale = dim_qk ** 0.5
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V, x):
        # Q, K, V shape: (B, H, L, D)
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (B, H, L, L)
        scores = self.dropout(scores) # Apply dropout in same place as vanilla attention, except there's no softmax before it.

        # attn_weights = F.softmax(scores, dim=-1)
        V_scaled = V * x # row-wise scaling of V by x
        output = torch.matmul(scores, V_scaled)  # (B, H, L, D)

        return output


class MultiheadPopulationAttention_NotResidual(nn.Module):
    def __init__(self, embed_dim, num_heads, fcn_dim_factor, attn_dropout, fcn_dropout, dim_qk=None, dim_v=None, dispersion=True):
        self.USES_CONDENSED = True
        self.USES_ODEINT = True
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # default and validation of qk & v dimensions
        if dim_qk is None:
            dim_qk = embed_dim
        if dim_v is None:
            dim_v = embed_dim
        assert dim_qk % num_heads == 0, "dim_qk must be divisible by num_heads"
        assert dim_v % num_heads == 0, "dim_v must be divisible by num_heads"

        self.head_qk_dim = dim_qk // num_heads
        self.head_v_dim = dim_v // num_heads

        self.q_proj = nn.Linear(embed_dim, dim_qk)
        self.k_proj = nn.Linear(embed_dim, dim_qk)
        self.v_proj = nn.Linear(embed_dim, dim_v) 

        if dispersion:
            self.attention = PopulationAttentionDispersed(self.head_qk_dim, attn_dropout)
        else:
            self.attention = PopulationAttention(self.head_qk_dim, attn_dropout)

        # MLP
        hidden_dim = int(fcn_dim_factor * dim_v)
        self.fcn1 = nn.Linear(dim_v, hidden_dim)
        self.dropout = nn.Dropout(fcn_dropout)
        self.gelu = nn.GELU()
        self.fcn2 = nn.Linear(hidden_dim, 1)


    def forward(self, x, z):
        B, L, _ = z.size()

        Q = self.q_proj(z).view(B, L, self.num_heads, self.head_qk_dim).transpose(1, 2)
        K = self.k_proj(z).view(B, L, self.num_heads, self.head_qk_dim).transpose(1, 2)
        V = self.v_proj(z).view(B, L, self.num_heads, self.head_v_dim).transpose(1, 2)

        attn_output = self.attention(Q, K, V, x)

        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L, self.head_v_dim * self.num_heads)

        # Apply MLP
        # Layernorm before MLP? Not sure since it might corrupt the weighted scales.
        y = self.fcn1(attn_output)
        y = self.dropout(y)
        y = self.gelu(y)
        y = self.fcn2(y)
        y = y.squeeze(-1)  # Squeeze last dimension to get shape (B, L)

        return y


class MultiheadPopulationAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, attn_dropout, dispersion=True):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim) 
        self.o_proj = nn.Linear(embed_dim, embed_dim)

        if dispersion:
            self.attention = PopulationAttentionDispersed(attn_dropout)
        else:
            self.attention = PopulationAttention(attn_dropout)
        
    def forward(self, x, z):
        B, L, _ = z.size()

        Q = self.q_proj(z).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(z).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(z).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        attn_output = self.attention(Q, K, V, x)

        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L, self.head_dim * self.num_heads)

        # Apply MLP
        y = self.o_proj(attn_output)

        return y


class ResidualPopulationAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout, learnable_skip, dispersion=True):
        """
        Args:
            embed_dim (int): Embedding dimension.
            num_heads (int): Number of attention heads.
            dropout (float): Dropout rate for attention.
            learnable_skip (bool): Whether to use a learnable gated skip connection.
            dispersion (bool): Use PopulationAttentionDispersed if True, otherwise use PopulationAttention.
        """
        super(self).__init__()
        self.layernorm = nn.LayerNorm(embed_dim)
        self.attention = MultiheadPopulationAttention(embed_dim, num_heads, dropout=dropout, dispersion=dispersion)
        if learnable_skip:
            self.skip = skips.GateSkip()
        else:
            self.skip = skips.StaticSkip()

    def forward(self, x, z):
        h = self.layernorm(z)
        attn_output, _ = self.attention(h, h, h, x)
        h = self.skip(attn_output, z)
        return h
    

class PopulationTransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim_factor, attn_dropout, mlp_dropout, learnable_skip, dispersion=True):
        """
        Args:
            embed_dim (int): Embedding dimension.
            num_heads (int): Number of attention heads.
            mlp_dim_factor (float): Factor to compute MLP hidden dimension.
            attn_dropout (float): Dropout rate for attention.
            mlp_dropout (float): Dropout rate for MLP.
            dispersion (bool): Use PopulationAttentionDispersed if True, otherwise use PopulationAttention.
        """
        super().__init__()
        self.attention = ResidualPopulationAttentionBlock(embed_dim, num_heads, attn_dropout, learnable_skip, dispersion=dispersion)
        self.mlp = ResidualMLPBlock(embed_dim, int(embed_dim * mlp_dim_factor), mlp_dropout, learnable_skip)

    def forward(self, x, z):
        h = self.attention(x, z)
        h = self.mlp(h)
        return h
    

class PopulationTransformer(nn.Module):
    def __init__(self, embed_dim, num_blocks, num_heads, mlp_dim_factor, attn_dropout, mlp_dropout, learnable_skip, dispersion=True):
        """
        A transformer composed of stacked PopulationTransformerBlocks. Each block consists of multihead attention followed by MLP block.

        Args:
            embed_dim (int): Input and output embedding dimension for each block.
            num_blocks (int): Number of PopulationTransformerBlocks to stack.
            num_heads (int): Number of attention heads in each block.
            mlp_dim_factor (float): Factor to scale the embed_dim for the MLP hidden layer.
            attn_dropout (float): Dropout rate used in the attention layers.
            mlp_dropout (float): Dropout rate used in the MLP layers.
            learnable_skip (bool): Use learnable gated skip connections if True, vanilla skip otherwise.
            dispersion (bool): Use PopulationAttentionDispersed if True, otherwise use PopulationAttention.
        """
        super().__init__()
        self.layers = nn.ModuleList([
            PopulationTransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_dim_factor=mlp_dim_factor,
                attn_dropout=attn_dropout,
                mlp_dropout=mlp_dropout,
                learnable_skip=learnable_skip,
                dispersion=dispersion
            ) for _ in range(num_blocks)
        ])
        self.final_layer = nn.Linear(embed_dim, 1) # Final layer to project to output dimension

    def forward(self, x, z):
        h = z
        for layer in self.layers:
            h = layer(x, h)
        y = self.final_layer(h)
        y = y.squeeze(-1)  # Squeeze last dimension to get shape (B, L)
        return y
    

class IterativePopulationTransformer(nn.Module):
    def __init__(self, embed_dim, num_subblocks, num_blocks, num_heads, mlp_dim_factor, attn_dropout, mlp_dropout, learnable_skip, dispersion=True):
        """
        A stack of PopulationTransformers, each of which residually updates the population in logspace and then softmaxes it to return to a population.
        """
        super().__init__()
        self.num_steps = num_blocks
        self.pop_transforms = nn.ModuleList([
            PopulationTransformer(
                embed_dim=embed_dim,
                num_blocks=num_subblocks,
                num_heads=num_heads,
                mlp_dim_factor=mlp_dim_factor,
                attn_dropout=attn_dropout,
                mlp_dropout=mlp_dropout,
                learnable_skip=learnable_skip,
                dispersion=dispersion
            ) for _ in range(num_blocks)
        ])
        if learnable_skip:
            self.skips = nn.ModuleList([
                skips.GateSkip() for _ in range(num_blocks)
            ])
        else:
            self.skips = nn.ModuleList([
                skips.StaticSkip() for _ in range(num_blocks)
            ])
        self.masked_softmax = MaskedSoftmax(dim=-1)
    
    def forward(self, x, z):
        """
        x: Population abundances, shape (..., L)
        z: Population embeddings, shape (..., L, D)
        Returns:
            y: Updated population abundances
        """
        logx = normed_log(x)
        # Note: could also include a learnable offset for initial logx. Learnable scale isn't necessary because learnable skip of first layer will cover that.
        for l in range(self.num_steps):
            pop_transform = self.pop_transforms[l]
            skip = self.skips[l]

            # Update the population embeddings using the transformer
            log_dx = pop_transform(x, z)

            # Residually update logspace representation of population abundances
            logx = skip(log_dx, logx)

            # Apply softmax to get linear-space (relative) population abundances
            x = self.masked_softmax(logx, x)


# Note: pytorch and original AIAYN paper do dropout after the softmax inside the attention module, before multiplying against V. 
# Obviously this means it no longer sums to 1, but intriguingly the dropout also rescales the attention weights to counteract that. As a result, statistically it is more likely to sum to 1, but if it's sparse or approximately sparse you end up deviating wildly from 1, including possibly getting individual weights larger than 1. 
# Getting dropped weights and a valid softmax output would require using my masked softmax using Dropout(Ones()) as the mask to achieve that. It's unclear what the computational hit of my masked softmax is, so I'm a bit hesitant to abandon nn.MultiheadAttention for a custom version with masked dropout in my other transformer blocks. Especially because I would sacrifice their efficient C(?) implementation. But it's an area for future exploration.




if __name__ == "__main__":
    # Testing to ensure that the optimized WeightedAttentionDispersed matches the original (inefficient) matrix formulation

    class WeightedAttentionDispersed_NotEfficient(nn.Module):
        def __init__(self, dropout):
            super().__init__()
            self.dropout = nn.Dropout(dropout)

        def forward(self, Q, K, V, x):
            """
            Q, K: (..., L, D_q)
            V:    (..., L, D_v)
            x:    (..., L)
            Returns:
                O: (..., L, D_v)
            """
            D_k = K.size(-1)

            # (..., L, L)
            S = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(D_k, dtype=Q.dtype, device=Q.device))
            S_max = S.amax(dim=(-2, -1), keepdim=True)
            S_shifted = S - S_max

            # (..., L, L)
            E = torch.exp(S_shifted)

            # (..., L)
            d = torch.matmul(E.transpose(-2, -1), x.unsqueeze(-1)).squeeze(-1)
            w = x / d  # (..., L)

            # (..., L, L)
            D = torch.diag_embed(w)
            A = torch.matmul(E, D)  # (..., L, L)

            A = self.dropout(A)
            O = torch.matmul(A, V)  # (..., L, D_v)

            return O


    # Define batch shape and tensor dimensions
    batch_shape = (1, 3)  # e.g., (T, B), or just (B,) for single batch
    L = 5                # sequence length
    D_q = 8              # query/key dimension
    D_v = 4              # value dimension

    torch.manual_seed(42)

    # Create tensors with leading batch dims
    shape = batch_shape + (L, D_q)
    Q = torch.randn(*shape)
    K = torch.randn(*shape)
    V = torch.randn(*batch_shape, L, D_v)
    x = torch.rand(*batch_shape, L)

    # Initialize models
    model_eff = PopulationAttentionDispersed(dropout=0.0)
    model_mat = WeightedAttentionDispersed_NotEfficient(dropout=0.0)

    # Run both models
    O_eff = model_eff(Q, K, V, x)
    O_mat = model_mat(Q, K, V, x)

    # Compare outputs
    difference = torch.abs(O_eff - O_mat).max()
    identical = torch.allclose(O_eff, O_mat, atol=1e-6)

    print(f"Max difference: {difference.item()}")
    print(f"Outputs are identical: {identical}")
