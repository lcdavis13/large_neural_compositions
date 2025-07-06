import torch.nn as nn
import model_skipgates as skips


class ResidualMLPBlock(nn.Module):
    def __init__(self, residual_dim, hidden_dim, dropout, learnable_skip, no_norm=False):
        """
        Args:
            dim (int): Input and output dimension.
            hidden_dim (int): Hidden dimension for the MLP.
            dropout (float): Dropout rate.
            learnable_gate (bool): Whether to use a learnable gated skip connection.
            no_norm (bool): Whether to skip layer normalization. Use this if the input is raw data that hasn't passed through any layer/projection yet.
        """
        super().__init__()
        if no_norm:
            self.layernorm = nn.Identity()
        else:
            self.layernorm = nn.LayerNorm(residual_dim)
        self.fc1 = nn.Linear(residual_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, residual_dim)
        if learnable_skip:
            self.skip = skips.GateSkip()
        else:
            self.skip = skips.StaticSkip()

    def forward(self, x):
        h = self.layernorm(x)
        h = self.fc1(h)
        h = self.dropout(h)
        h = self.gelu(h)
        h = self.fc2(h)
        return self.skip(h, x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout, learnable_skip):
        """
        Args:
            embed_dim (int): Embedding dimension.
            num_heads (int): Number of attention heads.
            dropout (float): Dropout rate for attention.
            learnable_skip (bool): Whether to use a learnable gated skip connection.
        """
        super().__init__()
        self.layernorm = nn.LayerNorm(embed_dim)
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        if learnable_skip:
            self.skip = skips.GateSkip()
        else:
            self.skip = skips.StaticSkip()

    def forward(self, x):
        h = self.layernorm(x)
        attn_output, _ = self.attention(h, h, h)
        h = self.skip(attn_output, x)
        return h


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, fcn_dim_factor, attn_dropout, fcn_dropout, learnable_skip):
        """
        Args:
            embed_dim (int): Embedding dimension.
            num_heads (int): Number of attention heads.
            mlp_dim_factor (float): Factor to compute MLP hidden dimension.
            attn_dropout (float): Dropout rate for attention.
            mlp_dropout (float): Dropout rate for MLP.
            learnable_skip (bool): Whether to use a learnable gated skip connection.
        """
        super().__init__()
        self.attention = ResidualAttentionBlock(embed_dim, num_heads, attn_dropout, learnable_skip)
        self.mlp = ResidualMLPBlock(embed_dim, int(embed_dim * fcn_dim_factor), fcn_dropout, learnable_skip)

    def forward(self, x):
        h = self.attention(x)
        h = self.mlp(h)
        return h
        
