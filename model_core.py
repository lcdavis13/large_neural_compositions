import torch
import torch.nn as nn
import model_commonblocks as blocks


# aliases for "models" that are a single pytorch module
Identity = nn.Identity
Linear = nn.Linear


class LearnedConstantVector(nn.Module):
    """A learnable constant vector. Initializes to 1/N (the mean for simplex / relative abundance vectors)."""
    def __init__(self, dim: int):
        super().__init__()
        self.constant = nn.Parameter(torch.ones(dim) / dim)

    def forward(self, x):
        return self.constant.unsqueeze(0).expand(x.size(0), -1)


class ShallowMLP(nn.Module):
    """A simple single-hidden-layer MLP with GELU activation. None of the fancy features of the comparable ResidualMLPBlock."""
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        return x


class ResidualMLP(nn.Module):
    def __init__(self, 
                 dim: int, 
                 depth: int, 
                 hidden_dim: int, 
                 dropout: float, 
                 learnable_skip: bool):
        """
        A deep residual MLP network composed of ResidualMLPBlocks. Each block expands the dimension, applies nonlinearity, shrinks it back, then skip connection.

        Args:
            dim (int): Input and output dimension of each block.
            depth (int): Number of ResidualMLPBlocks.
            hidden_dim (int): Hidden dimension in the MLP.
            dropout (float): Dropout rate for MLP layers.
            learnable_skip (bool): Use learnable gated skip connections if True, vanilla skip otherwise.
        """
        super().__init__()
        self.blocks = nn.ModuleList([
            blocks.ResidualMLPBlock(dim=dim, hidden_dim=hidden_dim, dropout=dropout, learnable_skip=learnable_skip)
            for _ in range(depth)
        ])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class Transformer(nn.Module):
    def __init__(self, 
                 embed_dim: int, 
                 depth: int, 
                 num_heads: int, 
                 mlp_dim_factor: float, 
                 attn_dropout: float, 
                 mlp_dropout: float, 
                 learnable_skip: bool):
        """
        A transformer composed of stacked TransformerBlocks. Each block consists of multihead attention followed by MLP block.

        Args:
            embed_dim (int): Input and output embedding dimension for each block.
            depth (int): Number of TransformerBlocks to stack.
            num_heads (int): Number of attention heads in each block.
            mlp_dim_factor (float): Factor to scale the embed_dim for the MLP hidden layer.
            attn_dropout (float): Dropout rate used in the attention layers.
            mlp_dropout (float): Dropout rate used in the MLP layers.
            learnable_skip (bool): Use learnable gated skip connections if True, vanilla skip otherwise.
        """
        super().__init__()
        self.layers = nn.ModuleList([
            blocks.TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_dim_factor=mlp_dim_factor,
                attn_dropout=attn_dropout,
                mlp_dropout=mlp_dropout,
                learnable_skip=learnable_skip
            ) for _ in range(depth)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
