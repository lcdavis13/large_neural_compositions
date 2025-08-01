import torch
import torch.nn as nn
from introspection import construct
import model_commonblocks as blocks


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class Linear(nn.Module):
    def __init__(self, data_dim: int):
        super().__init__()
        self.linear = nn.Linear(data_dim, data_dim)

    def forward(self, x):
        return self.linear(x)


class LearnedConstantVector(nn.Module):
    """A learnable constant vector. Initializes to 1/N (the mean for simplex / relative abundance vectors)."""
    def __init__(self, data_dim: int):
        super().__init__()
        self.constant = nn.Parameter(torch.ones(data_dim) / data_dim)

    def forward(self, x):
        return self.constant.unsqueeze(0).expand(x.size(0), -1)


class ShallowMLP(nn.Module):
    """A simple single-hidden-layer MLP with GELU activation. None of the fancy features of the comparable ResidualMLPBlock."""
    def __init__(self, data_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(data_dim, hidden_dim)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, data_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
    @classmethod
    def init_1d(cls, width, **kwargs):

        override = {
            "hidden_dim": width,
        }

        return construct(cls, kwargs, override), override


class ShallowMLP2(nn.Module):
    """A simple single-hidden-layer MLP with GELU activation. None of the fancy features of the comparable ResidualMLPBlock."""
    def __init__(self, data_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(data_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, data_dim)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x
    
    @classmethod
    def init_1d(cls, width, **kwargs):

        override = {
            "hidden_dim": width,
        }

        return construct(cls, kwargs, override), override


class ResidualMLP(nn.Module):
    def __init__(self, 
                 data_dim: int, 
                 num_blocks: int, 
                 hidden_dim: int, 
                 dropout: float, 
                 learnable_skip: bool):
        """
        A deep residual MLP network composed of ResidualMLPBlocks. Each block expands the dimension, applies nonlinearity, shrinks it back, then skip connection.

        Args:
            dim (int): Input and output dimension of each block.
            num_blocks (int): Number of ResidualMLPBlocks.
            hidden_dim (int): Hidden dimension in the MLP.
            dropout (float): Dropout rate for MLP layers.
            learnable_skip (bool): Use learnable gated skip connections if True, vanilla skip otherwise.
        """
        super().__init__()
        self.blocks = nn.ModuleList([
            blocks.ResidualMLPBlock(residual_dim=data_dim, hidden_dim=hidden_dim, dropout=dropout, learnable_skip=learnable_skip, no_norm=(i<1))
            for i in range(num_blocks)
        ])
        
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x
    
    @classmethod
    def init_2d(cls, width, depth, **kwargs):

        override = {
            "num_blocks": depth,
            "hidden_dim": width,
        }

        return construct(cls, kwargs, override), override


class Transformer(nn.Module):
    def __init__(self, 
                 embed_dim: int, 
                 num_blocks: int, 
                 num_heads: int, 
                 fcn_dim_factor: float, 
                 attn_dropout: float, 
                 fcn_dropout: float, 
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
                fcn_dim_factor=fcn_dim_factor,
                attn_dropout=attn_dropout,
                fcn_dropout=fcn_dropout,
                learnable_skip=learnable_skip
            ) for _ in range(num_blocks)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
