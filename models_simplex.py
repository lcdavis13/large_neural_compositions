import torch.nn as nn
from introspection import construct
import models_core as core
import model_wrappers_simplex as simpwrap


class EmbeddedSimplexIdentity(nn.Module):
    def __init__(self, data_dim, embed_dim):
        self.USES_CONDENSED = True
        super().__init__()

        self.core_model = core.Identity()
        self.simplex_model = simpwrap.SimplexModel_IdEmbed(self.core_model, data_dim, embed_dim)
    
    def forward(self, x, ids):
        return self.simplex_model(x, ids)
    
    @classmethod
    def init_1d(cls, width, **kwargs):

        override = {
            "embed_dim": width,
        }

        return construct(cls, kwargs, override), override
    

class SimplexConstant(nn.Module):
    def __init__(self, data_dim):
        super().__init__()

        self.core_model = core.LearnedConstantVector(data_dim)
        self.simplex_model = simpwrap.SimplexModel(self.core_model)

    def forward(self, x):
        return self.simplex_model(x)
    
    
class SimplexLinear(nn.Module):
    def __init__(self, data_dim):
        super().__init__()

        self.core_model = core.Linear(data_dim)
        self.simplex_model = simpwrap.SimplexModel(self.core_model)

    def forward(self, x):
        return self.simplex_model(x)
    

class SimplexShallowMLP(nn.Module):
    def __init__(self, data_dim, hidden_dim):
        super().__init__()

        self.core_model = core.ShallowMLP(
            data_dim=data_dim,
            hidden_dim=hidden_dim
        )
        self.simplex_model = simpwrap.SimplexModel(self.core_model)
    
    def forward(self, x):
        return self.simplex_model(x)
    
    @classmethod
    def init_1d(cls, width, **kwargs):

        override = {
            "hidden_dim": width,
        }

        return construct(cls, kwargs, override), override
    

class SimplexResidualMLP(nn.Module):
    def __init__(self, data_dim, num_blocks, hidden_dim, dropout, learnable_skip):
        super().__init__()

        self.core_model = core.ResidualMLP(
            data_dim=data_dim,
            num_blocks=num_blocks,
            hidden_dim=hidden_dim,
            dropout=dropout,
            learnable_skip=learnable_skip
        )
        self.simplex_model = simpwrap.SimplexModel(self.core_model)
    
    def forward(self, x):
        return self.simplex_model(x)
    
    @classmethod
    def init_2d(cls, width, depth, **kwargs):

        override = {
            "num_blocks": depth,
            "hidden_dim": width,
        }

        return construct(cls, kwargs, override), override
    

class SimplexTransformer(nn.Module):
    def __init__(self, data_dim, embed_dim, num_blocks, num_heads, fcn_dim_factor, attn_dropout, fcn_dropout, learnable_skip):
        self.USES_CONDENSED = True
        
        super().__init__()

        self.core_model = core.Transformer(
            embed_dim=embed_dim,
            num_blocks=num_blocks,
            num_heads=num_heads,
            fcn_dim_factor=fcn_dim_factor,
            attn_dropout=attn_dropout,
            fcn_dropout=fcn_dropout,
            learnable_skip=learnable_skip
        )
        self.simplex_model = simpwrap.SimplexModel_IdEmbed(
            self.core_model, 
            data_dim, 
            embed_dim
        )
    
    def forward(self, x, ids):
        return self.simplex_model(x, ids)
    
    @classmethod
    def init(cls, **kwargs):
        override = {
            "embed_dim": max(kwargs["embed_dim"] // kwargs["num_heads"], 1) * kwargs["num_heads"],
        }
        return construct(cls, kwargs, override), override
    
    @classmethod
    def init_2d(cls, width, depth, **kwargs):
        num_heads = kwargs["num_heads"]

        override = {
            "embed_dim": max(width // num_heads, 1) * num_heads,
            "num_blocks": depth,
        }

        return construct(cls, kwargs, override), override
    
        
