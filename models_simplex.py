import torch.nn as nn
import models_core as core
import model_wrappers_simplex as simpwrap


# class SimplexIdentity(nn.Module):
#     def __init__(self):
#         super().__init__()

#         self.core_model = core.Identity()
#         self.simplex_model = simpwrap.ResidualSimplexModel(self.core_model)

#     def forward(self, x):
#         return self.simplex_model(x)


class EmbeddedSimplexIdentity(nn.Module):
    def __init__(self, data_dim, embed_dim):
        self.USES_CONDENSED = True
        super().__init__()

        self.core_model = core.Identity()
        self.simplex_model = simpwrap.ResidualSimplexModel_IdEmbed(self.core_model, data_dim, embed_dim)

    def forward(self, x, ids):
        return self.simplex_model(x, ids)
    

class SimplexConstant(nn.Module):
    def __init__(self, data_dim):
        super().__init__()

        self.core_model = core.LearnedConstantVector(data_dim)
        self.simplex_model = simpwrap.ResidualSimplexModel(self.core_model)

    def forward(self, x):
        return self.simplex_model(x)
    
    
class SimplexLinear(nn.Module):
    def __init__(self, data_dim):
        super().__init__()

        self.core_model = core.Linear(data_dim, data_dim)
        self.simplex_model = simpwrap.ResidualSimplexModel(self.core_model)

    def forward(self, x):
        return self.simplex_model(x)
    

class SimplexShallowMLP(nn.Module):
    def __init__(self, data_dim, hidden_dim):
        super().__init__()

        self.core_model = core.ShallowMLP(
            data_dim=data_dim,
            hidden_dim=hidden_dim
        )
        self.simplex_model = simpwrap.ResidualSimplexModel(self.core_model)
    
    def forward(self, x, ids):
        return self.simplex_model(x, ids)
    

class SimplexResidualMLP(nn.Module):
    def __init__(self, data_dim, depth, hidden_dim, dropout, learnable_skip):
        super().__init__()

        self.core_model = core.ResidualMLP(
            dim=data_dim,
            depth=depth,
            hidden_dim=hidden_dim,
            dropout=dropout,
            learnable_skip=learnable_skip
        )
        self.simplex_model = simpwrap.ResidualSimplexModel(self.core_model)
    
    def forward(self, x, ids):
        return self.simplex_model(x, ids)
    

class SimplexTransformer(nn.Module):
    def __init__(self, data_dim, embed_dim, num_blocks, num_heads, mlp_dim_factor, attn_dropout, mlp_dropout, learnable_skip):
        self.USES_CONDENSED = True
        
        super().__init__()

        self.core_model = core.Transformer(
            embed_dim=embed_dim,
            depth=num_blocks,
            num_heads=num_heads,
            mlp_dim_factor=mlp_dim_factor,
            attn_dropout=attn_dropout,
            mlp_dropout=mlp_dropout,
            learnable_skip=learnable_skip
        )
        self.simplex_model = simpwrap.ResidualSimplexModel_IdEmbed(
            self.core_model, 
            data_dim, 
            embed_dim
        )
    
    def forward(self, x, ids):
        return self.simplex_model(x, ids)
    