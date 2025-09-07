import torch.nn as nn
import models_core as core
import model_wrappers_replicator as repwrap
import model_populationAttention as patt
from introspection import construct
    

class ReplicatorEmbeddedEncodedIdentity(nn.Module):
    def __init__(self, data_dim, embed_dim, use_logx, learnable_skip):
        self.USES_CONDENSED = True
        self.USES_ODEINT = True
        super().__init__()

        self.fitness_model = core.Identity()
        self.replicator_model = repwrap.Replicator_CustomFitness_IdEmbed_XEncode(
            core_fitness_fn=self.fitness_model,
            data_dim=data_dim,
            embed_dim=embed_dim,
            use_logx=use_logx,
            learnable_skip=learnable_skip
        )

    def forward(self, t, x, ids):
        return self.replicator_model(t, x, ids)
    
    @classmethod
    def init_1d(cls, width, **kwargs):

        override = {
            "embed_dim": width,
        }

        return construct(cls, kwargs, override), override
    

class EmbeddedEncodedReplicatorIdentity(nn.Module):
    def __init__(self, data_dim, embed_dim, use_logx, learnable_skip):
        self.USES_CONDENSED = True
        self.USES_ODEINT = True
        super().__init__()

        self.fitness_model = core.Identity()
        self.replicator_model = repwrap.Replicator_CustomFitness_IdEmbed_XEncode(
            core_fitness_fn=self.fitness_model,
            data_dim=data_dim,
            embed_dim=embed_dim,
            use_logx=use_logx,
            learnable_skip=learnable_skip
        )

    def forward(self, t, x, ids):
        return self.replicator_model(t, x, ids)
    
    @classmethod
    def init_1d(cls, width, **kwargs):

        override = {
            "embed_dim": width,
        }

        return construct(cls, kwargs, override), override
    

class ReplicatorConstant(nn.Module):
    def __init__(self, data_dim, learnable_skip):
        self.USES_ODEINT = True
        super().__init__()

        self.core_model = core.LearnedConstantVector(data_dim)
        self.replicator_model = repwrap.Replicator_CustomFitness(fitness_fn=self.core_model, learnable_skip=learnable_skip)

    def forward(self, t, x):
        return self.replicator_model(t, x)
    

class ReplicatorLinear(nn.Module):
    def __init__(self, data_dim, learnable_skip):
        self.USES_ODEINT = True
        super().__init__()

        self.core_model = core.Linear(data_dim)
        self.replicator_model = repwrap.Replicator_CustomFitness(fitness_fn=self.core_model, learnable_skip=learnable_skip)

    def forward(self, t, x):
        return self.replicator_model(t, x)
    

class ReplicatorSLP(nn.Module):
    def __init__(self, data_dim, hidden_dim, learnable_skip):
        self.USES_ODEINT = True
        super().__init__()

        self.core_model = core.SLP(
            data_dim=data_dim,
            hidden_dim=hidden_dim,
        )
        self.replicator_model = repwrap.Replicator_CustomFitness(fitness_fn=self.core_model, learnable_skip=learnable_skip)

    def forward(self, t, x):
        return self.replicator_model(t, x)
    
    @classmethod
    def init_1d(cls, width, **kwargs):

        override = {
            "hidden_dim": width,
        }

        return construct(cls, kwargs, override), override
    

class ReplicatorBasicMLP(nn.Module):
    def __init__(self, data_dim, hidden_dim, learnable_skip, dropout):
        self.USES_ODEINT = True
        super().__init__()

        self.core_model = core.BasicMLP(
            data_dim=data_dim,
            hidden_dim=hidden_dim,
            dropout=dropout
        )
        self.replicator_model = repwrap.Replicator_CustomFitness(fitness_fn=self.core_model, learnable_skip=learnable_skip)

    def forward(self, t, x):
        return self.replicator_model(t, x)
    
    @classmethod
    def init_1d(cls, width, **kwargs):

        override = {
            "hidden_dim": width,
        }

        return construct(cls, kwargs, override), override
    

class ReplicatorResidualMLP(nn.Module):
    def __init__(self, data_dim, num_blocks, hidden_dim, dropout, learnable_skip):
        self.USES_ODEINT = True
        super().__init__()

        self.core_model = core.ResidualMLP(
            data_dim=data_dim,
            num_blocks=num_blocks,
            hidden_dim=hidden_dim,
            dropout=dropout,
            learnable_skip=learnable_skip
        )
        self.replicator_model = repwrap.Replicator_CustomFitness(fitness_fn=self.core_model, learnable_skip=learnable_skip)

    def forward(self, t, x):
        return self.replicator_model(t, x)
    
    @classmethod
    def init_2d(cls, width, depth, **kwargs):

        override = {
            "num_blocks": depth,
            "hidden_dim": width,
        }

        return construct(cls, kwargs, override), override
    

class ReplicatorTransformer(nn.Module):
    def __init__(self, data_dim, embed_dim, enrich_blocks, fitness_blocks, num_heads, fcn_dim_factor, attn_dropout, fcn_dropout, learnable_skip, use_logx): 
        self.USES_CONDENSED = True
        self.USES_ODEINT = True
 
        super().__init__()

        if enrich_blocks > 0:
            self.enrich_model = core.Transformer(
                embed_dim=embed_dim,
                num_blocks=enrich_blocks,
                num_heads=num_heads,
                fcn_dim_factor=fcn_dim_factor,
                attn_dropout=attn_dropout, 
                fcn_dropout=fcn_dropout,
                learnable_skip=learnable_skip,
            )
        else:
            self.enrich_model = None
        self.fitness_model = core.Transformer(
            embed_dim=embed_dim,
            num_blocks=fitness_blocks,
            num_heads=num_heads,
            fcn_dim_factor=fcn_dim_factor,
            attn_dropout=attn_dropout,
            fcn_dropout=fcn_dropout,
            learnable_skip=learnable_skip,
        )
        self.replicator_model = repwrap.Replicator_CustomFitness_IdEmbed_XEncode(
            core_fitness_fn=self.fitness_model,
            data_dim=data_dim,
            embed_dim=embed_dim,
            enrich_fn=self.enrich_model, 
            use_logx=use_logx, 
            learnable_skip=learnable_skip
        )

    def forward(self, t, x, ids):
        return self.replicator_model(t, x, ids)
    
    @classmethod
    def init(cls, **kwargs):
        override = {
            "embed_dim": max(kwargs["embed_dim"] // kwargs["num_heads"], 1) * kwargs["num_heads"],
        }
        return construct(cls, kwargs, override), override
    
    @classmethod
    def init_2d(cls, width, depth, **kwargs):
        num_heads = kwargs["num_heads"]
        depth_fraction = kwargs["depth_fraction"]

        depth_fitness = max(depth_fraction // depth, 1)

        override = {
            "embed_dim": max(width // num_heads, 1) * num_heads,
            "fitness_blocks": depth_fitness,
            "enrich_blocks": depth - depth_fitness,
        }

        return construct(cls, kwargs, override), override
    

class ReplicatorPopulationTransformer(nn.Module):
    def __init__(self, data_dim, embed_dim, enrich_blocks, fitness_blocks, num_heads, fcn_dim_factor, attn_dropout, fcn_dropout, learnable_skip):
        self.USES_CONDENSED = True
        self.USES_ODEINT = True
        super().__init__()

        if enrich_blocks > 0:
            self.enrich_model = core.Transformer(
                embed_dim=embed_dim,
                num_blocks=enrich_blocks,
                num_heads=num_heads,
                fcn_dim_factor=fcn_dim_factor,
                attn_dropout=attn_dropout, 
                fcn_dropout=fcn_dropout,
                learnable_skip=learnable_skip,
            )
        else:
            self.enrich_model = None

        self.fitness_model = patt.PopulationTransformer(
            embed_dim=embed_dim,
            num_blocks=fitness_blocks,
            num_heads=num_heads,
            fcn_dim_factor=fcn_dim_factor,
            attn_dropout=attn_dropout,
            fcn_dropout=fcn_dropout,            
            learnable_skip=learnable_skip,
        )

        self.replicator_model = repwrap.Replicator_CustomFitness_IdEmbed(
             fitness_fn=self.fitness_model,
            data_dim=data_dim,
            embed_dim=embed_dim,
            enrich_fn=self.enrich_model, 
            learnable_skip=learnable_skip
        )

    def forward(self, t, x, ids):
        return self.replicator_model(t, x, ids)
    

class ReplicatorWeightedAttention_NoXEncode(nn.Module):
    def __init__(
            self, data_dim, embed_dim, enrich_blocks, num_heads, 
            fcn_dim_factor, attn_dropout, fcn_dropout, learnable_skip
        ):
        self.USES_CONDENSED = True
        self.USES_ODEINT = True
        super().__init__()

        if enrich_blocks > 0:
            self.enrich_model = core.Transformer(
                embed_dim=embed_dim,
                num_blocks=enrich_blocks,
                num_heads=num_heads,
                fcn_dim_factor=fcn_dim_factor,
                attn_dropout=attn_dropout, 
                fcn_dropout=fcn_dropout,
                learnable_skip=learnable_skip,
            )
        else:
            self.enrich_model = None

        self.fitness_model = patt.MultiheadPopulationAttention_NotResidual(
            embed_dim=embed_dim,
            num_heads=num_heads,
            fcn_dim_factor=fcn_dim_factor,
            attn_dropout=attn_dropout,
            fcn_dropout=fcn_dropout,
        )

        self.replicator_model = repwrap.Replicator_CustomFitness_IdEmbed(
            fitness_fn=self.fitness_model,
            data_dim=data_dim,
            embed_dim=embed_dim,
            enrich_fn=self.enrich_model,
            learnable_skip=learnable_skip,
        )

    def forward(self, t, x, ids):
        return self.replicator_model(t, x, ids)
    
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
            "enrich_blocks": depth - 1,
        }

        return construct(cls, kwargs, override), override
    

class ReplicatorPopTransformer(nn.Module):
    def __init__(
            self, data_dim, embed_dim, enrich_blocks, fitness_blocks, 
            num_heads, fcn_dim_factor, attn_dropout, fcn_dropout, learnable_skip
        ):
        self.USES_CONDENSED = True
        self.USES_ODEINT = True
        super().__init__()

        if enrich_blocks > 0:
            self.enrich_model = core.Transformer(
                embed_dim=embed_dim,
                num_blocks=enrich_blocks,
                num_heads=num_heads,
                fcn_dim_factor=fcn_dim_factor,
                attn_dropout=attn_dropout, 
                fcn_dropout=fcn_dropout,
                learnable_skip=learnable_skip,
            )
        else:
            self.enrich_model = None

        # Fitness model is a stack of Transformer blocks (different from enrichment blocks b/c the abundance encodings are added to the embeddings) 
        # with Weighted attention block at the end. Since this isn't useful for non-replicator models, it isn't defined externally like the other models.

        fitness_model_head = patt.MultiheadPopulationAttention_NotResidual(
            embed_dim=embed_dim,
            num_heads=num_heads,
            fcn_dim_factor=fcn_dim_factor,
            attn_dropout=attn_dropout,
            fcn_dropout=fcn_dropout,
        )

        if fitness_blocks > 1:
            ode_transformer = core.Transformer(
                    embed_dim=embed_dim,
                    num_blocks=fitness_blocks - 1,
                    num_heads=num_heads,
                    fcn_dim_factor=fcn_dim_factor,
                    attn_dropout=attn_dropout,
                    fcn_dropout=fcn_dropout,
                    learnable_skip=learnable_skip,
                )
            self.fitness_model = nn.ModuleList([ode_transformer, fitness_model_head])
        else:
            self.fitness_model = fitness_model_head

        self.replicator_model = repwrap.Replicator_CustomFitness_IdEmbed(
            fitness_fn=self.fitness_model,
            data_dim=data_dim,
            embed_dim=embed_dim,
            enrich_fn=self.enrich_model,
            learnable_skip=learnable_skip,
        )

    def forward(self, t, x, ids):
        return self.replicator_model(t, x, ids)
    
    @classmethod
    def init(cls, **kwargs):
        override = {
            "embed_dim": max(kwargs["embed_dim"] // kwargs["num_heads"], 1) * kwargs["num_heads"],
        }
        return construct(cls, kwargs, override), override
    
    @classmethod
    def init_2d(cls, width, depth, **kwargs):
        num_heads = kwargs["num_heads"]
        depth_fraction = kwargs["depth_fraction"]

        depth_fitness = max(depth_fraction // depth, 1)

        override = {
            "embed_dim": max(width // num_heads, 1) * num_heads,
            "fitness_blocks": depth_fitness,
            "enrich_blocks": depth - depth_fitness,
        }

        return construct(cls, kwargs, override), override
    

# class ReplicatorTransformerMLP(nn.Module):
#     """
#     First enriches embeddings with a Transformer, then applies a Residual MLP to predict fitness.
#     """
#     def __init__(self, data_dim, embed_dim, enrich_blocks, hidden_dim, fitness_blocks, num_heads, mlp_dim_factor, attn_dropout, mlp_dropout, dropout, learnable_skip):
#         # self.USES_CONDENSED = True  # MLP doesn't make sense on condensed data. It can do it, but it isn't permutation invariant, so presumably all it can do is overfit. 
#         self.USES_ODEINT = True
#         super().__init__()

#         self.enrich_model = core.Transformer(
#             embed_dim=embed_dim,
#             num_blocks=enrich_blocks,
#             num_heads=num_heads,
#             mlp_dim_factor=mlp_dim_factor,
#             attn_dropout=attn_dropout, 
#             mlp_dropout=mlp_dropout,
#             learnable_skip=learnable_skip,
#         )

#         # If switching to condensed data, get sparse_data_dim from the constructor args. But since it's not permutation invariant that doesn't make much sense and would be forced to overfit, so we're using non-condensed data to avoid permutation. This means we use data_dim. But I'm keeping an assignment to sparse_data_dim as notes for the future.
#         sparse_data_dim = data_dim
        
#         self.fitness_model = core.ResidualMLP(
#             data_dim=sparse_data_dim * embed_dim,  
#             num_blocks=fitness_blocks,
#             hidden_dim=hidden_dim,
#             dropout=dropout,
#             learnable_skip=learnable_skip
#         )

#         # PROBLEM (another): the output of Transformer would need to be concatenated / flattened to work as input to the MLP, which would require a custom wrapper

#         self.replicator_model = repwrap.Replicator_CustomFitness_IdEmbed_XEncode(
#             fitness_fn=self.fitness_model,
#             data_dim=sparse_data_dim * embed_dim,
#             embed_dim=embed_dim,
#             enrich_fn=self.enrich_model
#         )

#     def forward(self, t, x, ids):
#         return self.replicator_model(t, x, ids)
    
#     @classmethod
#     def init(cls, **kwargs):
#         override = {
#             "embed_dim": max(kwargs["embed_dim"] // kwargs["num_heads"], 1) * kwargs["num_heads"],
#         }
#         return construct(cls, kwargs, override), override
    
#     @classmethod
#     def init_2d(cls, width, depth, **kwargs):
#         num_heads = kwargs["num_heads"]
#         depth_fraction = kwargs["depth_fraction"]
#         width_fraction = kwargs["width_fraction"]

#         depth_fitness = max(depth_fraction // depth, 1)

#         embed_dim = width ** width_fraction
#         embed_dim = max(embed_dim // num_heads, 1) * num_heads

#         hidden_dim_multiple = 32 # can change this, I just figure it's good to use multiples of 32 for this
#         hidden_dim = width ** (1 - width_fraction)
#         hidden_dim = max(hidden_dim // hidden_dim_multiple, 1) * hidden_dim_multiple

#         override = {
#             "embed_dim": embed_dim,
#             "hidden_dim": hidden_dim,
#             "fitness_blocks": depth_fitness,
#             "enrich_blocks": depth - depth_fitness,
#         }

#         return construct(cls, kwargs, override), override

