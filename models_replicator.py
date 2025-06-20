import torch.nn as nn
import model_core as core
import model_wrappers_replicator as repwrap
import model_weightedAttention as wat

class ReplicatorIdentity(nn.Module):
    def __init__(self):
        self.USES_ODEINT = True
        super().__init__()

        self.core_model = core.Identity()
        self.replicator_model = repwrap.Replicator_CustomFitness(fitness_fn=self.core_model)

    def forward(self, t, x):
        return self.replicator_model(t, x)
    

class ReplicatorConstant(nn.Module):
    def __init__(self, data_dim, value=1.0):
        self.USES_ODEINT = True
        super().__init__()

        self.core_model = core.LearnedConstantVector(data_dim, value=value)
        self.replicator_model = repwrap.Replicator_CustomFitness(fitness_fn=self.core_model)

    def forward(self, t, x):
        return self.replicator_model(t, x)
    

class ReplicatorLinear(nn.Module):
    def __init__(self, data_dim):
        self.USES_ODEINT = True
        super().__init__()

        self.core_model = core.Linear(data_dim, data_dim)
        self.replicator_model = repwrap.Replicator_CustomFitness(fitness_fn=self.core_model)

    def forward(self, t, x):
        return self.replicator_model(t, x)
    

class ReplicatorShallowMLP(nn.Module):
    def __init__(self, data_dim, hidden_dim):
        self.USES_ODEINT = True
        super().__init__()

        self.core_model = core.ShallowMLP(
            data_dim=data_dim,
            hidden_dim=hidden_dim
        )
        self.replicator_model = repwrap.Replicator_CustomFitness(fitness_fn=self.core_model)

    def forward(self, t, x):
        return self.replicator_model(t, x)
    

class ReplicatorResidualMLP(nn.Module):
    def __init__(self, data_dim, depth, hidden_dim, dropout, learnable_skip):
        self.USES_ODEINT = True
        super().__init__()

        self.core_model = core.ResidualMLP(
            dim=data_dim,
            depth=depth,
            hidden_dim=hidden_dim,
            dropout=dropout,
            learnable_skip=learnable_skip
        )
        self.replicator_model = repwrap.Replicator_CustomFitness(fitness_fn=self.core_model)

    def forward(self, t, x):
        return self.replicator_model(t, x)
    

class ReplicatorTransformer(nn.Module):
    def __init__(self, data_dim, embed_dim, enrich_depth, fitness_depth, num_heads, mlp_dim_factor, attn_dropout, mlp_dropout, learnable_skip):
        self.USES_CONDENSED = True
        self.USES_ODEINT = True
        super().__init__()

        self.enrich_model = core.Transformer(
            embed_dim=embed_dim,
            depth=enrich_depth,
            num_heads=num_heads,
            mlp_dim_factor=mlp_dim_factor,
            attn_dropout=attn_dropout, 
            mlp_dropout=mlp_dropout,
            learnable_skip=learnable_skip,
        )
        self.fitness_model = core.Transformer(
            embed_dim=embed_dim,
            depth=fitness_depth,
            num_heads=num_heads,
            mlp_dim_factor=mlp_dim_factor,
            attn_dropout=attn_dropout,
            mlp_dropout=mlp_dropout,
            learnable_skip=learnable_skip,
        )
        self.replicator_model = repwrap.Replicator_CustomFitness_IdEmbedd_XEncode(
            fitness_fn=self.fitness_model,
            data_dim=data_dim,
            embed_dim=data_dim,
            enrich_fn=self.enrich_model
        )

    def forward(self, t, x, ids):
        return self.replicator_model(t, x, ids)
    

class ReplicatorWeightedAttention(nn.Module):
    def __init__(self, data_dim, embed_dim, enrich_depth, fitness_depth, num_heads, mlp_dim_factor, attn_dropout, mlp_dropout, learnable_skip):
        self.USES_CONDENSED = True
        self.USES_ODEINT = True
        super().__init__()

        self.enrich_model = core.Transformer(
            embed_dim=embed_dim,
            depth=enrich_depth,
            num_heads=num_heads,
            mlp_dim_factor=mlp_dim_factor,
            attn_dropout=attn_dropout, 
            mlp_dropout=mlp_dropout,
            learnable_skip=learnable_skip,
        )

        # Fitness model is a stack of Transformer blocks with Weighted attention block at the end. Since this isn't useful for non-replicator models, it isn't defined externally like the other models.

        fitness_model_head = wat.MultiheadWeightedAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_dim_factor=mlp_dim_factor,
            attn_dropout=attn_dropout,
            mlp_dropout=mlp_dropout,
        )

        if fitness_depth > 1:
            ode_transformer = core.Transformer(
                    embed_dim=embed_dim,
                    depth=fitness_depth - 1,
                    num_heads=num_heads,
                    mlp_dim_factor=mlp_dim_factor,
                    attn_dropout=attn_dropout,
                    mlp_dropout=mlp_dropout,
                    learnable_skip=learnable_skip,
                )
            self.fitness_model = nn.ModuleList([ode_transformer, fitness_model_head])
        else:
            self.fitness_model = fitness_model_head

        self.replicator_model = repwrap.Replicator_CustomFitness_IdEmbedd_XEncode(
            fitness_fn=self.fitness_model,
            data_dim=data_dim,
            embed_dim=data_dim,
            enrich_fn=self.enrich_model
        )

    def forward(self, t, x, ids):
        return self.replicator_model(t, x, ids)