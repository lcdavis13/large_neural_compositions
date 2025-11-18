import torch
import torch.nn as nn
import models_core as core
import model_wrappers_replicator as repwrap
import model_populationAttention as patt
from introspection import construct


class ReplicatorLinear(nn.Module): 
    def __init__(self, data_dim, learnable_skip, use_hofbauer: bool = False):
        super().__init__()
        self.USES_ODEINT = True

        ctor = lambda dim: core.Linear(dim)

        self.replicator_model = repwrap.Replicator_CustomFitness(
            fitness_fn_ctor=ctor,
            data_dim=data_dim,
            learnable_skip=learnable_skip,
            use_hofbauer=use_hofbauer,
        )

    def forward(self, t, x):
        return self.replicator_model(t, x)
    

class ReplicatorEmbeddedEncodedIdentity(nn.Module):
    def __init__(self, data_dim, embed_dim, use_logx, learnable_skip,
                 use_hofbauer: bool = False):
        super().__init__()
        self.USES_CONDENSED = True
        self.USES_ODEINT = True

        fitness_model = core.Identity()

        self.replicator_model = repwrap.Replicator_CustomFitness_IdEmbed_XEncode(
            core_fitness_fn=fitness_model,
            data_dim=data_dim,
            embed_dim=embed_dim,
            use_logx=use_logx,
            learnable_skip=learnable_skip,
            use_hofbauer=use_hofbauer,
        )

    def forward(self, t, x, ids):
        return self.replicator_model(t, x, ids)

    @classmethod
    def init_1d(cls, width, **kwargs):
        override = {"embed_dim": width}
        return construct(cls, kwargs, override), override


class EmbeddedEncodedReplicatorIdentity(nn.Module):
    def __init__(self, data_dim, embed_dim, use_logx, learnable_skip,
                 use_hofbauer: bool = False):
        super().__init__()
        self.USES_CONDENSED = True
        self.USES_ODEINT = True

        fitness_model = core.Identity()

        self.replicator_model = repwrap.Replicator_CustomFitness_IdEmbed_XEncode(
            core_fitness_fn=fitness_model,
            data_dim=data_dim,
            embed_dim=embed_dim,
            use_logx=use_logx,
            learnable_skip=learnable_skip,
            use_hofbauer=use_hofbauer,
        )

    def forward(self, t, x, ids):
        return self.replicator_model(t, x, ids)

    @classmethod
    def init_1d(cls, width, **kwargs):
        override = {"embed_dim": width}
        return construct(cls, kwargs, override), override


class ReplicatorConstant(nn.Module):
    def __init__(self, data_dim, learnable_skip, use_hofbauer: bool = False):
        super().__init__()
        self.USES_ODEINT = True

        ctor = lambda dim: core.LearnedConstantVector(dim)

        self.replicator_model = repwrap.Replicator_CustomFitness(
            fitness_fn_ctor=ctor,
            data_dim=data_dim,
            learnable_skip=learnable_skip,
            use_hofbauer=use_hofbauer,
        )

    def forward(self, t, x):
        return self.replicator_model(t, x)


class cNODE1(nn.Module):
    def __init__(self, data_dim):
        super().__init__()
        self.USES_ODEINT = True

        def ctor(dim):
            fcc = nn.Linear(dim, dim, bias=False)
            nn.init.zeros_(fcc.weight)
            return fcc

        self.replicator_model = repwrap.Replicator_CustomFitness(
            fitness_fn_ctor=ctor,
            data_dim=data_dim,
            learnable_skip=False,
            use_hofbauer=False,
        )

    def forward(self, t, x):
        return self.replicator_model(t, x)


class cNODE1_Hofbauer(nn.Module):
    def __init__(self, data_dim):
        super().__init__()
        self.USES_ODEINT = True

        def ctor(dim):
            fcc = nn.Linear(dim, dim, bias=False)
            nn.init.zeros_(fcc.weight)
            return fcc

        self.replicator_model = repwrap.Replicator_CustomFitness(
            fitness_fn_ctor=ctor,
            data_dim=data_dim,
            learnable_skip=False,
            use_hofbauer=True,
        )

    def forward(self, t, x):
        return self.replicator_model(t, x)


class cNODE1_HofbauerALR(nn.Module):
    def __init__(self, data_dim):
        super().__init__()
        self.USES_ODEINT = True

        def ctor(dim):
            fcc = nn.Linear(dim, dim, bias=False)
            nn.init.zeros_(fcc.weight)
            return fcc

        self.replicator_model = repwrap.ALR_Replicator_CustomFitness(
            fitness_fn_ctor=ctor,
            data_dim=data_dim,
            learnable_skip=False,
        )

    def forward(self, t, x):
        return self.replicator_model(t, x)

    
class ReplicatorSLP(nn.Module):
    def __init__(self, data_dim, hidden_dim, learnable_skip,
                 use_hofbauer: bool = False):
        super().__init__()
        self.USES_ODEINT = True

        ctor = lambda dim: core.SLP(data_dim=dim, hidden_dim=hidden_dim)

        self.replicator_model = repwrap.Replicator_CustomFitness(
            fitness_fn_ctor=ctor,
            data_dim=data_dim,
            learnable_skip=learnable_skip,
            use_hofbauer=use_hofbauer,
        )

    def forward(self, t, x):
        return self.replicator_model(t, x)

    @classmethod
    def init_1d(cls, width, **kwargs):
        override = {"hidden_dim": width}
        return construct(cls, kwargs, override), override


class ReplicatorBasicMLP(nn.Module):
    def __init__(self, data_dim, hidden_dim, learnable_skip, dropout,
                 use_hofbauer: bool = False):
        super().__init__()
        self.USES_ODEINT = True

        ctor = lambda dim: core.BasicMLP(
            data_dim=dim,
            hidden_dim=hidden_dim,
            dropout=dropout
        )

        self.replicator_model = repwrap.Replicator_CustomFitness(
            fitness_fn_ctor=ctor,
            data_dim=data_dim,
            learnable_skip=learnable_skip,
            use_hofbauer=use_hofbauer,
        )

    def forward(self, t, x):
        return self.replicator_model(t, x)

    @classmethod
    def init_1d(cls, width, **kwargs):
        override = {"hidden_dim": width}
        return construct(cls, kwargs, override), override


class ReplicatorResidualMLP(nn.Module):
    def __init__(self, data_dim, num_blocks, hidden_dim, dropout,
                 learnable_skip, use_hofbauer: bool = False):
        super().__init__()
        self.USES_ODEINT = True

        ctor = lambda dim: core.ResidualMLP(
            data_dim=dim,
            num_blocks=num_blocks,
            hidden_dim=hidden_dim,
            dropout=dropout,
            learnable_skip=learnable_skip,
        )

        self.replicator_model = repwrap.Replicator_CustomFitness(
            fitness_fn_ctor=ctor,
            data_dim=data_dim,
            learnable_skip=learnable_skip,
            use_hofbauer=use_hofbauer,
        )

    def forward(self, t, x):
        return self.replicator_model(t, x)

    @classmethod
    def init_2d(cls, width, depth, **kwargs):
        override = {"num_blocks": depth, "hidden_dim": width}
        return construct(cls, kwargs, override), override

    
class ReplicatorTransformer(nn.Module):
    def __init__(self, data_dim, embed_dim, enrich_blocks, fitness_blocks,
                 num_heads, fcn_dim_factor, attn_dropout, fcn_dropout,
                 learnable_skip, use_logx, use_hofbauer: bool = False):
        super().__init__()
        self.USES_CONDENSED = True
        self.USES_ODEINT = True

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

        fitness_model = core.Transformer(
            embed_dim=embed_dim,
            num_blocks=fitness_blocks,
            num_heads=num_heads,
            fcn_dim_factor=fcn_dim_factor,
            attn_dropout=attn_dropout,
            fcn_dropout=fcn_dropout,
            learnable_skip=learnable_skip,
        )

        self.replicator_model = repwrap.Replicator_CustomFitness_IdEmbed_XEncode(
            core_fitness_fn=fitness_model,
            data_dim=data_dim,
            embed_dim=embed_dim,
            enrich_fn=self.enrich_model,
            use_logx=use_logx,
            learnable_skip=learnable_skip,
            use_hofbauer=use_hofbauer,
        )

    def forward(self, t, x, ids):
        return self.replicator_model(t, x, ids)


class ReplicatorPopulationTransformer(nn.Module):
    def __init__(self, data_dim, embed_dim, enrich_blocks, fitness_blocks,
                 num_heads, fcn_dim_factor, attn_dropout, fcn_dropout,
                 learnable_skip, use_hofbauer: bool = False):
        super().__init__()
        self.USES_CONDENSED = True
        self.USES_ODEINT = True

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

        fitness_model = patt.PopulationTransformer(
            embed_dim=embed_dim,
            num_blocks=fitness_blocks,
            num_heads=num_heads,
            fcn_dim_factor=fcn_dim_factor,
            attn_dropout=attn_dropout,
            fcn_dropout=fcn_dropout,
            learnable_skip=learnable_skip,
        )

        self.replicator_model = repwrap.Replicator_CustomFitness_IdEmbed(
            fitness_fn=fitness_model,
            data_dim=data_dim,
            embed_dim=embed_dim,
            enrich_fn=self.enrich_model,
            learnable_skip=learnable_skip,
            use_hofbauer=use_hofbauer,
        )

    def forward(self, t, x, ids):
        return self.replicator_model(t, x, ids)
  

class ReplicatorWeightedAttention_NoXEncode(nn.Module):
    def __init__(self, data_dim, embed_dim, enrich_blocks, num_heads,
                 fcn_dim_factor, attn_dropout, fcn_dropout,
                 learnable_skip, use_hofbauer: bool = False):
        super().__init__()
        self.USES_CONDENSED = True
        self.USES_ODEINT = True

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

        fitness_model = patt.MultiheadPopulationAttention_NotResidual(
            embed_dim=embed_dim,
            num_heads=num_heads,
            fcn_dim_factor=fcn_dim_factor,
            attn_dropout=attn_dropout,
            fcn_dropout=fcn_dropout,
        )

        self.replicator_model = repwrap.Replicator_CustomFitness_IdEmbed(
            fitness_fn=fitness_model,
            data_dim=data_dim,
            embed_dim=embed_dim,
            enrich_fn=self.enrich_model,
            learnable_skip=learnable_skip,
            use_hofbauer=use_hofbauer,
        )

    def forward(self, t, x, ids):
        return self.replicator_model(t, x, ids)


class ReplicatorPopTransformer(nn.Module):
    def __init__(self, data_dim, embed_dim, enrich_blocks, fitness_blocks,
                 num_heads, fcn_dim_factor, attn_dropout, fcn_dropout,
                 learnable_skip, use_hofbauer: bool = False):
        super().__init__()
        self.USES_CONDENSED = True
        self.USES_ODEINT = True

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

        fitness_head = patt.MultiheadPopulationAttention_NotResidual(
            embed_dim=embed_dim,
            num_heads=num_heads,
            fcn_dim_factor=fcn_dim_factor,
            attn_dropout=attn_dropout,
            fcn_dropout=fcn_dropout,
        )

        if fitness_blocks > 1:
            transformer_stack = core.Transformer(
                embed_dim=embed_dim,
                num_blocks=fitness_blocks - 1,
                num_heads=num_heads,
                fcn_dim_factor=fcn_dim_factor,
                attn_dropout=attn_dropout,
                fcn_dropout=fcn_dropout,
                learnable_skip=learnable_skip,
            )
            fitness_model = nn.ModuleList([transformer_stack, fitness_head])
        else:
            fitness_model = fitness_head

        self.replicator_model = repwrap.Replicator_CustomFitness_IdEmbed(
            fitness_fn=fitness_model,
            data_dim=data_dim,
            embed_dim=embed_dim,
            enrich_fn=self.enrich_model,
            learnable_skip=learnable_skip,
            use_hofbauer=use_hofbauer,
        )

    def forward(self, t, x, ids):
        return self.replicator_model(t, x, ids)


