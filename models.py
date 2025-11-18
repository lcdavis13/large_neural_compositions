from typing import Type
import models_cNODE
import models_core
import models_replicator
import models_simplex
from introspection import call, construct
from model_parameterizer import load_model_2d, load_model_1d
import torch.nn as nn


def get_model_classes():
    models = {
        'cNODE1': models_cNODE.cNODE1,
        'cNODE1_vanilla': models_replicator.cNODE1, 
        'cNODE1_hofbauer': models_replicator.cNODE1_Hofbauer, 
        'cNODE1_hofbauerALR': models_replicator.cNODE1_HofbauerALR,
        # 'cNODE1+1-proper': models_cNODE.glv1NODE,
        # 'cNODE1+1-colFrozen': models_cNODE.glv2NODE,
        # 'cNODE1+1-noFreeze': models_cNODE.envNODE,
        'cNODE2': models_cNODE.cNODE2,
        'cNODE2Improved': models_cNODE.cNODE2_NonLinear_Biased,

        # identity
        # Don't include vanilla "identity" since it doesn't need training. Tested separately in the benchmarks.
        # The embedded versions of both wrappers have some modeling power on their own, so wrapping the identity gives us a benchmark of that. 
        # We don't need to test the non-embedded wrappers with identity, since both are equivalent to the pure identity.
        'SimplexEmbeddedIdentity': models_simplex.SimplexEmbeddedIdentity,
        'ReplicatorEmbeddedEncodedIdentity': models_replicator.ReplicatorEmbeddedEncodedIdentity,

        # learned constant vector
        'Constant': models_core.LearnedConstantVector,
        'SimplexConstant': models_simplex.SimplexConstant,
        'ReplicatorConstant': models_replicator.ReplicatorConstant,

        # linear
        'Linear': models_core.Linear,
        'SimplexLinear': models_simplex.SimplexLinear,
        'ReplicatorLinear': models_replicator.ReplicatorLinear,

        # shallow MLPs
        'SLP': models_core.SLP,
        'SimplexSLP': models_simplex.SimplexSLP,
        'ReplicatorSLP': models_replicator.ReplicatorSLP,

        # slightly-less-shallow MLPs
        'BasicMLP': models_core.BasicMLP,
        'SimplexBasicMLP': models_simplex.SimplexBasicMLP,
        'ReplicatorBasicMLP': models_replicator.ReplicatorBasicMLP,

        # Residual MLPs
        'ResidualMLP': models_core.ResidualMLP,
        'SimplexResidualMLP': models_simplex.SimplexResidualMLP,
        'ReplicatorResidualMLP': models_replicator.ReplicatorResidualMLP,

        # Transformers
        # no "plain" transformer since embedding is required
        'SimplexTransformer': models_simplex.SimplexTransformer,
        'ReplicatorTransformer': models_replicator.ReplicatorTransformer,

        # Population-Weighted Attention models
        'ReplicatorPopTransformer': models_replicator.ReplicatorPopulationTransformer,
        'SimplexPopTransformer': models_simplex.SimplexPopTransformer,

        # Should include old versions of custom models for comparison? transformSoftmax, canODE-FitMat, canODE-attendFit
    }

    return models


def construct_model(model_cls: Type[nn.Module], args: dict):
    return construct(model_cls, args)


def construct_model_parameterized(model_cls: Type[nn.Module], parameter_target: int, width_depth_tradeoff: float, args: dict): 

    # Handle parameterized models with reparameterization
    if parameter_target > 0:
        factory_fn = getattr(model_cls, "init_2d", None)
        if callable(factory_fn):
            return load_model_2d(factory_fn, parameter_target, width_depth_tradeoff, args)
        
        factory_fn = getattr(model_cls, "init_1d", None)
        if callable(factory_fn):
            return load_model_1d(factory_fn, parameter_target, args)

    # Below here we are using non-parameterized initialization, either because no parameter 
    # target is specified, or because the model does not support parameterization

    # Fallback to standard init() if available, to retrieve overrides
    factory_fn = getattr(model_cls, "init", None)
    if callable(factory_fn):
        return call(factory_fn, args)  # BUG: can't use call() here because it will automatically 

    # Final fallback: use constructor directly, assume no overrides
    return construct(model_cls, args), {}