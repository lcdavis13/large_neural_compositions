from typing import Type
import models_cNODE1
import models_core
import models_replicator
import models_simplex
from introspection import call, construct
from model_parameterizer import load_model_2d, load_model_1d
import torch.nn as nn


def get_model_classes():
    models = {
        'cNODE1': models_cNODE1.cNODE1,

        # identity
        # Don't include vanilla "identity" since it doesn't need training. Tested separately in the benchmarks.
        # The embedded versions of both wrappers have some modeling power on their own, so wrapping the identity gives us a benchmark of that. 
        # We don't need to test the non-embedded wrappers with identity, since both are equivalent to the pure identity.
        'EmbeddedSimplexIdentity': models_simplex.EmbeddedSimplexIdentity,
        'EmbeddedReplicatorIdentity': models_replicator.EmbeddedReplicatorIdentity,

        # learned constant vector
        'Constant': models_core.LearnedConstantVector,
        'SimplexConstant': models_simplex.SimplexConstant,
        'ReplicatorConstant': models_replicator.ReplicatorConstant,

        # linear
        'Linear': models_core.Linear,
        'SimplexLinear': models_simplex.SimplexLinear,
        'ReplicatorLinear': models_replicator.ReplicatorLinear,

        # shallow MLPs
        'ShallowMLP': models_core.ShallowMLP,
        'SimplexShallowMLP': models_simplex.SimplexShallowMLP,
        'ReplicatorShallowMLP': models_replicator.ReplicatorShallowMLP,

        # Residual MLPs
        'ResidualMLP': models_core.ResidualMLP,
        'SimplexResidualMLP': models_simplex.SimplexResidualMLP,
        'ReplicatorResidualMLP': models_replicator.ReplicatorResidualMLP,

        # Transformers
        # no "plain" transformer since embedding is required
        'SimplexTransformer': models_simplex.SimplexTransformer,
        'ReplicatorTransformer': models_replicator.ReplicatorTransformer,

        # Weighted Attention model
        # Currently only makes sense as a replicator model
        'ReplicatorWeightedAttention': models_replicator.ReplicatorWeightedAttention,

        # Transformer enrichment followed by MLP replicator
        'ReplicatorTransformerMLP': models_replicator.ReplicatorTransformerMLP,

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
        return call(factory_fn, args)

    # Final fallback: use constructor directly, assume no overrides
    return construct(model_cls, args), {}