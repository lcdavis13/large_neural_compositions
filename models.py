import models_cNODE1
import models_core
import models_replicator
import models_simplex


def get_model_constructors():
    
    # Specify model constructors for experiment
    # Note that each must be a constructor function that takes a dicy/dictionary args. Lamda is recommended.
    models = {
        'cNODE1': lambda args: models_cNODE1.cNODE1(data_dim=args.data_dim),
        
        # identity
        # Don't include vanilla "identity" since it doesn't need training. Tested separately in the benchmarks.
        # The embedded versions of both wrappers have some modeling power on their own, so wrapping the identity gives us a benchmakr of that. 
        # We don't need to test the non-embedded wrappers with identity, since both are equivalent to the pure identity.
        'EmbeddedSimplexIdentity': lambda args: models_simplex.EmbeddedSimplexIdentity(args.data_dim, args.embed_dim),
        'EmbeddedReplicatorIdentity': lambda args: models_replicator.EmbeddedReplicatorIdentity(args.data_dim, args.embed_dim),

        # learned constant vector
        'Constant': lambda args: models_core.LearnedConstantVector(data_dim=args.data_dim),
        'SimplexConstant': lambda args: models_simplex.SimplexConstant(data_dim=args.data_dim),
        'ReplicatorConstant': lambda args: models_replicator.ReplicatorConstant(data_dim=args.data_dim),

        # linear
        'Linear': lambda args: models_core.Linear(data_dim=args.data_dim, out_dim=args.data_dim),
        'SimplexLinear': lambda args: models_simplex.SimplexLinear(data_dim=args.data_dim),
        'ReplicatorLinear': lambda args: models_replicator.ReplicatorLinear(data_dim=args.data_dim),

        # shallow MLPs
        'ShallowMLP': lambda args: models_core.ShallowMLP(data_dim=args.data_dim, hidden_dim=args.hidden_dim),
        'SimplexShallowMLP': lambda args: models_simplex.SimplexShallowMLP(data_dim=args.data_dim, hidden_dim=args.hidden_dim),
        'ReplicatorShallowMLP': lambda args: models_replicator.ReplicatorShallowMLP(data_dim=args.data_dim, hidden_dim=args.hidden_dim),

        # Residual MLPs
        'ResidualMLP': lambda args: models_core.ResidualMLP(
            dim=args.data_dim, 
            depth=args.depth, 
            hidden_dim=args.hidden_dim, 
            dropout=args.dropout, 
            learnable_skip=args.learnable_skip
        ),
        'SimplexResidualMLP': lambda args: models_simplex.SimplexResidualMLP(
            data_dim=args.data_dim, 
            depth=args.depth, 
            hidden_dim=args.hidden_dim, 
            dropout=args.dropout, 
            learnable_skip=args.learnable_skip
        ),
        'ReplicatorResidualMLP': lambda args: models_replicator.ReplicatorResidualMLP(
            data_dim=args.data_dim, 
            depth=args.depth, 
            hidden_dim=args.hidden_dim, 
            dropout=args.dropout, 
            learnable_skip=args.learnable_skip
        ),

        # Transformers
        # no "plain" transformer since embedding is required
        'SimplexTransformer': lambda args: models_simplex.SimplexTransformer(
            data_dim=args.data_dim, 
            embed_dim=args.embed_dim, 
            num_blocks=args.depth, 
            num_heads=args.num_heads, 
            mlp_dim_factor=args.ffn_dim_multiplier,
            attn_dropout=args.attn_dropout, 
            mlp_dropout=args.dropout, 
            learnable_skip=args.learnable_skip
        ),
        'ReplicatorTransformer': lambda args: models_replicator.ReplicatorTransformer(
            data_dim=args.data_dim, 
            embed_dim=args.embed_dim, 
            enrich_depth=args.enrich_depth, 
            fitness_depth=args.fitness_depth,
            num_heads=args.num_heads, 
            mlp_dim_factor=args.ffn_dim_multiplier,
            attn_dropout=args.attn_dropout, 
            mlp_dropout=args.dropout, 
            learnable_skip=args.learnable_skip
        ),

        # Weighted Attention model
        # Currently only makes sense as a replicator model
        'ReplicatorWeightedAttention': lambda args: models_replicator.ReplicatorWeightedAttention(
            data_dim=args.data_dim, 
            embed_dim=args.embed_dim, 
            enrich_depth=args.enrich_depth, 
            fitness_depth=args.fitness_depth,
            num_heads=args.num_heads, 
            mlp_dim_factor=args.ffn_dim_multiplier,
            attn_dropout=args.attn_dropout, 
            mlp_dropout=args.dropout, 
            learnable_skip=args.learnable_skip
        ),

        # Should include old versions of custom models for comparison? transformSoftmax, canODE-FitMat, canODE-attendFit
    }

    return models