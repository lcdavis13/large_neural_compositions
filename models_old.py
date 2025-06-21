import models_cNODE1
import models_core
import models_replicator
import models_simplex


def get_model_constructors():
    
    # Specify model constructors for experiment
    # Note that each must be a constructor function that takes a dicy/dictionary args. Lamda is recommended.
    models = {
        # most useful models
        'baseline-ConstSoftmax': lambda args: models_baseline.ConstOutputFilteredNormalized(args.data_dim, identity_gate=args.identity_gate),
        'baseline-SLPSoftmax': lambda args: models_baseline.SLPFilteredNormalized(args.data_dim, args.hidden_dim, identity_gate=args.identity_gate),
        'baseline-Linear': lambda args: models_baseline.LinearRegression(args.data_dim),
        'baseline-LinearSoftmax': lambda args: models_baseline.LinearFilteredNormalized(args.data_dim, identity_gate=args.identity_gate),
        'cNODE1': lambda args: models_cnode.cNODE1(args.data_dim, bias=args.cnode_bias, init_zero=args.cnode1_init_zero, identity_gate=args.identity_gate),
        'cNODE2': lambda args: models_cnode.cNODE2(args.data_dim, bias=True, identity_gate=args.identity_gate),
        'transformSoftmax': lambda args: models_embedded.TransformerNormalized(
            data_dim=args.data_dim, id_embed_dim=args.attend_dim, num_heads=args.num_heads, depth=args.depth,
            ffn_dim_multiplier=args.ffn_dim_multiplier, dropout=args.dropout, 
            identity_gate=args.identity_gate
        ),
        # 'transformRZSoftmax': lambda args: models_embedded.RZTransformerNormalized(
        #     data_dim=args.data_dim, id_embed_dim=args.attend_dim, num_heads=args.num_heads, depth=args.depth,
        #     ffn_dim_multiplier=args.ffn_dim_multiplier, dropout=args.dropout
        # ),
        'canODE-FitMat': lambda args: models_embedded.canODE_GenerateFitMat(
            data_dim=args.data_dim, id_embed_dim=args.attend_dim, num_heads=args.num_heads, depth=args.depth,
            ffn_dim_multiplier=args.ffn_dim_multiplier, fitness_qk_dim=args.attend_dim, dropout=args.dropout, 
            bias=args.cnode_bias, identity_gate=args.identity_gate
        ),
        'canODE-attendFit': lambda args: models_embedded.canODE_ReplicatorAttendFit(
            data_dim=args.data_dim, id_embed_dim=args.attend_dim, num_heads=args.num_heads, depth=args.depth,
            ffn_dim_multiplier=args.ffn_dim_multiplier, fitness_qk_dim=args.attend_dim, dropout=args.dropout, 
            identity_gate=args.identity_gate
        ),
        'cNODE-hourglass': lambda args: models_cnode.cNODE_HourglassFitness(
            data_dim=args.data_dim, hidden_dim=args.hidden_dim, depth=args.depth, bias=args.cnode_bias, identity_gate=args.identity_gate
        ),
        'baseline-cNODE0': lambda args: models_baseline.cNODE0(args.data_dim, init_zero=args.cnode1_init_zero, identity_gate=args.identity_gate),
        
    }

    return models