
import models_baseline
import models_cnode
import models_embedded


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
        
        
        # additional baseline models
        'baseline-1const': lambda args: models_baseline.SingleConst(),
        'baseline-1constSoftmax': lambda args: models_baseline.SingleConstFilteredNormalized(),
        'baseline-const': lambda args: models_baseline.ConstOutput(args.data_dim),
        'baseline-SLPSumSoftmax': lambda args: models_baseline.SLPSumFilteredNormalized(args.data_dim, args.hidden_dim),
        'baseline-SLPMultSoftmax': lambda args: models_baseline.SLPMultFilteredNormalized(args.data_dim, args.hidden_dim, identity_gate=args.identity_gate),
        'baseline-SLPMultSumSoftmax': lambda args: models_baseline.SLPMultSumFilteredNormalized(args.data_dim, args.hidden_dim),
        'baseline-cNODE0-1step': lambda args: models_baseline.cNODE0_singlestep(args.data_dim, init_zero=args.cnode1_init_zero, identity_gate=args.identity_gate),
        'baseline-cNODE1-1step': lambda args: models_baseline.cNODE1_singlestep(args.data_dim, args.cnode_bias, init_zero=args.cnode1_init_zero, identity_gate=args.identity_gate),
        'baseline-cAttend-1step': lambda args: models_embedded.cAttend_simple(args.data_dim, args.attend_dim, args.attend_dim),
        'baseline-SLP-ODE': lambda args: models_baseline.SLPODE(args.data_dim, args.hidden_dim),
        'baseline-cNODE2-width1': lambda args: models_cnode.cNODE_HourglassFitness(
            data_dim=args.data_dim, hidden_dim=1, depth=2, bias=args.cnode_bias
        ),
        'baseline-cNODE2-width2': lambda args: models_cnode.cNODE_HourglassFitness(
            data_dim=args.data_dim, hidden_dim=2, depth=2, bias=args.cnode_bias
        ),
        
        
        # additional attention-based models
        'transformer': lambda args: models_embedded.JustATransformer(
            data_dim=args.data_dim, id_embed_dim=args.attend_dim, num_heads=args.num_heads, depth=args.depth,
            ffn_dim_multiplier=args.ffn_dim_multiplier, dropout=args.dropout
        ),
        'canODE-transformer': lambda args: models_embedded.canODE_transformer(args.data_dim, args.attend_dim, args.num_heads, args.depth, args.ffn_dim_multiplier),
        'canODE-noValue': lambda args: models_embedded.canODE_attentionNoValue(args.data_dim, args.attend_dim, args.attend_dim),
        'canODE-noValue-static': lambda args: models_embedded.canODE_attentionNoValue_static(args.data_dim, args.attend_dim, args.attend_dim),
        'canODE-attention': lambda args: models_embedded.canODE_attention(args.data_dim, args.attend_dim, args.attend_dim),
        'canODE-multihead': lambda args: models_embedded.canODE_attentionMultihead(args.data_dim, args.attend_dim, args.num_heads),
        
        
        # sanity test models
        'cNODE1-GenFn': lambda args: models_cnode.cNODE2_ExternalFitnessFn(args.data_dim, args.cnode_bias, init_zero=args.cnode1_init_zero, identity_gate=args.identity_gate), # for testing, identical to cNODE1
        'cNODE2-DKI': lambda args: models_cnode.cNODE2_DKI(args.data_dim, args.cnode_bias), # sanity test, this is the same as cNODE2 but less optimized
        'cNODE2-Gen': lambda args: models_cnode.cNODEGen_ConstructedFitness(lambda: nn.Sequential(nn.Linear(args.data_dim, args.data_dim, args.cnode_bias), nn.Linear(args.data_dim, args.data_dim, args.cnode_bias)), init_zero=args.cnode1_init_zero, identity_gate=args.identity_gate),  # sanity test, this is the same as cNODE2 but generated at runtime
        "cNODE2-static": lambda args: models_cnode.cNODE2_ExternalFitness(args.data_dim, args.cnode_bias, init_zero=args.cnode1_init_zero, identity_gate=args.identity_gate), # sanity test
        "cNODE2-FnFitness": lambda args: models_cnode.cNODE2_FnFitness(args.data_dim, args.cnode_bias, init_zero=args.cnode1_init_zero, identity_gate=args.identity_gate), # sanity test, this is the same as cNODE2 but testing externally-supplied fitness functions
    }

    return models