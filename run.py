import os

import epoch_managers

# Set a default value if the environment variable is not specified
# os.environ.setdefault("SOLVER", "torchdiffeq")
# os.environ.setdefault("SOLVER", "torchdiffeq_memsafe")
# os.environ.setdefault("SOLVER", "torchode")
# os.environ.setdefault("SOLVER", "torchode_memsafe")
os.environ.setdefault("SOLVER", "trapezoid")

import itertools
import time

import pandas as pd
import torch
from dotsy import dicy
import torch.nn as nn

import data
import models_cnode
import models_baseline
import models_embedded
import stream
from optimum import Optimum, summarize, unrolloptims
from stream_plot import plotstream
import user_confirmation as ui
from hyperparams import HyperparameterBuilder

import experiment as expt




def loss_bc_dki(y_pred, y_true):
    return torch.sum(torch.abs(y_pred - y_true)) / torch.sum(
        torch.abs(y_pred + y_true))  # DKI repo implementation (incorrect)


def loss_bc(y_pred, y_true):  # Bray-Curtis Dissimilarity
    return torch.mean(torch.sum(torch.abs(y_pred - y_true), dim=-1) / torch.sum(torch.abs(y_pred) + torch.abs(y_true), dim=-1))


def loss_logbc(y_pred, y_true):  # Bray-Curtis Dissimilarity on log-transformed data to emphasize loss of rare species
    return loss_bc(torch.log(y_pred + 1), torch.log(y_true + 1))


def loss_loglogbc(y_pred, y_true):  # Bray-Curtis Dissimilarity on log-log-transformed data to emphasize loss of rare species even more
    return loss_logbc(torch.log(y_pred + 1), torch.log(y_true + 1))


def loss_bc_old(y_pred, y_true):  # Bray-Curtis Dissimilarity
    return torch.sum(torch.abs(y_pred - y_true)) / torch.sum(torch.abs(y_pred) + torch.abs(y_true))


def loss_bc_scaled(y_pred, y_true, epsilon=1e-10):
    numerator = torch.sum(torch.abs(y_pred - y_true) / (torch.abs(y_true) + epsilon), dim=-1)
    denominator = torch.sum(torch.abs(y_pred) + torch.abs(y_true) / (torch.abs(y_true) + epsilon), dim=-1)
    return torch.mean(numerator / denominator)


def loss_bc_root(y_pred, y_true):
    return torch.sqrt(loss_bc(y_pred, y_true))


def loss_bc_logscaled(y_pred, y_true, epsilon=1e-10):
    numerator = torch.sum(torch.abs(y_pred - y_true) / torch.log(torch.abs(y_true) + 1 + epsilon))
    denominator = torch.sum(torch.abs(y_pred) + torch.abs(y_true) / torch.log(torch.abs(y_true) + 1 + epsilon))
    return numerator / denominator


def loss_bc_unbounded(y_pred, y_true, avg_richness, epsilon=1e-10):
    # performs the normalization per element, such that if y_pred has an extra elemen, it adds an entire 1 to the loss. This avoids the "free lunch" of adding on extra elements with small value.
    batch_loss = torch.sum(torch.div(torch.abs(y_pred - y_true), torch.abs(y_pred) + torch.abs(y_true) + epsilon))
    batch_loss = batch_loss / avg_richness
    return batch_loss / y_pred.shape[0]


def distribution_error(x):  # penalties for invalid distributions
    a = 1.0
    b = 1.0
    feature_penalty = torch.sum(torch.clamp(torch.abs(x - 0.5) - 0.5, min=0.0))  # each feature penalized for distance from range [0,1]. Currently not normalized.
    sum_penalty = torch.sum(torch.abs(torch.sum(x, dim=-1) - 1.0))  # sum penalized for distance from 1.0
    # normalize by the product of all dimensions except the final one?
    return a * feature_penalty + b * sum_penalty


def unrolldict(d):
    unrolled_items = list(itertools.chain(*(d.items())))
    return unrolled_items


def main():
    # device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    

    # Experiment configuration & hyperparameter space
    # Each is also a command-line argument which can accept multiple comma-separate values for a gridsearch which will be evaluated in sequence.

    hpbuilder = HyperparameterBuilder()

    hpbuilder.add_param("model_name", 
                        # "junk", 
                        # 'baseline-constShaped',
                        # 'baseline-SLPMultShaped',
                        'cNODE1',
                        # 'cNODE2',
                        # 'transformShaped',
                        # 'transformShaped-AbundEncoding',
                        # 'transformRZShaped',
                        # 'canODE-FitMat',
                        # 'canODE-attendFit',
                        # "canODE-FitMat-AbundEncoding", 
                        # 'cNODE-hourglass',
                        # 'baseline-cNODE0',
                        help="model(s) to run")

    # data params
    datacat = "data"
    hpbuilder.add_param("dataset", 
                        # "waimea", 
                        # "waimea-std", 
                        # "cNODE-paper-ocean", 
                        # "cNODE-paper-ocean-std", 
                        # "cNODE-paper-human-oral", 
                        # "cNODE-paper-human-oral-std", 
                        "69@4_48_richness50",
                        # "5000@7_48_richness170",
                        category=datacat, help="dataset to use")
    hpbuilder.add_param("data_subset", 500, 
                        category=datacat, help="number of data samples to use, -1 for all")
    hpbuilder.add_param("kfolds", 5, 
                        category=datacat, help="how many data folds, -1 for leave-one-out. If data_validation_samples is <= 0, K-Fold cross-validation will be used. The total samples will be determined by data_subset and divided into folds for training and validation.")
    hpbuilder.add_param("whichfold", -1, 
                        category=datacat, help="which fold to run, -1 for all")
    hpbuilder.add_param("data_validation_samples", 200,
                        category=datacat, help="Number of samples to use for validation. If <= 0, uses K-Fold crossvalidation (see other arguments). If positive, K-Fold will not be used, and instead the first data_validation_samples samples will be used for validation and the following data_subset samples will be used for training.")
    
    # slurm params
    hpbuilder.add_param("batchid", 0,
                        help="slurm array job id")
    hpbuilder.add_param("taskid", 0,
                        help="slurm array task id")
    hpbuilder.add_param("jobid", "-1", 
                        help="slurm job id")
    hpbuilder.add_flag("headless", False, 
                       help="run without plotting")
    
    # experiment params
    hpbuilder.add_param("epochs", 1000, 
                        help="maximum number of epochs")
    hpbuilder.add_flag("subset_increases_epochs", False,
                        help="if true, epochs will be adjusted based on the subset size to run the same number of total samples")
    hpbuilder.add_param("min_epochs", 1, 
                        help="minimum number of epochs")
    hpbuilder.add_param("minibatch_examples", 100, 
                        help="minibatch size")
    hpbuilder.add_param("accumulated_minibatches", 1, 
                        help="number of minibatches to accumulate before stepping")
    hpbuilder.add_param("run_test", False,
                        category=datacat, help="run the test set after training")
    
    hpbuilder.add_param("epoch_manager", "AdaptiveValPlateau",
                        help="which type of epoch manager to use")

    hpbuilder.add_param("early_stop", True, 
                        help="whether or not to use early stopping")
    hpbuilder.add_param("patience", 5, 
                        help="patience for early stopping")
    
    # Optimizer params
    hpbuilder.add_param("lr", 10.0, 1.0, 0.1, 0.01,  
                        help="learning rate")
    hpbuilder.add_param("reptile_lr", 1.0, 
                        help="reptile outer-loop learning rate")
    hpbuilder.add_param("wd_factor", 0.0, 
                        help="weight decay factor (multiple of LR)")
    hpbuilder.add_param("noise", 0.075, 
                        help="noise level")
    
    # Data augmentation params
    hpbuilder.add_param("ode_timesteps", 15, 
                        help="number of ODE timesteps")
    hpbuilder.add_param("interpolate", False, 
                        help="whether or not to use supervised interpolation steps")
    hpbuilder.add_param("interpolate_noise", False,
                        help="whether or not to use independent noise for interpolation")
    
    # Model architecture params
    hpbuilder.add_param("cnode_bias", False, 
                        help="whether or not to use a bias term when predicting fitness in cNODE and similar models")
    hpbuilder.add_param("num_heads", 2, 
                        help="number of attention heads in transformer-based models")
    hpbuilder.add_param("hidden_dim", 8, 
                        help="hidden dimension")
    hpbuilder.add_param("attend_dim_per_head", 4, 
                        help="dimension of attention embedding, per attention head")
    hpbuilder.add_param("depth", 6, 
                        help="depth of model")
    hpbuilder.add_param("ffn_dim_multiplier", 4.0, 
                        help="multiplier for feedforward network dimension in transformer-based models")
    hpbuilder.add_param("dropout", 0.1, 
                        help="dropout rate")
    

    # Specify model constructors for experiment
    # Note that each must be a constructor function that takes a dicy/dictionary args. Lamda is recommended.
    models = {
        # most useful models
        'baseline-constShaped': lambda args: models_baseline.ConstOutputFilteredNormalized(args.data_dim),
        'baseline-SLPMultShaped': lambda args: models_baseline.SLPMultFilteredNormalized(args.data_dim, args.hidden_dim),
        'cNODE1': lambda args: models_cnode.cNODE1(args.data_dim, args.cnode_bias),
        'cNODE2': lambda args: models_cnode.cNODE2(args.data_dim, args.cnode_bias),
        'transformShaped': lambda args: models_embedded.TransformerNormalized(
            data_dim=args.data_dim, id_embed_dim=args.attend_dim, num_heads=args.num_heads, depth=args.depth,
            ffn_dim_multiplier=args.ffn_dim_multiplier, dropout=args.dropout
        ),
        'transformRZShaped': lambda args: models_embedded.RZTransformerNormalized(
            data_dim=args.data_dim, id_embed_dim=args.attend_dim, num_heads=args.num_heads, depth=args.depth,
            ffn_dim_multiplier=args.ffn_dim_multiplier, dropout=args.dropout
        ),
        'canODE-FitMat': lambda args: models_embedded.canODE_GenerateFitMat(
            data_dim=args.data_dim, id_embed_dim=args.attend_dim, num_heads=args.num_heads, depth=args.depth,
            ffn_dim_multiplier=args.ffn_dim_multiplier, fitness_qk_dim=args.attend_dim, dropout=args.dropout, bias=args.cnode_bias
        ),
        'canODE-attendFit': lambda args: models_embedded.canODE_ReplicatorAttendFit(
            data_dim=args.data_dim, id_embed_dim=args.attend_dim, num_heads=args.num_heads, depth=args.depth,
            ffn_dim_multiplier=args.ffn_dim_multiplier, fitness_qk_dim=args.attend_dim, dropout=args.dropout
        ),
        'cNODE-hourglass': lambda args: models_cnode.cNODE_HourglassFitness(
            data_dim=args.data_dim, hidden_dim=args.hidden_dim, depth=args.depth
        ),
        'baseline-cNODE0': lambda args: models_baseline.cNODE0(args.data_dim),
        
        
        # additional baseline models
        'baseline-1const': lambda args: models_baseline.SingleConst(),
        'baseline-1constShaped': lambda args: models_baseline.SingleConstFilteredNormalized(),
        'baseline-const': lambda args: models_baseline.ConstOutput(args.data_dim),
        'baseline-SLPShaped': lambda args: models_baseline.SLPFilteredNormalized(args.data_dim, args.hidden_dim),
        'baseline-SLPSumShaped': lambda args: models_baseline.SLPSumFilteredNormalized(args.data_dim, args.hidden_dim),
        'baseline-SLPMultSumShaped': lambda args: models_baseline.SLPMultSumFilteredNormalized(args.data_dim, args.hidden_dim),
        'baseline-cNODE0-1step': lambda args: models_baseline.cNODE0_singlestep(args.data_dim),
        'baseline-cNODE1-1step': lambda args: models_baseline.cNODE1_singlestep(args.data_dim, args.cnode_bias),
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
        'transformSoftmax': lambda args: models_embedded.TransformerSoftmax(
            data_dim=args.data_dim, id_embed_dim=args.attend_dim, num_heads=args.num_heads, depth=args.depth,
            ffn_dim_multiplier=args.ffn_dim_multiplier, dropout=args.dropout
        ),
        'canODE-transformer': lambda args: models_embedded.canODE_transformer(args.data_dim, args.attend_dim, args.num_heads, args.depth, args.ffn_dim_multiplier),
        'canODE-noValue': lambda args: models_embedded.canODE_attentionNoValue(args.data_dim, args.attend_dim, args.attend_dim),
        'canODE-noValue-static': lambda args: models_embedded.canODE_attentionNoValue_static(args.data_dim, args.attend_dim, args.attend_dim),
        'canODE-attention': lambda args: models_embedded.canODE_attention(args.data_dim, args.attend_dim, args.attend_dim),
        'canODE-multihead': lambda args: models_embedded.canODE_attentionMultihead(args.data_dim, args.attend_dim, args.num_heads),
        
        
        # sanity test models
        'cNODE1-GenFn': lambda args: models_cnode.cNODE2_ExternalFitnessFn(args.data_dim, args.cnode_bias), # for testing, identical to cNODE1
        'cNODE2-DKI': lambda args: models_cnode.cNODE2_DKI(args.data_dim, args.cnode_bias), # sanity test, this is the same as cNODE2 but less optimized
        'cNODE2-Gen': lambda args: models_cnode.cNODEGen_ConstructedFitness(lambda: nn.Sequential(nn.Linear(args.data_dim, args.data_dim, args.cnode_bias), nn.Linear(args.data_dim, args.data_dim, args.cnode_bias))),  # sanity test, this is the same as cNODE2 but generated at runtime
        "cNODE2-static": lambda args: models_cnode.cNODE2_ExternalFitness(args.data_dim, args.cnode_bias), # sanity test
        "cNODE2-FnFitness": lambda args: models_cnode.cNODE2_FnFitness(args.data_dim, args.cnode_bias), # sanity test, this is the same as cNODE2 but testing externally-supplied fitness functions
    }

    # Epoch managers
    # TODO: make all of these args accessible to command line
    epoch_mngr_constructors = {
        "Fixed": lambda args: epoch_managers.FixedManager(max_epochs=args.epochs),
        "AdaptiveValPlateau": lambda args: epoch_managers.AdaptiveValPlateauManager(memory=0.85, rate_threshold_factor=0.05, min_epochs=args.min_epochs, max_epochs=args.epochs, patience=args.patience),
    }


    for dp in hpbuilder.parse_and_generate_combinations(category=datacat):

        # load data
        filepath_train = f'data/{dp.dataset}_train.csv'
        filepath_train_pos = f'data/{dp.dataset}_train-pos.csv'
        filepath_train_val = f'data/{dp.dataset}_train-val.csv'
        filepath_test = f'data/{dp.dataset}_test.csv'
        filepath_test_pos = f'data/{dp.dataset}_test-pos.csv'
        filepath_test_val = f'data/{dp.dataset}_test-val.csv'
        if dp.data_validation_samples > 0 and dp.data_subset > 0:
            samples_to_load = dp.data_subset + dp.data_validation_samples
        else:
            samples_to_load = dp.data_subset
        x, y, xcon, ycon, idcon, dp.data_fraction = data.load_data(filepath_train, filepath_train_pos, filepath_train_val, device, subset=samples_to_load)
        if dp.run_test:
            xtest, ytest, xcontest, ycontest, idcontest, _ = data.load_data(filepath_test, filepath_test_pos, filepath_test_val, device, subset=-1)
            testdata = [xtest, ytest, xcontest, ycontest, idcontest]
        else:
            testdata = None
        if dp.data_validation_samples > 0:
            data_folded = data.split_data([x, y, xcon, ycon, idcon], dp.data_validation_samples)
        else:
            data_folded = data.fold_data([x, y, xcon, ycon, idcon], dp.kfolds)  
            data_folded = [data_folded[dp.whichfold]] if dp.whichfold >= 0 else data_folded  # only run a single fold based on args
        # whether in K-Folds or split validation set, data_folded dimensions are (kfolds, datasets [x, y, xcon, ycon, or idcon], train vs valid, samples, features)
        
        # assert (data.check_leakage(data_folded))

        print('dataset:', filepath_train)
        print(f'using {dp.data_subset} samples, which is {dp.data_fraction * 100}% of the data')
        print(f'data shape: {x.shape}\n')
        print(f'training data shape: {data_folded[0][0][0].shape}')
        print(f'condensed training shape: {data_folded[0][2][0].shape}\n')
        print(f'validation data shape: {data_folded[0][0][1].shape}')
        print(f'condensed validation shape: {data_folded[0][2][1].shape}\n')
        if dp.run_test:
            print(f'test data shape: {xtest.shape}')
            print(f'test condensed data shape: {xcontest.shape}\n')

        
        # specify loss function
        loss_fn = loss_bc
        # avg_richness = x.count_nonzero()/x.size(0)
        # loss_fn = lambda y_pred, y_true: loss_bc_unbounded(y_pred, y_true, avg_richness)
        score_fn = loss_bc
        # loss_fn = lambda y_pred,y_true: loss_bc(y_pred, y_true) + distribution_error(y_pred)
        
        distr_error_fn = distribution_error
        

        # Establish baseline performance and add to plots
        identity_model = models_baseline.ReturnInput()
        identity_loss, identity_score, identity_distro_error = expt.validate_epoch(identity_model, [x, y], 100,  # TODO: using 100 instead of hp.minibatch_examples, because this model doesn't learn so the only concern is computational throughput.
                                                                        [0.0, 1.0], loss_fn, score_fn, distr_error_fn,
                                                                        device)
        print(f"\n\nIDENTITY MODEL loss: {identity_loss}, score: {identity_score}\n\n")
        plotstream.plot_horizontal_line(f"loss {dp.dataset}", identity_loss, f"Identity")
        # plotstream.plot_horizontal_line(f"score {dp.dataset}", identity_score, f"Identity")


        for hp in hpbuilder.parse_and_generate_combinations():
        
            # things that are needed for reporting an exception, so they go before the try block
            jobid_substring = int(hp.jobid.split('_')[0])
            jobstring = f"_job{jobid_substring}" if jobid_substring >= 0 else ""
            filepath_out_expt = f'results/expt/{dp.dataset}{jobstring}_experiments.csv'
            num_params = -1
            optdict = {"epoch": -1, "trn_loss": -1.0, "trn_score": -1.0, "val_loss": -1.0,
                    "val_score": -1.0, "lr": -1.0, "time": -1.0, "gpu_memory": -1.0,
                    "metric": -1.0, "stop_metric": -1.0, "stop_threshold": -1.0}
            val_loss_optims = [Optimum('val_loss', 'min', dict=optdict)]
            trn_loss_optims = [Optimum('trn_loss', 'min', dict=optdict)]
            val_score_optims = [Optimum('val_score', 'min', dict=optdict)]
            trn_score_optims = [Optimum('trn_score', 'min', dict=optdict)]

            # try:
            if True:
                # computed hyperparams
                _, hp.data_dim = x.shape
                hp.WD = hp.lr * hp.wd_factor
                hp.attend_dim = hp.attend_dim_per_head * hp.num_heads
                hp.model_config = f"{hp.model_name}_{hp.configid}"

                # conditionally adjust epochs to compensate for subset size
                if hp.subset_increases_epochs:
                    hp.epochs = int(hp.epochs // dp.data_fraction)
                    print(f"Adjusted epochs to {hp.epochs} to compensate for subset size")
                
                if hp.headless:
                    plotstream.set_headless()

                assert hp.attend_dim % hp.num_heads == 0, "attend_dim must be divisible by num_heads"
                
                
                reeval_train = True # False # hp.interpolate or hp.noise > 0.0 # This isn't a general rule, you might want to do this for other reasons. But it's an easy way to make sure training loss curves are readable.
                
                
                model_constr = models[hp.model_name]
                epoch_manager_constr = epoch_mngr_constructors[hp.epoch_manager]
                
                
                # scaler = torch.amp.GradScaler(device)
                scaler = torch.cuda.amp.GradScaler()
                
                # time step "data"
                ode_timemax = 1.0
                ode_stepsize = ode_timemax / hp.ode_timesteps
                timesteps = torch.arange(0.0, ode_timemax + 0.1*ode_stepsize, ode_stepsize).to(device)
                
                # TODO: Experiment dictionary. Model, data set, hyperparam override(s).
                # Model dictionary. Hyperparam override(s)
                
                seed = int(time.time())  # currently only used to set the data shuffle seed in find_LR
                print(f"Seed: {seed}")
                
                # test construction and print parameter count
                print(f"\nModel construction test for: {hp.model_config}")
                model = model_constr(hp)
                num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                print(f"Number of parameters in model: {num_params}")
                
                # find optimal LR
                # hp.WD, hp.lr = hyperparameter_search_with_LRfinder(
                #     model_constr, hp, model_name, scaler, data_folded, hp.minibatch_examples, hp.accumulated_minibatches,
                #     device, 3, 3, dataname, timesteps, loss_fn, score_fn, distr_error_fn, verbosity=1, seed=seed)
                # print(f"LR:{hp.lr}, WD:{hp.WD}")
                
                # hp = ui.ask(hp, keys=["LR", "WD"])
                
                # print hyperparams
                for key, value in hp.items():
                    print(f"{key}: {value}")
                
                # Just for the sake of logging experiments before cross validation...
                stream.stream_scores(filepath_out_expt, True, True, True,
                                    "mean_val_loss", -1,
                                    "mean_val_loss @ epoch", -1,
                                    "mean_val_loss @ time", -1,
                                    "mean_val_loss @ trn_loss", -1,
                                    "identity loss", identity_loss,
                                    "model parameters", num_params,
                                    "fold", -1,
                                    "device", device,
                                    "solver", os.environ["SOLVER"],
                                    *unrolldict(dp),  # unroll the data params dictionary
                                    *unrolldict(hp),  # unroll the hyperparams dictionary
                                    *unrolloptims(val_loss_optims[0], val_score_optims[0], trn_loss_optims[0],
                                                trn_score_optims[0]),
                                    prefix="\n=======================================================EXPERIMENT========================================================\n",
                                    suffix="\n=========================================================================================================================\n")
                
                
                # train and test the model across multiple folds
                val_loss_optims, val_score_optims, trn_loss_optims, trn_score_optims, final_optims, training_curves = expt.crossvalidate_model(
                    hp.lr, scaler, hp.accumulated_minibatches, data_folded, testdata, hp.noise, hp.interpolate, device, hp.early_stop, hp.patience,
                    dp.kfolds, hp.min_epochs, hp.epochs, hp.minibatch_examples, model_constr, epoch_manager_constr, hp,
                    hp.model_name, hp.model_config, dp.dataset, timesteps, loss_fn, score_fn, distr_error_fn, hp.WD, verbosity=1,
                    reptile_rewind=(1.0 - hp.reptile_lr), reeval_train=reeval_train, whichfold=dp.whichfold, jobstring=jobstring
                )
                
                # print all folds
                print(f'Val Loss optimums: \n{[f'Fold {num}: {opt}\n' for num, opt in enumerate(val_loss_optims)]}\n')
                print(f'Val Score optimums: \n{[f'Fold {num}: {opt}\n' for num, opt in enumerate(val_score_optims)]}\n')
                print(f'Trn Loss optimums: \n{[f'Fold {num}: {opt}\n' for num, opt in enumerate(trn_loss_optims)]}\n')
                print(f'Trn Score optimums: \n{[f'Fold {num}: {opt}\n' for num, opt in enumerate(trn_score_optims)]}\n')
                
                # calculate fold summaries
                avg_val_loss_optim = summarize(val_loss_optims)
                avg_val_score_optim = summarize(val_score_optims)
                avg_trn_loss_optim = summarize(trn_loss_optims)
                avg_trn_score_optim = summarize(trn_score_optims)
                
                # print summariesrun.py
                print(f'Avg Val Loss optimum: {avg_val_loss_optim}')
                print(f'Avg Val Score optimum: {avg_val_score_optim}')
                print(f'Avg Trn Loss optimum: {avg_trn_loss_optim}')
                print(f'Avg Trn Score optimum: {avg_trn_score_optim}')
                
                # find optimal epoch
                # training_curves is a list of dictionaries, convert to a dataframe
                all_data = [entry for fold in training_curves for entry in fold]
                df = pd.DataFrame(all_data)
                df_clean = df.dropna(subset=['val_loss'])
                # Check if df_clean is not empty
                if not df_clean.empty:
                    average_metrics = df_clean.groupby('epoch').mean(numeric_only=True).reset_index()
                    min_val_loss_epoch = average_metrics.loc[average_metrics['val_loss'].idxmin()]
                    best_epoch_metrics = min_val_loss_epoch.to_dict()
                else:
                    min_val_loss_epoch = None  # or handle the empty case as needed
                    best_epoch_metrics = {"epoch": -1, "val_loss": -1.0, "trn_loss": -1.0, "val_score": -1.0, "trn_score": -1.0, "time": -1.0}
                
                # write folds to log file
                for i in range(len(val_loss_optims)):
                    stream.stream_scores(filepath_out_expt, True, True, True,
                                        "mean_val_loss", best_epoch_metrics["val_loss"],
                                        "mean_val_loss @ epoch", best_epoch_metrics["epoch"],
                                        "mean_val_loss @ time", best_epoch_metrics["time"],
                                        "mean_val_loss @ trn_loss", best_epoch_metrics["trn_loss"],
                                        "identity loss", identity_loss,
                                        "model parameters", num_params,
                                        "fold", i if dp.whichfold < 0 else dp.whichfold,
                                        "device", device,
                                        "solver", os.environ["SOLVER"],
                                        *unrolldict(dp),  # unroll the data params dictionary
                                        *unrolldict(hp),  # unroll the hyperparams dictionary
                                        *unrolloptims(val_loss_optims[i], val_score_optims[i], trn_loss_optims[i],
                                                    trn_score_optims[i]),
                                        prefix="\n=======================================================EXPERIMENT========================================================\n",
                                        suffix="\n=========================================================================================================================\n")
                
            # except Exception as e:
            #     stream.stream_scores(filepath_out_expt, True, True, True,
            #                     "mean_val_loss", -1,
            #                     "mean_val_loss @ epoch", -1,
            #                     "mean_val_loss @ time", -1,
            #                     "mean_val_loss @ trn_loss", -1,
            #                     "identity loss", identity_loss,
            #                     "model parameters", num_params,
            #                     "fold", -1,
            #                     "device", device,
            #                     "solver", os.environ["SOLVER"],
            #                     *unrolldict(dp),  # unroll the data params dictionary
            #                     *unrolldict(hp),  # unroll the hyperparams dictionary
            #                     *unrolloptims(val_loss_optims[0], val_score_optims[0], trn_loss_optims[0],
            #                                 trn_score_optims[0]),
            #                     prefix="\n=======================================================EXPERIMENT========================================================\n",
            #                     suffix="\n=========================================================================================================================\n")
            #     print(f"Model {hp.model_name} failed with error:\n{e}")
    
    print("\n\nDONE")
    plotstream.wait_for_plot_exit()


# main
if __name__ == "__main__":
    main()
