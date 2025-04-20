import os
import traceback

import chunked_dataset
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
import stream_plot as plotstream
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


class DummyGradScaler:
    def scale(self, loss):
        return loss  # no scaling, just return as-is

    def step(self, optimizer):
        optimizer.step()

    def update(self):
        pass



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
                        # 'cNODE1',
                        # 'cNODE2',
                        'transformShaped',
                        # 'transformShaped-AbundEncoding',
                        # 'transformRZShaped',
                        # 'canODE-FitMat',
                        # 'canODE-attendFit',
                        # "canODE-FitMat-AbundEncoding", 
                        'cNODE-hourglass',
                        # 'baseline-cNODE0',
                        help="model(s) to run")

    # data params
    datacat = "data"
    # hpbuilder.add_param("dataset", 
    #                     # "waimea", 
    #                     # "waimea-std", 
    #                     # "cNODE-paper-ocean", 
    #                     # "cNODE-paper-ocean-std", 
    #                     # "cNODE-paper-human-oral", 
    #                     # "cNODE-paper-human-oral-std", 
    #                     "69@4_48_richness50",
    #                     # "5000@7_48_richness170",
    #                     category=datacat, help="dataset to use")
    hpbuilder.add_param("x_dataset", 
                        "256",
                        category=datacat, help="dataset to use for inputs")
    hpbuilder.add_param("y_dataset", 
                        "256-random",
                        category=datacat, help="dataset to use for supervising outputs")
    hpbuilder.add_param("data_subset", 
                        1000,
                        # 10000, 
                        # 30, 
                        # 100000, 
                        # 1, 10, 100, 1000, 10000, #100000, 
                        category=datacat, help="number of data samples to use, -1 for all")
    hpbuilder.add_param("kfolds", 5, 
                        category=datacat, help="how many data folds, -1 for leave-one-out. If data_validation_samples is <= 0, K-Fold cross-validation will be used. The total samples will be determined by data_subset and divided into folds for training and validation.")
    hpbuilder.add_param("whichfold", -1, 
                        category=datacat, help="which fold to run, -1 for all")
    hpbuilder.add_param("data_validation_samples", 0,
                        category=datacat, help="Number of samples to use for validation. If <= 0, uses K-Fold crossvalidation (see other arguments). If positive, K-Fold will not be used, and instead the first data_validation_samples samples will be used for validation and the following data_subset samples will be used for training.")
    hpbuilder.add_param("minibatch_examples", 100, 
                        help="minibatch size",
                        category=datacat)
    
    # slurm params
    config_cat = "config"
    hpbuilder.add_param("batchid", 0,
                        help="slurm array job id", 
                        category=config_cat)
    hpbuilder.add_param("taskid", 0,
                        help="slurm array task id", 
                        category=config_cat)
    hpbuilder.add_param("jobid", "-1", 
                        help="slurm job id", 
                        category=config_cat)
    hpbuilder.add_flag("headless", False, 
                        help="run without plotting", 
                        category=config_cat)
    
    # experiment params
    hpbuilder.add_param("epochs", 
                        # 6, 20, 64, 200, 
                        # 64, 
                        # 25, 
                        7, 
                        # 200, 
                        help="maximum number of epochs")
    hpbuilder.add_flag("subset_increases_epochs", True,
                        help="if true, epochs will be adjusted based on the subset size to run the same number of total samples")
    hpbuilder.add_param("min_epochs", 1, 
                        help="minimum number of epochs")
    hpbuilder.add_param("accumulated_minibatches", 1, 
                        help="number of minibatches to accumulate before stepping")
    hpbuilder.add_param("run_test", False,
                        category=datacat, help="run the test set after training")
    hpbuilder.add_flag("use_best_model", 
                        # True,
                        False,
                        help="whether or not to use the best model for testing")
    hpbuilder.add_flag("preeval_training_set", False,
                        help="whether or not to pre-evaluate the training set before starting training")
    hpbuilder.add_flag("reeval_training_set_epoch", False,
                        help="whether or not to re-evaluate the training set after each epoch")
    hpbuilder.add_flag("reeval_training_set_final", True,
                        help="whether or not to re-evaluate the training set after the final epoch")
    
    hpbuilder.add_param("epoch_manager", 
                        "Fixed", 
                        # "AdaptiveValPlateau",
                        help="which type of epoch manager to use")
    hpbuilder.add_param("mini_epoch_size", 
                        # 500,
                        100, 
                        # 0, 
                        help="number of training samples before running validation and/or tests. If <= 0, uses a full epoch before validation (equivalent to setting mini_epoch_size to the total number of training samples). Default -1.")

    hpbuilder.add_param("early_stop", False, 
                        help="whether or not to use early stopping")
    hpbuilder.add_param("patience", 5, 
                        help="patience for early stopping")
    
    
    # Optimizer params
    hpbuilder.add_param("lr", 
                        # 0.1, 
                        0.00160707665, 
                        # 0.001,
                        # 0.01, 
                        # 1.0, 0.32, 0.1, 0.032, 
                        # 0.01, 0.0032, 0.001, 0.00032, 0.0001, 
                        # 0.32, 0.1, 0.032, 0.01, 0.0032,
                        # 0.32, 0.1, 0.032, 0.01, 0.0032, #0.001, 0.00032,   
                        help="learning rate")
    hpbuilder.add_param("reptile_lr", 1.0, 
                        help="reptile outer-loop learning rate")
    hpbuilder.add_param("wd", 0.0, 
                        help="weight decay")
    hpbuilder.add_param("noise", 0.0,   
                        help="noise level")
    
    # Data augmentation params
    # hpbuilder.add_param("ode_timesteps", 15, 
    #                     help="number of ODE timesteps")
    hpbuilder.add_param("ode_timesteps_file", 
                        "t.csv",
                        # "t_linear.csv",
                        # "t_shortlinear.csv",
                        help="ODE integration timesteps file")
    hpbuilder.add_param("interpolate", False, 
                        help="whether or not to use supervised interpolation steps")
    hpbuilder.add_param("interpolate_noise", False,
                        help="whether or not to use independent noise for interpolation")
    
    # Model architecture params
    hpbuilder.add_param("identity_gate", 
                        True,
                        # False, 
                        help="whether or not to use 'ReZero'-style learnable gate scalars, initialized such that each model starts as an identity function")
    hpbuilder.add_param("cnode1_init_zero", 
                        True, 
                        # False, 
                        help="whether or not to use 'ReZero'-style gates to ensure all models start as an identity function")
    hpbuilder.add_param("cnode_bias", 
                        True, 
                        # False,
                        help="whether or not to use a bias term when predicting fitness in cNODE and similar models")
    hpbuilder.add_param("num_heads", 15,  
                        help="number of attention heads in transformer-based models")
    hpbuilder.add_param("hidden_dim", 8, 
                        help="hidden dimension")
    hpbuilder.add_param("attend_dim_per_head", 9,
                        help="dimension of attention embedding, per attention head")
    hpbuilder.add_param("depth", 4, 
                        help="depth of model")
    hpbuilder.add_param("ffn_dim_multiplier", 
                        # 4.0,
                        3.924555754,  
                        help="multiplier for feedforward network dimension in transformer-based models")
    hpbuilder.add_param("dropout", 0.04137076975, 
                        help="dropout rate")

    

    # Specify model constructors for experiment
    # Note that each must be a constructor function that takes a dicy/dictionary args. Lamda is recommended.
    models = {
        # most useful models
        'baseline-constShaped': lambda args: models_baseline.ConstOutputFilteredNormalized(args.data_dim, identity_gate=args.identity_gate),
        'baseline-SLPMultShaped': lambda args: models_baseline.SLPMultFilteredNormalized(args.data_dim, args.hidden_dim, identity_gate=args.identity_gate),
        'cNODE1': lambda args: models_cnode.cNODE1(args.data_dim, bias=args.cnode_bias, init_zero=args.cnode1_init_zero, identity_gate=args.identity_gate),
        'cNODE2': lambda args: models_cnode.cNODE2(args.data_dim, bias=True, identity_gate=args.identity_gate),
        'transformShaped': lambda args: models_embedded.TransformerNormalized(
            data_dim=args.data_dim, id_embed_dim=args.attend_dim, num_heads=args.num_heads, depth=args.depth,
            ffn_dim_multiplier=args.ffn_dim_multiplier, dropout=args.dropout, 
            identity_gate=args.identity_gate
        ),
        # 'transformRZShaped': lambda args: models_embedded.RZTransformerNormalized(
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
        'baseline-1constShaped': lambda args: models_baseline.SingleConstFilteredNormalized(),
        'baseline-const': lambda args: models_baseline.ConstOutput(args.data_dim),
        'baseline-SLPShaped': lambda args: models_baseline.SLPFilteredNormalized(args.data_dim, args.hidden_dim),
        'baseline-SLPSumShaped': lambda args: models_baseline.SLPSumFilteredNormalized(args.data_dim, args.hidden_dim),
        'baseline-SLPMultSumShaped': lambda args: models_baseline.SLPMultSumFilteredNormalized(args.data_dim, args.hidden_dim),
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
        'cNODE1-GenFn': lambda args: models_cnode.cNODE2_ExternalFitnessFn(args.data_dim, args.cnode_bias, init_zero=args.cnode1_init_zero, identity_gate=args.identity_gate), # for testing, identical to cNODE1
        'cNODE2-DKI': lambda args: models_cnode.cNODE2_DKI(args.data_dim, args.cnode_bias), # sanity test, this is the same as cNODE2 but less optimized
        'cNODE2-Gen': lambda args: models_cnode.cNODEGen_ConstructedFitness(lambda: nn.Sequential(nn.Linear(args.data_dim, args.data_dim, args.cnode_bias), nn.Linear(args.data_dim, args.data_dim, args.cnode_bias)), init_zero=args.cnode1_init_zero, identity_gate=args.identity_gate),  # sanity test, this is the same as cNODE2 but generated at runtime
        "cNODE2-static": lambda args: models_cnode.cNODE2_ExternalFitness(args.data_dim, args.cnode_bias, init_zero=args.cnode1_init_zero, identity_gate=args.identity_gate), # sanity test
        "cNODE2-FnFitness": lambda args: models_cnode.cNODE2_FnFitness(args.data_dim, args.cnode_bias, init_zero=args.cnode1_init_zero, identity_gate=args.identity_gate), # sanity test, this is the same as cNODE2 but testing externally-supplied fitness functions
    }

    # Epoch managers
    # TODO: make all of these args accessible to command line
    epoch_mngr_constructors = {
        "Fixed": lambda args: epoch_managers.FixedManager(max_epochs=args.adjusted_epochs),
        "AdaptiveValPlateau": lambda args: epoch_managers.AdaptiveValPlateauManager(memory=0.75, rate_threshold_factor=0.05, min_epochs=args.min_epochs, max_epochs=args.adjusted_epochs, patience=args.patience),
    }

                

    cp = hpbuilder.parse_and_generate_combinations(category=config_cat)[0] # There should only be one configuration
    
    for key, value in cp.items():
        print(f"{key}: {value}")

    if cp.headless:
        plotstream.set_headless()


    # loop through possible combinations of dataset hyperparams
    for dp in hpbuilder.parse_and_generate_combinations(category=datacat):

        # datasets
        base_filepath = f"data/{dp.x_dataset}/"

        filenames = {
            chunked_dataset.DK_BINARY: f"{dp.x_dataset}_binary",
            chunked_dataset.DK_IDS: f"{dp.x_dataset}_ids-sparse",
            chunked_dataset.DK_X: f"{dp.x_dataset}_x0", 
            chunked_dataset.DK_XSPARSE: f"{dp.x_dataset}_x0-sparse",
            chunked_dataset.DK_Y: f"{dp.y_dataset}_y",
            chunked_dataset.DK_YSPARSE: f"{dp.y_dataset}_y-sparse",
        }

        data_folded, dp.total_train_samples, dp.data_fraction, dense_columns, sparse_columns = chunked_dataset.load_folded_datasets(
            base_filepath, 
            filenames,
            dp.minibatch_examples,
            dp.data_subset,
            dp.data_validation_samples,
            dp.kfolds
        )
        print(f"length of data_folded: {len(data_folded)}")
        # dimensions are (kfolds, train vs valid, datasets tuple, batches, samples)
        # previously, dimensions were (kfolds, datasets [x, y, xcon, ycon, or idcon], train vs valid, samples, features)

        if dp.run_test:
            testdata = chunked_dataset.TestCSVDataset(base_filepath, filenames, dp.minibatch_examples)
            print(f"Test samples: {testdata.total_samples}")
        else:
            testdata = None
        
        print('-' * 50 + '\n')
        
        for key, value in dp.items():
            print(f"{key}: {value}")

        print('-' * 50 + '\n')

        # # assert (data.check_leakage(data_folded))
        # # assert(data.check_simplex(y))
        # # assert(data.check_finite(y))

        
        # specify loss function
        loss_fn = loss_bc
        # avg_richness = x.count_nonzero()/x.size(0)
        # loss_fn = lambda y_pred, y_true: loss_bc_unbounded(y_pred, y_true, avg_richness)
        score_fn = loss_bc
        # loss_fn = lambda y_pred,y_true: loss_bc(y_pred, y_true) + distribution_error(y_pred)
        
        distr_error_fn = distribution_error


        # Establish baseline performance and add to plots
        identity_model = models_baseline.ReturnInput()
        identity_loss, identity_score, identity_distro_error = expt.validate_epoch(identity_model, False, data_folded[0][1], 100,  # using 100 instead of hp.minibatch_examples, because this model doesn't learn so the only concern is computational throughput.
                                                                        [0.0, 1.0], loss_fn, score_fn, distr_error_fn,
                                                                        device)
        print(f"\n\nIDENTITY MODEL loss: {identity_loss}, score: {identity_score}\n\n")
        plotstream.plot_horizontal_line(f"loss {dp.y_dataset}", identity_loss, f"Identity")
        # plotstream.plot_horizontal_line(f"score {dp.y_dataset}", identity_score, f"Identity")

        # loop through possible combinations of generic hyperparams
        for hp in hpbuilder.parse_and_generate_combinations():
        
            # things that are needed for reporting an exception, so they go before the try block
            jobid_substring = int(cp.jobid.split('_')[0])
            jobstring = f"_job{jobid_substring}" if jobid_substring >= 0 else ""
            filepath_out_expt = f'results/expt/{dp.y_dataset}{jobstring}_experiments.csv'
            num_params = -1
            optdict = {"epoch": -1, "mini_epoch": -1, "trn_loss": -1.0, "trn_score": -1.0, "val_loss": -1.0,
                    "val_score": -1.0, "lr": -1.0, "time": -1.0, "gpu_memory": -1.0,
                    "metric": -1.0, "stop_metric": -1.0, "stop_threshold": -1.0}
            val_loss_optims = [Optimum('val_loss', 'min', dict=optdict)]
            trn_loss_optims = [Optimum('trn_loss', 'min', dict=optdict)]
            val_score_optims = [Optimum('val_score', 'min', dict=optdict)]
            trn_score_optims = [Optimum('trn_score', 'min', dict=optdict)]
            final_optims = [Optimum(metric=None, dict=optdict)]

            # if True:
            try:
                # computed hyperparams
                hp.data_dim = dense_columns
                hp.sparse_data_dim = sparse_columns
                # hp.WD = hp.lr * hp.wd_factor
                hp.attend_dim = hp.attend_dim_per_head * hp.num_heads
                hp.model_config = f"{hp.model_name}_{dp.data_configid}x{hp.configid}"

                # conditionally adjust epochs to compensate for subset size
                if hp.subset_increases_epochs:
                    hp.adjusted_epochs = int(hp.epochs // dp.data_fraction)
                    print(f"Adjusted epochs from {hp.epochs} to {hp.adjusted_epochs} to compensate for subset size")
                else:
                    hp.adjusted_epochs = hp.epochs

                assert hp.attend_dim % hp.num_heads == 0, "attend_dim must be divisible by num_heads"
                
                model_constr = models[hp.model_name]
                epoch_manager_constr = epoch_mngr_constructors[hp.epoch_manager]
                
                if device.type == "cuda":
                    scaler = torch.cuda.amp.GradScaler()
                else:
                    scaler = DummyGradScaler()

                # load timesteps from file
                print(f"Loading time steps from {hp.ode_timesteps_file}")
                timesteps = pd.read_csv(hp.ode_timesteps_file, header=None).values.flatten()
                # ode_timemax = 100.0
                # ode_stepsize = ode_timemax / hp.ode_timesteps
                # timesteps = torch.arange(0.0, ode_timemax + 0.1*ode_stepsize, ode_stepsize).to(device)
                
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
                                    "mean_val_loss @ mini-epoch", -1,
                                    "mean_val_loss @ time", -1,
                                    "mean_val_loss @ trn_loss", -1,
                                    "identity loss", identity_loss,
                                    "model parameters", num_params,
                                    "fold", -1,
                                    "device", device,
                                    "solver", os.environ["SOLVER"],
                                    *unrolldict(cp),  # unroll the data params dictionary
                                    *unrolldict(dp),  # unroll the data params dictionary
                                    *unrolldict(hp),  # unroll the hyperparams dictionary
                                    *unrolloptims(val_loss_optims[0], val_score_optims[0], trn_loss_optims[0],
                                                trn_score_optims[0], final_optims[0]),
                                    prefix="\n=======================================================EXPERIMENT========================================================\n",
                                    suffix="\n=========================================================================================================================\n")
                
                
                # train and test the model across multiple folds
                val_loss_optims, val_score_optims, trn_loss_optims, trn_score_optims, final_optims, training_curves = expt.crossvalidate_model(
                    hp.lr, scaler, hp.accumulated_minibatches, data_folded, testdata, dp.total_train_samples, hp.noise, hp.interpolate, hp.interpolate_noise, device, hp.early_stop, hp.patience,
                    dp.kfolds, hp.min_epochs, hp.adjusted_epochs, hp.mini_epoch_size, dp.minibatch_examples, model_constr, epoch_manager_constr, hp,
                    hp.model_name, hp.model_config, dp.y_dataset, timesteps, loss_fn, score_fn, distr_error_fn, hp.wd, verbosity=1,
                    reptile_rewind=(1.0 - hp.reptile_lr), preeval_training_set=hp.preeval_training_set, reeval_train_epoch=hp.reeval_training_set_epoch, reeval_train_final=hp.reeval_training_set_final, 
                    whichfold=dp.whichfold, jobstring=jobstring, use_best_model=hp.use_best_model
                )
                
                # print all folds
                print(f'Val Loss optimums: \n{[f'Fold {num}: {opt}\n' for num, opt in enumerate(val_loss_optims)]}\n')
                print(f'Val Score optimums: \n{[f'Fold {num}: {opt}\n' for num, opt in enumerate(val_score_optims)]}\n')
                print(f'Trn Loss optimums: \n{[f'Fold {num}: {opt}\n' for num, opt in enumerate(trn_loss_optims)]}\n')
                print(f'Trn Score optimums: \n{[f'Fold {num}: {opt}\n' for num, opt in enumerate(trn_score_optims)]}\n')
                print(f'Final optimums: \n{[f'Fold {num}: {opt}\n' for num, opt in enumerate(final_optims)]}\n')
                
                # calculate fold summaries
                avg_val_loss_optim = summarize(val_loss_optims)
                avg_val_score_optim = summarize(val_score_optims)
                avg_trn_loss_optim = summarize(trn_loss_optims)
                avg_trn_score_optim = summarize(trn_score_optims)
                avg_final_optims = summarize(final_optims)
                
                # print summariesrun.py
                print(f'Avg Val Loss optimum: {avg_val_loss_optim}')
                print(f'Avg Val Score optimum: {avg_val_score_optim}')
                print(f'Avg Trn Loss optimum: {avg_trn_loss_optim}')
                print(f'Avg Trn Score optimum: {avg_trn_score_optim}')
                print(f'Avg Final optimum: {avg_final_optims}')
                
                # find optimal mini-epoch
                # training_curves is a list of dictionaries, convert to a dataframe
                all_data = [entry for fold in training_curves for entry in fold]
                df = pd.DataFrame(all_data)
                df_clean = df.dropna(subset=['val_loss'])
                # Check if df_clean is not empty
                if not df_clean.empty:
                    average_metrics = df_clean.groupby('mini_epoch').mean(numeric_only=True).reset_index()
                    min_val_loss_epoch = average_metrics.loc[average_metrics['val_loss'].idxmin()]
                    best_epoch_metrics = min_val_loss_epoch.to_dict()
                else:
                    min_val_loss_epoch = None  # or handle the empty case as needed
                    best_epoch_metrics = {"epoch": -1, "mini_epoch": -1, "val_loss": -1.0, "trn_loss": -1.0, "val_score": -1.0, "trn_score": -1.0, "time": -1.0}
                
                # write folds to log file
                for i in range(len(val_loss_optims)):
                    stream.stream_scores(filepath_out_expt, True, True, True,
                                        "optimal early stop val_loss", best_epoch_metrics["val_loss"],
                                        "optimal early stop epoch", best_epoch_metrics["epoch"],
                                        "optimal early stop mini-epoch", best_epoch_metrics["mini_epoch"],
                                        "optimal early stop time", best_epoch_metrics["time"],
                                        "optimal early stop trn_loss", best_epoch_metrics["trn_loss"],
                                        "identity loss", identity_loss,
                                        "model parameters", num_params,
                                        "fold", i if dp.whichfold < 0 else dp.whichfold,
                                        "device", device,
                                        "solver", os.environ["SOLVER"],
                                        *unrolldict(cp),  # unroll the data params dictionary
                                        *unrolldict(dp),  # unroll the data params dictionary
                                        *unrolldict(hp),  # unroll the hyperparams dictionary
                                        *unrolloptims(val_loss_optims[i], val_score_optims[i], trn_loss_optims[i],
                                                    trn_score_optims[i], final_optims[i]),
                                        prefix="\n=======================================================EXPERIMENT========================================================\n",
                                        suffix="\n=========================================================================================================================\n")
                
            except Exception as e:
                stream.stream_scores(filepath_out_expt, True, True, True,
                                "mean_val_loss", -1,
                                "mean_val_loss @ epoch", -1,
                                "mean_val_loss @ mini-epoch", -1,
                                "mean_val_loss @ time", -1,
                                "mean_val_loss @ trn_loss", -1,
                                "identity loss", identity_loss,
                                "model parameters", num_params,
                                "fold", -1,
                                "device", device,
                                "solver", os.environ["SOLVER"],
                                *unrolldict(cp),  # unroll the data params dictionary
                                *unrolldict(dp),  # unroll the data params dictionary
                                *unrolldict(hp),  # unroll the hyperparams dictionary
                                *unrolloptims(val_loss_optims[0], val_score_optims[0], trn_loss_optims[0],
                                            trn_score_optims[0], final_optims[0]),
                                prefix="\n=======================================================EXPERIMENT========================================================\n",
                                suffix="\n=========================================================================================================================\n")
                print(f"Model {hp.model_name} failed with error:\n{e}")
                traceback.print_exc()
    
    print("\n\nDONE")
    plotstream.wait_for_plot_exit()


# main
if __name__ == "__main__":
    main()
