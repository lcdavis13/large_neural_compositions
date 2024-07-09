import itertools
import math
import time
import numpy as np
import torch
import torch.nn as nn

import data
import stream
import models
import models_condensed
import models_baseline


def loss_bc(y_pred, y_true):  # Bray-Curtis Dissimilarity
    return torch.sum(torch.abs(y_pred - y_true)) / torch.sum(torch.abs(y_pred) + torch.abs(y_true))  # more robust implementation?
    # return torch.sum(torch.abs(y_pred - y_true)) / torch.sum(torch.abs(y_pred + y_true))   # DKI implementation
    # return torch.sum(torch.abs(y_pred - y_true)) / (2.0 * y_pred.shape[0])   # simplified by assuming every vector is a species composition that sums to 1 ... but some models may violate that constraint so maybe don't use this

def distribution_error(x):  # penalties for invalid distributions
    a = 1.0
    b = 1.0
    feature_penalty = torch.sum(torch.clamp(torch.abs(x - 0.5) - 0.5, min=0.0)) / x.shape[0]  # each feature penalized for distance from range [0,1]
    sum_penalty = torch.sum(torch.abs(torch.sum(x, dim=1) - 1.0)) / x.shape[0]  # sum penalized for distance from 1.0
    return a*feature_penalty + b*sum_penalty


def ceildiv(a, b):
    return -(a // -b)


def validate_epoch(model, x_val, y_val, minibatch_examples, t, loss_fn, distr_error_fn, device):
    model.eval()
    
    total_loss = 0
    total_distr_error = 0
    total_samples = x_val.size(0)
    minibatches = ceildiv(total_samples, minibatch_examples)
    current_index = 0  # Initialize current index to start of dataset

    for mb in range(minibatches):
        x, y, current_index = data.get_batch(x_val, y_val, minibatch_examples, current_index)
        mb_examples = x.size(0)

        with torch.no_grad():
            y_pred = model(t, x).to(device)

            loss = loss_fn(y_pred, y)
            distr_error = distr_error_fn(y_pred)
            total_loss += loss.item() * mb_examples  # Multiply loss by batch size
            total_distr_error += distr_error.item() * mb_examples

    avg_loss = total_loss / total_samples
    avg_penalty = total_distr_error / total_samples
    return avg_loss, avg_penalty


def train_epoch(model, x_train, y_train, minibatch_examples, accumulated_minibatches, optimizer, t, outputs_per_epoch,
                prev_examples, fold, epoch_num, model_name, dataname, loss_fn, distr_error_fn, device, verbosity=1):
    model.train()

    filepath_out_incremental = f'results/logs/{model_name}_{dataname}_incremental.csv'
    
    total_loss = 0
    total_penalty = 0
    total_samples = x_train.size(0)
    minibatches = ceildiv(total_samples, minibatch_examples)
    current_index = 0  # Initialize current index to start of dataset
    new_examples = 0

    stream_interval = max(1, minibatches // outputs_per_epoch)

    optimizer.zero_grad()

    # set up metrics for streaming
    prev_time = time.time()
    stream_loss = 0
    stream_penalty = 0
    stream_examples = 0

    # TODO: shuffle the data before starting an epoch
    for mb in range(minibatches):

        x, y, current_index = data.get_batch(x_train, y_train, minibatch_examples, current_index) #
        mb_examples = x.size(0)
        
        if current_index >= total_samples:
            current_index = 0  # Reset index if end of dataset is reached
        
        y_pred = model(t, x).to(device)

        loss = loss_fn(y_pred, y)
        actual_loss = loss.item() * mb_examples
        loss = loss / accumulated_minibatches # Normalize the loss by the number of accumulated minibatches, since loss function can't normalize by this

        loss.backward()

        distr_error = distr_error_fn(y_pred)
        actual_penalty = distr_error.item() * mb_examples
        
        #del y_pred, loss, distr_error

        total_loss += actual_loss
        total_penalty += actual_penalty
        new_examples += mb_examples
        
        stream_loss += actual_loss
        stream_penalty += actual_penalty
        stream_examples += mb_examples

        if (mb + 1) % stream_interval == 0:
            end_time = time.time()
            examples_per_second = stream_examples / max(end_time - prev_time, 0.0001)  # TODO: Find a better way to handle div by zero, or at least a more appropriate nonzero value
            stream.stream_results(filepath_out_incremental, verbosity > 0,
               "fold", fold+1,
                "epoch", epoch_num+1,
                "minibatch", mb+1,
                "total examples seen", prev_examples + new_examples,
                "Avg Loss", stream_loss / stream_examples,
                "Avg Distr Error", stream_penalty / stream_examples,
                "Examples per second", examples_per_second)
            stream_loss = 0
            stream_penalty = 0
            prev_time = end_time
            stream_examples = 0

        if ((mb + 1) % accumulated_minibatches == 0) or (mb == minibatches - 1):
            optimizer.step()
            optimizer.zero_grad()

        #del x, y


    avg_loss = total_loss / total_samples
    avg_penalty = total_penalty / total_samples
    new_total_examples = prev_examples + new_examples
    return avg_loss, avg_penalty, new_total_examples


def run_epochs(model, min_epochs, max_epochs, minibatch_examples, accumulated_minibatches, LR, x_train, y_train, x_valid, y_valid, t,
               model_name, dataname, fold, loss_fn, distr_error_fn, weight_decay, device,
               earlystop_patience=10, outputs_per_epoch=10, verbosity=1):
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=weight_decay)
    
    best_epoch_loss = float('inf')
    best_epoch_trn_loss = float('inf')
    best_epoch = -1
    best_epoch_time = -1
    epochs_worsened = 0
    early_stop = False
    
    filepath_out_epoch = f'results/logs/{model_name}_{dataname}_epochs.csv'
    filepath_out_model = f'results/logs/{model_name}_{dataname}_model.pth'
    
    # initial validation benchmark
    l_val, p_val = validate_epoch(model, x_valid, y_valid, minibatch_examples, t, loss_fn, distr_error_fn, device)
    stream.stream_results(filepath_out_epoch, verbosity > 0,
        "fold", fold+1,
        "epoch", 0,
        "training examples", 0,
        "Avg Training Loss", -1.0,
        "Avg Training Distr Error", -1.0,
        "Avg Validation Loss", l_val,
        "Avg Validation Distr Error", p_val,
        "Elapsed Time", 0.0,
        "GPU Footprint (MB)", -1.0,
        prefix="================PRE-VALIDATION===============\n",
        suffix="\n=============================================\n")


    train_examples_seen = 0
    start_time = time.time()
    
    for e in range(max_epochs):
        l_trn, p_trn, train_examples_seen = train_epoch(model, x_train, y_train, minibatch_examples, accumulated_minibatches,
                                                 optimizer, t, outputs_per_epoch, train_examples_seen, fold, e,
                                                 model_name, dataname, loss_fn, distr_error_fn, device, verbosity - 1)
        l_val, p_val = validate_epoch(model, x_valid, y_valid, minibatch_examples, t, loss_fn, distr_error_fn, device)
        # l_trn = validate_epoch(model, x_train, y_train, minibatch_examples, t, loss_fn, device) # Sanity test, should use running loss from train_epoch instead as a cheap approximation
        
        current_time = time.time()
        elapsed_time = current_time - start_time
        gpu_memory_reserved = torch.cuda.memory_reserved(device)
        
        stream.stream_results(filepath_out_epoch, verbosity > 0,
            "fold", fold+1,
            "epoch", e+1,
            "training examples", train_examples_seen,
            "Avg Training Loss", l_trn,
            "Avg Training Distr Error", l_trn,
            "Avg Validation Loss", l_val,
            "Avg Validation Distr Error", p_val,
            "Elapsed Time", elapsed_time,
            "GPU Footprint (MB)", gpu_memory_reserved / (1024 ** 2),
            prefix="==================VALIDATION=================\n",
            suffix="\n=============================================\n")
        
        # early stopping & model backups
        if l_val < best_epoch_loss:
            best_epoch = e
            best_epoch_loss = l_val
            best_epoch_trn_loss = l_trn
            best_epoch_time = elapsed_time
            epochs_worsened = 0
            torch.save(model.state_dict(), filepath_out_model)
        else:
            epochs_worsened += 1
            if verbosity > 0 and e+1 >= min_epochs:
                print(f"VALIDATION DIDN'T IMPROVE. PATIENCE {epochs_worsened}/{earlystop_patience}")
            # early stop
            if epochs_worsened >= earlystop_patience and e+1 >= min_epochs:
                if verbosity > 0:
                    print(f'Early stopping triggered after {e + 1} epochs.')
                early_stop = True
                break
    
    if verbosity > 0:
        if not early_stop:
            print(f"Completed training. Optimal loss was {best_epoch_loss}")
        else:
            print(f"Training stopped early due to lack of improvement in validation loss. Optimal loss was {best_epoch_loss}")
    
    # TODO: Check if this is the best model of a given name, and if so, save the weights and logs to a separate folder for that model name
    # TODO: could also try to save the source code, but would need to copy it at time of execution and then rename it if it gets the best score.
    
    return best_epoch_loss, best_epoch, best_epoch_time, best_epoch_trn_loss


def crossvalidate_model(LR, accumulated_minibatches, data_folded, device, earlystop_patience, kfolds, min_epochs, max_epochs,
                        minibatch_examples, model_constr, model_args, model_name, dataname, timesteps, loss_fn, distr_error_fn, weight_decay, verbosity=1):
    
    filepath_out_fold = f'results/logs/{model_name}_{dataname}_folds.csv'
    
    fold_losses = []
    fold_trn_losses = []
    fold_epochs = []
    fold_times = []
    for fold_num, data_fold in enumerate(data_folded):
        model = model_constr(model_args).to(device)
        
        x_train, y_train, x_valid, y_valid = data_fold
        print(f"Fold {fold_num + 1}/{kfolds}")
        
        val, e, elapsed_time, trn_loss = run_epochs(model, min_epochs, max_epochs, minibatch_examples, accumulated_minibatches, LR, x_train, y_train,
                            x_valid, y_valid, timesteps, model_name, dataname, fold_num, loss_fn, distr_error_fn, weight_decay, device,
                            earlystop_patience=earlystop_patience, outputs_per_epoch=10, verbosity=verbosity-1)
        fold_losses.append(val)
        fold_epochs.append(e)
        fold_trn_losses.append(trn_loss)
        fold_times.append(elapsed_time)
        
        stream.stream_results(filepath_out_fold, verbosity > 0,
            "fold", fold_num+1,
            "Validation Loss", val,
            "epochs", e+1,
            "time", elapsed_time,
            "training loss", trn_loss,
            prefix="\n========================================FOLD=========================================\n",
            suffix="\n=====================================================================================\n")
        
    return fold_losses, fold_epochs, fold_times, fold_trn_losses


def pessimistic_summary(fold_losses):
    model_score = np.max([np.mean(fold_losses), np.median(fold_losses)])
    return model_score


def main():
    
    # Experiment parameters
    
    # dataname = "waimea"
    # dataname = "waime a-condensed"
    dataname = "cNODE-paper-ocean"
    # dataname = "cNODE-paper-human-gut"
    # dataname = "cNODE-paper-human-oral"
    # dataname = "cNODE-paper-drosophila"
    # dataname = "cNODE-paper-soil-vitro"
    # dataname = "cNODE-paper-soil-vivo"
    # dataname = "dki-synth"
    # dataname = "dki-real"
    
    kfolds = 2
    min_epochs = 20
    max_epochs = 500
    earlystop_patience = 5
    
    # device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    
    # load data
    filepath_train = f'data/{dataname}_train.csv'
    x, y = data.load_data(filepath_train, device)
    data_folded = data.fold_data(x, y, kfolds)
    print('dataset:', filepath_train)
    print(f'training data shape: {data_folded[0][0].shape}')
    print(f'validation data shape: {data_folded[0][2].shape}')
    
    # get dimensions of data for model construction
    _, data_dim = x.shape
    
    
    # Hyperparameters to tune
    minibatch_examples = 1
    accumulated_minibatches = 1
    LR_base = 0.002
    WD_base = 0.00003
    
    hidden_dim = math.isqrt(data_dim)
    attend_dim = 16 # math.isqrt(hidden_dim)
    num_heads = 4
    depth = 2
    ffn_dim_multiplier = 0.5
    assert attend_dim % num_heads == 0, "attend_dim must be divisible by num_heads"
    
    # Specify model(s) for experiment
    # Note that each must be a constructor function that takes a dictionary args. Lamda is recommended.
    models_to_test = {
        # 'baseline-cNODE0': lambda args: models_baseline.cNODE0(data_dim),
        # 'baseline-SLP': lambda args: models_baseline.SingleLayerPerceptron(data_dim),
        # 'baseline-SLPMult': lambda args: models_baseline.SingleLayerMultiplied(data_dim),
        # 'baseline-SLPSum': lambda args: models_baseline.SingleLayerSummed(data_dim),
        # 'baseline-SLPMultSum': lambda args: models_baseline.SingleLayerMultipliedSummed(data_dim),
        # 'baseline-SLPReplicator': lambda args: models_baseline.SingleLayerReplicator(data_dim),
        # 'baseline-cNODE2-width1': lambda args: models.cNODEGen_ConstructedFitness(lambda: nn.Sequential(
        #     nn.Linear(data_dim, 1),
        #     nn.Linear(1, data_dim))),
        
        'canODE-noValue': lambda args: models_condensed.canODE_attentionNoValue(data_dim, args["attend_dim"], args["attend_dim"]),
        # 'canODE-noValue-static': lambda args: models_condensed.canODE_attentionNoValue_static(data_dim, args["attend_dim"], args["attend_dim"]),
        'canODE': lambda args: models_condensed.canODE_attention(data_dim, args["attend_dim"], args["attend_dim"]),
        'canODE-multihead': lambda args: models_condensed.canODE_attentionMultihead(data_dim, args["attend_dim"], args["num_heads"]),
        'canODE-singlehead': lambda args: models_condensed.canODE_attentionMultihead(data_dim, args["attend_dim"], 1),
        'canODE-transformer': lambda args: models_condensed.canODE_transformer(data_dim, args["attend_dim"], args["num_heads"], args["depth"], args["ffn_dim_multiplier"]),
        # 'canODE-transformer-d2': lambda args: models_condensed.canODE_transformer(data_dim, args["attend_dim"], args["num_heads"], 2, args["ffn_dim_multiplier"]),
        # 'canODE-transformer-d6': lambda args: models_condensed.canODE_transformer(data_dim, args["attend_dim"], args["num_heads"], 6, args["ffn_dim_multiplier"]),
        # 'canODE-transformer-d6-old': lambda args: models_condensed.canODE_transformer(data_dim, args["attend_dim"], 4, 6, args["ffn_dim_multiplier"]),
        # 'canODE-transformer-d3-a8-h2-f0.5': lambda args: models_condensed.canODE_transformer(data_dim, 8, 2, 3, 0.5),
        
        # 'cNODE2-custom': lambda args: models.cNODEGen_ConstructedFitness(lambda: nn.Sequential(
        #     nn.Linear(data_dim, args["hidden_dim"]),
        #     nn.Linear(args["hidden_dim"], data_dim))),
        # 'cNODE2-custom-nl': lambda args: models.cNODEGen_ConstructedFitness(lambda: nn.Sequential(
        #     nn.Linear(data_dim, args["hidden_dim"]),
        #     nn.ReLU(),
        #     nn.Linear(args["hidden_dim"], data_dim))),
        # 'cNODE-deep3': lambda args: models.cNODEGen_ConstructedFitness(lambda: nn.Sequential(
        #     nn.Linear(data_dim, args["hidden_dim"]),
        #     nn.Linear(args["hidden_dim"], args["hidden_dim"]),
        #     nn.Linear(args["hidden_dim"], data_dim))),
        # 'cNODE-deep3-nl': lambda args: models.cNODEGen_ConstructedFitness(lambda: nn.Sequential(
        #     nn.Linear(data_dim, args["hidden_dim"]),
        #     nn.ReLU(),
        #     nn.Linear(args["hidden_dim"], args["hidden_dim"]),
        #     nn.ReLU(),
        #     nn.Linear(args["hidden_dim"], data_dim))),
        # 'cNODE-deep4-flat': lambda args: models.cNODEGen_ConstructedFitness(lambda: nn.Sequential(
        #     nn.Linear(data_dim, args["hidden_dim"]),
        #     nn.Linear(args["hidden_dim"], args["hidden_dim"]),
        #     nn.Linear(args["hidden_dim"], args["hidden_dim"]),
        #     nn.Linear(args["hidden_dim"], data_dim))),
        # 'cNODE-deep4-flat-nl': lambda args: models.cNODEGen_ConstructedFitness(lambda: nn.Sequential(
        #     nn.Linear(data_dim, args["hidden_dim"]),
        #     nn.ReLU(),
        #     nn.Linear(args["hidden_dim"], args["hidden_dim"]),
        #     nn.ReLU(),
        #     nn.Linear(args["hidden_dim"], args["hidden_dim"]),
        #     nn.ReLU(),
        #     nn.Linear(args["hidden_dim"], data_dim))),
        # 'cAttend-simple': lambda args: models_condensed.cAttend_simple(data_dim, args["attend_dim"], args["attend_dim"]),
        # 'cNODE1': lambda args: models.cNODE1(data_dim),
        # 'cNODE2': lambda args: models.cNODE2(data_dim),
        # 'Embedded-cNODE2': lambda args: models.Embedded_cNODE2(data_dim, args["hidden_dim"]),  # this model is not good
        # 'cNODE2_DKI': lambda args: models.cNODE2_DKI(data_dim), # sanity test, this is the same as cNODE2 but less optimized
        # 'cNODE2-Gen': lambda args: models.cNODEGen_ConstructedFitness(lambda: nn.Sequential(nn.Linear(data_dim, data_dim), nn.Linear(data_dim, data_dim))),  # sanity test, this is the same as cNODE2 but generated at runtime
        # "cNODE2-static": lambda args: models.cNODE2_ExternalFitness(data_dim),
        # "cNODE2-FnFitness": lambda args: models.cNODE2_FnFitness(data_dim), # sanity test, this is the same as cNODE2 but testing externally-supplied fitness functions
    }


    # adjusted learning rate and decay
    LR = LR_base * math.sqrt(minibatch_examples * accumulated_minibatches)
    WD = WD_base * math.sqrt(minibatch_examples * accumulated_minibatches)

    # specify loss function
    loss_fn = loss_bc
    # loss_fn = lambda y_pred,y_true: loss_bc(y_pred, y_true) + distribution_error(y_pred)
    
    distr_error_fn = distribution_error
    
    # time step "data"
    ode_timesteps = 2  # must be at least 2. TODO: run this through hyperparameter opt to verify that it doesn't impact performance
    timesteps = torch.arange(0.0, 1.0, 1.0 / ode_timesteps).to(device)
    
    # START of hacky hyperparam search - remove
    # for hidden_dim in [1, 2, 4, 8, 16, 32, 64]:
    #     attend_dim = hidden_dim
    #     # num_heads = attend_dim
    #     num_heads = {1:1, 2:2, 4:2, 8:4, 16:4, 32:8, 64:8}[attend_dim]
    #     # END of hacky hyperparam search - remove

    # START of hacky hyperparam search - remove
    # for num_heads in [2, 4, 8, 16]:
    #     for head_dim in [4, 8, 16, 32]:
    #         attend_dim = num_heads * head_dim
    #         if attend_dim > data_dim:
    #             continue
    #         for ffn_dim_multiplier in [0.5, 1.0, 2.0]:
    #             for depth in [2, 3]:
    #     # END of hacky hyperparam search - remove
                    
    model_args = {"hidden_dim": hidden_dim, "attend_dim": attend_dim, "num_heads": num_heads, "depth": depth, "ffn_dim_multiplier": ffn_dim_multiplier}
    
    filepath_out_expt = f'results/{dataname}_experiments.csv'
    for model_name, model_constr in models_to_test.items():
        
        # # TODO remove this: it's just to resume from where we were previously
        # if ((attend_dim == 4 or attend_dim == 16) and model_name == 'canODE-transformer-d6' and num_heads == 4):
        #     continue
        
        try:
            print(f"\nRunning model: {model_name}")
        
            # test construction and print parameter count
            model = model_constr(model_args)
            num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Number of parameters in model: {num_params}")
            
            fold_losses, fold_epochs, fold_times, fold_trn_losses = crossvalidate_model(LR, accumulated_minibatches, data_folded, device, earlystop_patience,
                                              kfolds, min_epochs, max_epochs, minibatch_examples, model_constr, model_args,
                                              model_name, dataname, timesteps, loss_fn, distr_error_fn, WD, verbosity=3)
            
    
            print(f"Val Losses: {fold_losses}")
            print(f"Epochs: {fold_epochs}")
            print(f"Durations: {fold_times}")
            print(f"Trn Losses: {fold_trn_losses}")
            
            for i in len(fold_losses):
                stream.stream_scores(filepath_out_expt, True,
                    "model", model_name,
                    "model parameters", num_params,
                    "Validation Score", fold_losses,
                    "@ Epoch", fold_epochs,
                    "@ Elapsed Time", fold_times,
                    "@ Training Loss", fold_trn_losses,
                    "k-folds", kfolds,
                    "early stop patience", earlystop_patience,
                    "minibatch_examples", minibatch_examples,
                    "accumulated_minibatches", accumulated_minibatches,
                    "learning rate", LR,
                    "weight decay", WD,
                    "LR_base", LR_base,
                    "WD_base", WD_base,
                    "timesteps", ode_timesteps,
                     *list(itertools.chain(*model_args.items())), # unroll the model args dictionary
                    prefix="\n=======================================================EXPERIMENT========================================================\n",
                    suffix="\n=========================================================================================================================\n")
            
            # summary score for each result vector
            model_score = pessimistic_summary(fold_losses)
            model_epoch = pessimistic_summary(fold_epochs)
            model_time = pessimistic_summary(fold_times)
            model_trn_loss = pessimistic_summary(fold_trn_losses)
            
            print(f"Avg Val Losses: {model_score}")
            print(f"Avg Epochs: {model_epoch}")
            print(f"Avg Durations: {model_time}")
            print(f"Avg Trn Losses: {model_trn_loss}")
            
        except Exception as e:
            stream.stream_scores(filepath_out_expt, True,
                "model", model_name,
                "model parameters", -1,
                "Avg Validation Score", -1,
                "@ Avg Epoch", -1,
                "@ Avg Elapsed Time", -1,
                "@ Avg Training Loss", -1,
                "k-folds", kfolds,
                "early stop patience", earlystop_patience,
                "minibatch_examples", minibatch_examples,
                "accumulated_minibatches", accumulated_minibatches,
                "learning rate", LR,
                "weight decay", WD,
                "LR_base", LR_base,
                "WD_base", WD_base,
                "timesteps", ode_timesteps,
                 *list(itertools.chain(*model_args.items())), # unroll the model args dictionary
                prefix="\n=======================================================EXPERIMENT========================================================\n",
                suffix="\n=========================================================================================================================\n")
            print(f"Model {model_name} failed with error:\n{e}")


# main
if __name__ == "__main__":
    main()

# TODO: hyperparameter optimization
# TODO: time limit and/or time based early stopping
# TODO: hyperparameters optimization based on loss change rate against clock time, somehow
# TODO: Confirm that changes of t shape do not alter performance, then remove t argument to model forward for non-ODEfuncs. The t can be generated in forward to pass to odeint.
# TODO: realtime visualization
# TODO: Add param count to filename of logs
# TODO: Record each fold as a separate row instead of saving the summary statistic. I can summarize during graphing.

# TODO: Try transfer learning with shared ODE but separate embed/unembed - espcially x-shaped conjoined networks for joint learning. Or Reptile for similar metalearning.
# TODO: Create a parameterized generalized version of the canODE models so I can explore the model architectures as hyperparameters
# TODO: Attention layers in a DEQ (deep equilibrium model) similar to the nODE to produce F(x) for the ODE
# TODO: (Probably not a good idea) As an alternative to attention, try condensing into a high enough space that we can still use channels as IDs (minimum size would be the largest summed assemblage in the dataset) and then use a vanilla (but smaller) cNODE. The difficulty here is that the embed mapping needs to be dynamic because many input channels will map to the same embedded channels and some of those could occur at the same time. There effectively needs to be the ability to decide on the fly that "A should go in M but that's already occupied by B, so A will go in N instead, which has the same properties as M" ... which means the embedding needs to have redundancies. I guess I could hardcode there being a few redundant copies of each channel somehow (shared weights?) but I don't like that idea. In general this seems much weaker than using attention to divorce the species representations from the preferred basis, allowing them to share dynamics to exactly the extent that is helpful via independent subspaces.
# TODO: Test baseline ODE model that does not learn F(x) at all - either it's just torch.ones() or it's a torch.random() at initialization or it's a torch.random() in the forward call.
# TODO: Test multi-layer attention-based models that are not fed into ODEs.
