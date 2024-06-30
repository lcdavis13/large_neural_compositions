import math
import csv
import time
import numpy as np
import torch
import torch.nn as nn

import data
import models


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


def stream_results(filename, print_console, *args, prefix="", suffix=""):
    if len(args) % 2 != 0:
        raise ValueError("Arguments should be in pairs of names and values.")
    
    names = args[0::2]
    values = args[1::2]
    
    if print_console:
        print(prefix + (", ".join([f"{name}: {value}" for name, value in zip(names, values)])) + suffix)
    
    # Check if file exists
    if filename:
        # Initialize the set of filenames if it doesn't exist
        if not hasattr(stream_results, 'filenames'):
            stream_results.filenames = set()
        
        # Check if it's the first time the function is called for this filename
        if filename not in stream_results.filenames:
            stream_results.filenames.add(filename)
            file_started = False
        else:
            file_started = True
        
        mode = 'a' if file_started else 'w'
        with open(filename, mode, newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # If the file is new, write the header row
            if not file_started:
                writer.writerow(names)
            
            # Write the values row
            writer.writerow(values)


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
                prev_examples, fold, epoch_num, model_name, dataname, loss_fn, distr_error_fn, device, verbose=True):
    model.train()

    filepath_out_incremental = f'results/{model_name}_{dataname}_incremental.csv'
    
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
            examples_per_second = stream_examples / (end_time - prev_time)
            stream_results(filepath_out_incremental, verbose,
               "fold", fold,
                "epoch", epoch_num,
                "minibatch", mb,
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


def run_epochs(model, max_epochs, minibatch_examples, accumulated_minibatches, LR, x_train, y_train, x_valid, y_valid, t,
               model_name, dataname, fold, loss_fn, distr_error_fn, weight_decay, device,
               earlystop_patience=10, outputs_per_epoch=10, verbose=True):
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=weight_decay)
    best_loss = float('inf')
    best_epoch = -1
    epochs_worsened = 0
    early_stop = False
    
    filepath_out_epoch = f'results/{model_name}_{dataname}_epochs.csv'
    filepath_out_model = f'results/{model_name}_{dataname}_model.pth'

    train_examples_seen = 0
    start_time = time.time()
    
    # initial validation benchmark
    l_val, p_val = validate_epoch(model, x_valid, y_valid, minibatch_examples, t, loss_fn, distr_error_fn, device)
    stream_results(filepath_out_epoch, verbose,
        "fold", fold,
        "epoch", -1,
        "training examples", 0,
        "Avg Training Loss", -1.0,
        "Avg Validation Loss", l_val,
        "Avg Distribution Distr Error", p_val,
        "Elapsed Time", -1.0,
        "GPU Footprint (MB)", -1.0,
        prefix="================PRE-VALIDATION===============\n",
        suffix="\n=============================================\n")
    
    
    for e in range(max_epochs):
        l_trn, p_trn, train_examples_seen = train_epoch(model, x_train, y_train, minibatch_examples, accumulated_minibatches,
                                                 optimizer, t, outputs_per_epoch, train_examples_seen, fold, e,
                                                 model_name, dataname, loss_fn, distr_error_fn, device, verbose)
        l_val, p_val = validate_epoch(model, x_valid, y_valid, minibatch_examples, t, loss_fn, distr_error_fn, device)
        # l_trn = validate_epoch(model, x_train, y_train, minibatch_examples, t, loss_fn, device) # Sanity test, should use running loss from train_epoch instead as a cheap approximation
        
        current_time = time.time()
        elapsed_time = current_time - start_time
        gpu_memory_reserved = torch.cuda.memory_reserved(device)
        
        stream_results(filepath_out_epoch, verbose,
            "fold", fold,
            "epoch", e,
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
        if l_val < best_loss:
            best_loss = l_val
            best_epoch = e
            epochs_worsened = 0
            torch.save(model.state_dict(), filepath_out_model)
        else:
            epochs_worsened += 1
            if verbose:
                print(f"VALIDATION DIDN'T IMPROVE. PATIENCE {epochs_worsened}/{earlystop_patience}")
            # early stop
            if epochs_worsened >= earlystop_patience:
                if verbose:
                    print(f'Early stopping triggered after {e + 1} epochs.')
                early_stop = True
                break
    
    if verbose:
        if not early_stop:
            print(f"Completed training. Optimal loss was {best_loss}")
        else:
            print(f"Training stopped early due to lack of improvement in validation loss. Optimal loss was {best_loss}")
    
    # TODO: Check if this is the best model of a given name, and if so, save the weights and logs to a separate folder for that model name
    # TODO: could also try to save the source code, but would need to copy it at time of execution and then rename it if it gets the best score.
    
    return best_loss, best_epoch


def crossvalidate_model(LR, accumulated_minibatches, data_folded, device, earlystop_patience, kfolds, max_epochs,
                        minibatch_examples, model_constr, model_name, dataname, timesteps, loss_fn, distr_error_fn, weight_decay, verbose=True):
    
    filepath_out_fold = f'results/{model_name}_{dataname}_folds.csv'
    
    fold_losses = []
    for fold_num, data_fold in enumerate(data_folded):
        model = model_constr().to(device)
        
        x_train, y_train, x_valid, y_valid = data_fold
        print(f"Fold {fold_num + 1}/{kfolds}")
        
        val, e = run_epochs(model, max_epochs, minibatch_examples, accumulated_minibatches, LR, x_train, y_train,
                            x_valid, y_valid, timesteps, model_name, dataname, fold_num, loss_fn, distr_error_fn, weight_decay, device,
                            earlystop_patience=earlystop_patience, outputs_per_epoch=10, verbose=verbose)
        fold_losses.append(val)
        
        stream_results(filepath_out_fold, verbose,
                       "fold", fold_num,
                       "Validation Loss", val,
                       "epochs", e,
                       prefix="\n========================================FOLD=========================================\n",
                       suffix="\n=====================================================================================\n")
    # min of mean and mode to avoid over-optimism from outliers
    model_score = np.min([np.mean(fold_losses), np.median(fold_losses)])
    print(f"Losses: {fold_losses}")
    return model_score


def main():
    
    # Experiment parameters
    
    # dataname = "waimea"
    # dataname = "waimea-condensed"
    # dataname = "dki-synth"
    dataname = "dki-real"
    
    kfolds = 3
    max_epochs = 50000
    earlystop_patience = 5
    
    
    # Hyperparameters to tune
    minibatch_examples = 500
    accumulated_minibatches = 1
    LR_base = 0.002
    WD_base = 0.0003


    # adjusted learning rate and decay
    LR = LR_base * math.sqrt(minibatch_examples * accumulated_minibatches)
    WD = WD_base * math.sqrt(minibatch_examples * accumulated_minibatches)
    
    # device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    
    # load data
    filepath_train = f'data/{dataname}_train.csv'
    x, y = data.load_data(filepath_train, device)
    data_folded = data.fold_data(x, y, kfolds)
    print('dataset:', filepath_train)
    print(f'training data shape: {data_folded[0][0].shape}')
    print(f'validation data shape: {data_folded[0][1].shape}')
    
    # get dimensions of data for model construction
    _, N = x.shape
    n_root = math.isqrt(N)
    
    
    # Specify model(s) for experiment
    # Note that each must be a constructor function with no args, not a pre-constructed model. Lamda is recommended.
    models_to_test = {
        'cNODE-slim': lambda: models.cNODE_Gen(lambda: nn.Sequential(
            nn.Linear(N, n_root),
            nn.Linear(n_root, n_root),
            nn.Linear(n_root, N))),
        # 'cNODE2': lambda: models.cNODE2(N),
        # 'Embedded-cNODE2': lambda: models.Embedded_cNODE2(N, n_root),  # this model is not good
        # 'cNODE2_DKI': lambda: cNODE2_DKI(N), # sanity test, this is the same as cNODE2 but less optimized
        # 'cNODE2-Gen': lambda: models.cNODE_Gen(lambda: nn.Sequential(nn.Linear(N, N), nn.Linear(N, N))),  # sanity test, this is the same as cNODE2 but generated at runtime
        # "cNODE2-GenRun": lambda: models.cNODE2_GenRun(N), # sanity test, this is the same as cNODE2 but with f(x) computed outside the ODE
    }


    # specify loss function
    loss_fn = loss_bc
    # loss_fn = lambda y_pred,y_true: loss_bc(y_pred, y_true) + distribution_error(y_pred)
    
    distr_error_fn = distribution_error
    
    # time step "data"
    ode_timesteps = 2  # must be at least 2
    timesteps = torch.arange(0.0, 1.0, 1.0 / ode_timesteps).to(device)
    
    for model_name, model_constr in models_to_test.items():
        
        
        print(f"\nRunning model: {model_name}")
    
        # test construction and print parameter count
        model = model_constr()
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Number of parameters in model: {num_params}")
        
        model_score = crossvalidate_model(LR, accumulated_minibatches, data_folded, device, earlystop_patience,
                                          kfolds, max_epochs, minibatch_examples, model_constr,
                                          model_name, dataname, timesteps, loss_fn, distr_error_fn, WD, verbose=True)
        
        print(f"Model score: {model_score}\n")


# main
if __name__ == "__main__":
    main()
