import argparse
import os

# Set a default value if the environment variable is not specified
# os.environ.setdefault("SOLVER", "torchdiffeq")
# os.environ.setdefault("SOLVER", "torchdiffeq_memsafe")
# os.environ.setdefault("SOLVER", "torchode")
# os.environ.setdefault("SOLVER", "torchode_memsafe")
os.environ.setdefault("SOLVER", "trapezoid")

import copy
import itertools
import math
import time

import numpy as np
import pandas as pd
import torch
from dotsy import dicy
from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau
import torch.nn as nn

import data
import epoch_managers
import lr_schedule
import models_cnode
import models_baseline
import models_embedded
import stream
from optimum import Optimum, summarize, unrolloptims
from stream_plot import plotstream
import user_confirmation as ui
import psutil

import tracemalloc

tracemalloc.start()


def eval_model(model, x, timesteps):
    # evaluates models whether they require ODE timesteps or not
    requires_timesteps = getattr(model, 'USES_ODEINT', False)
    
    if requires_timesteps and timesteps is not None:
        # Call models that require the timesteps argument
        y_steps = model(timesteps, x)
    else:
        # Call models that do not require the timesteps argument
        y = model(x)
        y_steps = y.unsqueeze(0)
    
    return y_steps


def loss_bc_dki(y_pred, y_true):
    return torch.sum(torch.abs(y_pred - y_true)) / torch.sum(
        torch.abs(y_pred + y_true))  # DKI repo implementation (incorrect)


def loss_bc(y_pred, y_true):  # Bray-Curtis Dissimilarity
    return torch.mean(
        torch.sum(torch.abs(y_pred - y_true), dim=-1) / torch.sum(torch.abs(y_pred) + torch.abs(y_true), dim=-1))


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
    feature_penalty = torch.sum(torch.clamp(torch.abs(x - 0.5) - 0.5, min=0.0)) / x.shape[
        0]  # each feature penalized for distance from range [0,1]
    sum_penalty = torch.sum(torch.abs(torch.sum(x, dim=1) - 1.0)) / x.shape[0]  # sum penalized for distance from 1.0
    return a * feature_penalty + b * sum_penalty


def ceildiv(a, b):
    return -(a // -b)


def validate_epoch(model, x_val, y_val, minibatch_examples, t, loss_fn, score_fn, distr_error_fn, device):
    model.eval()
    
    total_loss = 0.0
    total_score = 0.0
    total_distr_error = 0.0
    total_samples = x_val.size(0)
    minibatches = ceildiv(total_samples, minibatch_examples)
    current_index = 0  # Initialize current index to start of dataset
    
    for mb in range(minibatches):
        z, current_index = data.get_batch(x_val, y_val, t, minibatch_examples, current_index, noise_level_x=0.0,
                                          noise_level_y=0.0)
        x, y = z[0], z[-1]
        mb_examples = x.size(0)
        
        with torch.no_grad():
            y_pred = eval_model(model, x, t)[-1]
            y_pred = y_pred.to(device)
            
            loss = loss_fn(y_pred, y)
            score = score_fn(y_pred, y)
            distr_error = distr_error_fn(y_pred)
            total_loss += loss.item() * mb_examples  # Multiply loss by batch size
            total_score += score.item() * mb_examples
            total_distr_error += distr_error.item() * mb_examples
    
    avg_loss = total_loss / total_samples
    avg_score = total_score / total_samples
    avg_penalty = total_distr_error / total_samples
    return avg_loss, avg_score, avg_penalty


def train_epoch(model, x_train, y_train, minibatch_examples, accumulated_minibatches, noise, interpolate, optimizer,
                scheduler, scaler, t,
                outputs_per_epoch,
                prev_examples, fold, epoch_num, model_name, dataname, loss_fn, score_fn, distr_error_fn, device,
                filepath_out_incremental, lr_plot=None, loss_plot=None, lr_loss_plot=None, verbosity=1,
                supervised_timesteps=1):
    model.train()
    
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
        model_requires_timesteps = getattr(model, 'USES_ODEINT', False)
        supervise_steps = interpolate and model_requires_timesteps
        
        z, current_index = data.get_batch(x_train, y_train, t, minibatch_examples, current_index, noise_level_x=noise,
                                          noise_level_y=noise)  #
        mb_examples = z.shape[-2]
        
        if current_index >= total_samples:
            current_index = 0  # Reset index if end of dataset is reached
            x_train, y_train = data.shuffle_data(x_train, y_train)
        
        y_pred = eval_model(model, z[0], t)
        if supervise_steps:
            y_pred = y_pred[1:]
            y_true = z[1:]
        else:
            y_pred = y_pred[-1:]
            y_true = z[-1:]
        y_pred = y_pred.to(device)
        
        loss = loss_fn(y_pred, y_true)
        actual_loss = loss.item() * mb_examples
        loss = loss / accumulated_minibatches  # Normalize the loss by the number of accumulated minibatches, since loss function can't normalize by this
        
        scaled_loss = scaler.scale(loss)
        if scaled_loss.requires_grad and scaled_loss.grad_fn is not None:
            scaled_loss.backward()
        else:
            print(
                f"GRADIENT ERROR: Loss at epoch {epoch_num} minibatch {mb} does not require gradient. Computation graph detached?")
        
        distr_error = distr_error_fn(y_pred)
        actual_penalty = distr_error.item() * mb_examples
        
        # del y_pred, loss, distr_error
        
        total_loss += actual_loss
        total_penalty += actual_penalty
        new_examples += mb_examples
        
        stream_loss += actual_loss
        stream_penalty += actual_penalty
        stream_examples += mb_examples
        
        if (mb + 1) % stream_interval == 0:
            end_time = time.time()
            examples_per_second = stream_examples / max(end_time - prev_time,
                                                        0.0001)  # TODO: Find a better way to handle div by zero, or at least a more appropriate nonzero value
            stream.stream_results(filepath_out_incremental, verbosity > 0, verbosity > 0, verbosity > -1,
                                  "fold", fold,
                                  "epoch", epoch_num + 1,
                                  "minibatch", mb + 1,
                                  "total examples seen", prev_examples + new_examples,
                                  "Avg Loss", stream_loss / stream_examples,
                                  "Avg Distr Error", stream_penalty / stream_examples,
                                  "Examples per second", examples_per_second,
                                  "Learning Rate", scheduler.get_last_lr(),
                                  )
            if lr_plot:
                plotstream.plot_single(lr_plot, "epochs", "LR", f"{model_name} fold {fold}",
                                       epoch_num + mb / minibatches, scheduler.get_last_lr(), False, y_log=False)
            if loss_plot:
                plotstream.plot_loss(loss_plot, f"{model_name} fold {fold}", epoch_num + mb / minibatches,
                                     stream_loss / stream_examples, None, add_point=False)
            if lr_loss_plot:
                plotstream.plot_single(lr_loss_plot, "log( Learning Rate )", "Loss", f"{model_name} fold {fold}",
                                       scheduler.get_last_lr(), stream_loss / stream_examples, False, x_log=True)
            stream_loss = 0
            stream_penalty = 0
            prev_time = end_time
            stream_examples = 0
        
        if ((mb + 1) % accumulated_minibatches == 0) or (mb == minibatches - 1):
            scaler.step(optimizer)
            scaler.update()
            scheduler.batch_step()  # TODO: Add accum_loss metric in case I ever want to do ReduceLROnPlateau with batch_step mode
            optimizer.zero_grad()
        
        # del x, y
    
    avg_loss = total_loss / total_samples
    avg_penalty = total_penalty / total_samples
    new_total_examples = prev_examples + new_examples
    return avg_loss, avg_penalty, new_total_examples


# backup model parameters for reptile
def backup_parameters(model):
    return {name: param.clone() for name, param in model.state_dict().items()}


# perform weighted averaging of parameters for reptile
def weighted_average_parameters(model, paramsA, paramsB, alpha=0.5):
    averaged_params = {
        key: alpha * paramsA[key] + (1 - alpha) * paramsB[key]
        for key in paramsA
    }
    model.load_state_dict(averaged_params)


def run_epochs(model, optimizer, scheduler, manager, minibatch_examples, accumulated_minibatches, noise, interpolate,
               scaler, x_train,
               y_train, x_valid, y_valid, t,
               model_name, dataname, fold, loss_fn, score_fn, distr_error_fn, device, outputs_per_epoch=10, verbosity=1,
               reptile_rate=0.0, reeval_train=False, jobstring=""):
    # assert(data.check_leakage([(x_train, y_train, x_valid, y_valid)]))
    
    # track stats at various definitions of the "best" epoch
    val_opt = Optimum('val_loss', 'min')
    trn_opt = Optimum('trn_loss', 'min')
    valscore_opt = Optimum('val_score', 'min')
    trnscore_opt = Optimum('trn_score', 'min')
    last_opt = Optimum(metric=None)  # metric None to update it every time. metric="epoch" would do the same
    
    old_lr = scheduler.get_last_lr()
    
    filepath_out_epoch = f'results/epochs/{model_name}_{dataname}{jobstring}_epochs.csv'
    # filepath_out_model = f'results/logs/{model_name}_{dataname}_model.pth'
    filepath_out_incremental = f'results/incr/{model_name}_{dataname}{jobstring}_incremental.csv'
    
    # initial validation benchmark
    l_val, score_val, p_val = validate_epoch(model, x_valid, y_valid, minibatch_examples, t, loss_fn, score_fn,
                                             distr_error_fn, device)
    l_trn, score_trn, p_trn = validate_epoch(model, x_train, y_train, minibatch_examples, t, loss_fn, score_fn,
                                             distr_error_fn, device)
    
    gpu_memory_reserved = torch.cuda.memory_reserved(device)
    _, cpuRam = tracemalloc.get_traced_memory()
    stream.stream_results(filepath_out_epoch, verbosity > 0, verbosity > 0, verbosity > -1,
                          "fold", fold,
                          "epoch", 0,
                          "training examples", 0,
                          "Avg Training Loss", l_trn,
                          "Avg DKI Trn Loss", score_trn,
                          "Avg Training Distr Error", p_trn,
                          "Avg Validation Loss", l_val,
                          "Avg DKI Val Loss", score_val,
                          "Avg Validation Distr Error", p_val,
                          "Learning Rate", old_lr,
                          "Elapsed Time", 0.0,
                          "VRAM (GB)", gpu_memory_reserved / (1024 ** 3),
                          "Peak RAM (GB)", cpuRam / (1024 ** 3),
                          prefix="================PRE-VALIDATION===============\n",
                          suffix="\n=============================================\n")
    plotstream.plot_loss(f"loss {dataname}", f"{model_name} fold {fold}", 0, l_trn, l_val, add_point=False)
    plotstream.plot_loss(f"score {dataname}", f"{model_name} fold {fold}", 0, score_trn, score_val, add_point=False)
    # plotstream.plot(dataname, "epoch", "loss", [f"{model_name} fold {fold} - Val", f"{model_name} fold {fold} - Trn", f"{model_name} fold {fold} - DKI Val", f"{model_name} fold {fold} - DKI Trn"], 0, [l_val, None, score_val, None], add_point=False)
    
    train_examples_seen = 0
    start_time = time.time()
    
    if reptile_rate > 0.0:
        # Create a copy of the model to serve as the meta-model
        meta_model = copy.deepcopy(model)
        outer_optimizer = type(optimizer)(meta_model.parameters())
        outer_optimizer.load_state_dict(optimizer.state_dict())
        outer_optimizer.lr = reptile_rate
    
    training_curve = []
    while True:
        l_trn, p_trn, train_examples_seen = train_epoch(model, x_train, y_train, minibatch_examples,
                                                        accumulated_minibatches, noise, interpolate, optimizer,
                                                        scheduler, scaler, t,
                                                        outputs_per_epoch, train_examples_seen,
                                                        fold, manager.epoch, model_name, dataname, loss_fn, score_fn,
                                                        distr_error_fn, device, filepath_out_incremental,
                                                        lr_plot="Learning Rate", verbosity=verbosity - 1)
        if reptile_rate > 0.0:
            # Meta-update logic
            with torch.no_grad():
                # Apply the difference as pseudo-gradients to the outer optimizer
                meta_weights = {name: param.clone() for name, param in meta_model.state_dict().items()}
                for name, param in meta_model.named_parameters():
                    param.grad = (meta_weights[name] - model.state_dict()[name])
                
                # Use the outer optimizer to step with these gradients
                outer_optimizer.step()
                
                # Synchronize the inner model with the updated meta-model weights
                model.load_state_dict(meta_model.state_dict())
        
        l_val, score_val, p_val = validate_epoch(model, x_valid, y_valid, minibatch_examples, t, loss_fn, score_fn,
                                                 distr_error_fn, device)
        if reeval_train:
            l_trn, score_trn, p_trn = validate_epoch(model, x_train, y_train, minibatch_examples, t, loss_fn, score_fn,
                                                     distr_error_fn, device)
        else:
            score_trn = -1.0
        
        # Update learning rate based on loss
        scheduler.epoch_step(l_trn)
        new_lr = scheduler.get_last_lr()
        lr_changed = not np.isclose(new_lr, old_lr)
        add_point = lr_changed and isinstance(scheduler.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
        
        current_time = time.time()
        elapsed_time = current_time - start_time
        gpu_memory_reserved = torch.cuda.memory_reserved(device)
        _, cpuRam = tracemalloc.get_traced_memory()
        
        stream.stream_results(filepath_out_epoch, verbosity > 0, verbosity > 0, verbosity > -1,
                          "fold", fold,
                              "epoch", manager.epoch + 1,
                              "training examples", train_examples_seen,
                              "Avg Training Loss", l_trn,
                              "Avg DKI Trn Loss", -1.0,
                              "Avg Training Distr Error", p_trn,
                              "Avg Validation Loss", l_val,
                              "Avg DKI Val Loss", score_val,
                              "Avg Validation Distr Error", p_val,
                              "Learning Rate", old_lr,  # should I track average LR in the epoch? Max and min LR?
                              "Elapsed Time", elapsed_time,
                              "VRAM (GB)", gpu_memory_reserved / (1024 ** 3),
                              "Peak RAM (GB)", cpuRam / (1024 ** 3),
                              prefix="==================VALIDATION=================\n",
                              suffix="\n=============================================\n")
        plotstream.plot_loss(f"loss {dataname}", f"{model_name} fold {fold}", manager.epoch + 1, l_trn, l_val,
                             add_point=add_point)
        plotstream.plot_loss(f"score {dataname}", f"{model_name} fold {fold}", manager.epoch + 1, score_trn, score_val,
                             add_point=add_point)
        # plotstream.plot(dataname, "epoch", "loss", [f"{model_name} fold {fold} - Val", f"{model_name} fold {fold} - Trn", f"{model_name} fold {fold} - DKI Val", f"{model_name} fold {fold} - DKI Trn"], manager.epoch + 1, [l_val, l_trn, score_val, score_trn], add_point=add_point)
        # if l_val != score_val:
        #     print("WARNING: CURRENT LOSS METRIC DISAGREES WITH DKI LOSS METRIC")
        
        # TODO: replace this quick and dirty dict packing. They should have always been in a dict.
        dict = {"epoch": manager.epoch, "trn_loss": l_trn, "trn_score": score_trn, "val_loss": l_val,
                "val_score": score_val, "lr": old_lr, "time": elapsed_time, "gpu_memory": gpu_memory_reserved,
                "metric": p_val}
        # track various optima
        val_opt.track_best(dict)
        valscore_opt.track_best(dict)
        trn_opt.track_best(dict)
        trnscore_opt.track_best(dict)
        last_opt.track_best(dict)
        
        training_curve.append(
            {"fold": fold, "epoch": manager.epoch, "trn_loss": l_trn, "val_loss": l_val, "time": elapsed_time})
        
        old_lr = new_lr
        
        # check if we should continue
        if manager.should_stop(last_opt):
            break
    
    # TODO: Check if this is the best model of a given name, and if so, save the weights and logs to a separate folder for that model name
    # TODO: could also try to save the source code, but would need to copy it at time of execution and then rename it if it gets the best score.
    
    return val_opt, valscore_opt, trn_opt, trnscore_opt, last_opt, training_curve

def train_and_evaluate_model(LR, scaler, accumulated_minibatches, x_train, y_train, x_test, y_test, noise, interpolate, device,
                             min_epochs, max_epochs, minibatch_examples, model_constr, model_args, model_name,
                             timesteps, loss_fn, score_fn, distr_error_fn, weight_decay, verbosity=1):
    model = model_constr(model_args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=weight_decay)
    manager = epoch_managers.FixedManager(max_epochs=min_epochs)

    base_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=100, cooldown=100,
                                       threshold_mode='rel', threshold=0.01)
    scheduler = lr_schedule.LRScheduler(base_scheduler, initial_lr=LR)

    print(f"Training model {model_name}")

    # Train the model
    run_epochs(model, optimizer, scheduler, manager, minibatch_examples, accumulated_minibatches, noise, interpolate,
               scaler, x_train, y_train, x_train[0:1], y_train[0:1], timesteps, model_name, "", 0, loss_fn, score_fn, distr_error_fn, device,
               outputs_per_epoch=10, verbosity=verbosity)

    # Evaluate the model
    l_sample, score_sample, penalty_sample = validate_epoch(model, x_test, y_test, minibatch_examples, timesteps, loss_fn,
                                                            score_fn, distr_error_fn, device)

    print(f"Test Loss: {l_sample}, Test Score: {score_sample}, Test Distribution Error: {penalty_sample}")

    return model, l_sample, score_sample, penalty_sample


def main():
    import argparse

    # device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    # Command-line argument for index
    parser = argparse.ArgumentParser(description='run')
    parser.add_argument('--index', type=int, required=False, help='Index for the configuration to use')
    args = parser.parse_args()
    args.index = 0

    # Predefined configurations
    configurations = [
        {
            "model": "baseline-SLPMultShaped",
            "noise_level": 0.075,
            "supervised_interpolation": True,
            "learning_rate": 0.1,
            "reptile_rate": 0.01,
            "epochs": 2
        },
        {
            "model": "baseline-SLPMultShaped",
            "noise_level": 0.075,
            "supervised_interpolation": True,
            "learning_rate": 0.1,
            "reptile_rate": 0.01,
            "epochs": 345
        },
        {
            "model": "baseline-SLPMultShaped",
            "noise_level": 0.0,
            "supervised_interpolation": False,
            "learning_rate": 0.1,
            "reptile_rate": 1.0,
            "epochs": 34
        },
        {
            "model": "cNODE1",
            "noise_level": 0.075,
            "supervised_interpolation": True,
            "learning_rate": 0.1,
            "reptile_rate": 0.5,
            "epochs": 233
        },
        {
            "model": "cNODE1",
            "noise_level": 0.0,
            "supervised_interpolation": False,
            "learning_rate": 0.1,
            "reptile_rate": 1.0,
            "epochs": 68
        }
    ]

    # Load selected configuration
    config = configurations[args.index]

    # Data
    dataname = "cNODE-paper-ocean-std"
    filepath_train = f'data/{dataname}_train.csv'
    filepath_test = f'data/{dataname}_test.csv'
    x_train, y_train = data.load_data(filepath_train, device)
    x_test, y_test = data.load_data(filepath_test, device)

    print(f'Training data shape: {x_train.shape}')
    print(f'Testing data shape: {x_test.shape}')

    # Select a single sample for evaluation
    sample_data = (x_train[0:1], y_train[0:1])

    # Set hyperparameters from config
    model_name = config["model"]
    noise_level = config["noise_level"]
    supervised_interpolation = config["supervised_interpolation"]
    learning_rate = config["learning_rate"]
    reptile_rate = config["reptile_rate"]
    epochs = config["epochs"]

    # Hyperparameters
    hp = dicy()
    hp.solver = os.getenv("SOLVER")
    hp.min_epochs = epochs
    hp.max_epochs = epochs
    hp.minibatch_examples = 20
    hp.accumulated_minibatches = 1
    hp.LR = learning_rate
    hp.WD = 0.0
    hp.noise = noise_level
    hp.interpolate = supervised_interpolation
    hp.ode_timesteps = 30

    # Model shape hyperparameters
    _, hp.data_dim = x_train.shape
    hp.hidden_dim = math.isqrt(hp.data_dim)
    hp.attend_dim = 16
    hp.num_heads = 4
    hp.depth = 2
    hp.ffn_dim_multiplier = 0.5

    # Specify model(s) for experiment
    if model_name == 'baseline-SLPMultShaped':
        model_constr = lambda args: models_baseline.SLPMultFilteredNormalized(hp.data_dim, hp.hidden_dim)
    elif model_name == 'cNODE1':
        model_constr = lambda args: models.cNODE1(hp.data_dim)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    model_args = {}

    # Loss and evaluation functions
    loss_fn = loss_bc
    score_fn = loss_bc
    distr_error_fn = distribution_error

    scaler = torch.cuda.amp.GradScaler()

    # Time steps
    ode_timemax = 1.0
    ode_stepsize = ode_timemax / hp.ode_timesteps
    timesteps = torch.arange(0.0, ode_timemax + 0.1 * ode_stepsize, ode_stepsize).to(device)

    # Train and evaluate the model
    train_and_evaluate_model(hp.LR, scaler, hp.accumulated_minibatches, x_train, y_train, x_test, y_test, hp.noise,
                             hp.interpolate, device, hp.min_epochs, hp.max_epochs, hp.minibatch_examples, model_constr,
                             model_args, model_name, timesteps, loss_fn, score_fn, distr_error_fn, hp.WD, verbosity=4)

if __name__ == "__main__":
    main()
