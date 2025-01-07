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


def eval_model(model, x, timesteps, ids):
    # evaluates models whether they require ODE timesteps or not
    requires_timesteps = getattr(model, 'USES_ODEINT', False)
    requires_condensed = getattr(model, 'USES_CONDENSED', False)
    
    if requires_timesteps and timesteps is not None:
        # Call models that require the timesteps argument
        if requires_condensed:
            y_steps = model(timesteps, x, ids)
        else:
            y_steps = model(timesteps, x)
    else:
        # Call models that do not require the timesteps argument
        if requires_condensed:
            y = model(x, ids)
        else:
            y = model(x)
        y_steps = y.unsqueeze(0)
    
    return y_steps


def loss_bc_dki(y_pred, y_true):
    return torch.sum(torch.abs(y_pred - y_true)) / torch.sum(
        torch.abs(y_pred + y_true))  # DKI repo implementation (incorrect)


def loss_bc(y_pred, y_true):  # Bray-Curtis Dissimilarity
    return torch.mean(torch.sum(torch.abs(y_pred - y_true), dim=-1) / torch.sum(torch.abs(y_pred) + torch.abs(y_true), dim=-1))


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


def ceildiv(a, b):
    return -(a // -b)


def validate_epoch(model, data_val, minibatch_examples, t, loss_fn, score_fn, distr_error_fn, device):
    model.eval()
    
    total_loss = 0.0
    total_score = 0.0
    total_distr_error = 0.0
    total_samples = data_val[0].size(0)
    minibatches = ceildiv(total_samples, minibatch_examples)
    current_index = 0  # Initialize current index to start of dataset
    
    for mb in range(minibatches):
        z, ids, current_index = data.get_batch(data_val, t, minibatch_examples, current_index, noise_level_x=0.0, noise_level_y=0.0, requires_timesteps=False)
        x, y = z[0], z[-1]
        mb_examples = x.size(0)
        
        with torch.no_grad():
            y_pred = eval_model(model, x, t, ids)[-1]
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


def train_epoch(model, data_train, minibatch_examples, accumulated_minibatches, noise, interpolate, optimizer, scheduler, scaler, t,
                outputs_per_epoch,
                prev_examples, fold, epoch_num, model_name, dataname, loss_fn, score_fn, distr_error_fn, device,
                filepath_out_incremental, lr_plot=None, loss_plot=None, lr_loss_plot=None, verbosity=1, supervised_timesteps=1):
    model.train()
    
    total_loss = 0
    total_penalty = 0
    total_samples = data_train[0].size(0)
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
        
        z, ids, current_index = data.get_batch(data_train, t, minibatch_examples, current_index, noise_level_x=noise, noise_level_y=noise, requires_timesteps=supervise_steps)  #
        mb_examples = z.shape[-2]
        
        if current_index >= total_samples:
            current_index = 0  # Reset index if end of dataset is reached
            data_train = data.shuffle_data(data_train)
        
        y_pred = eval_model(model, z[0], t, ids)
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
            print(f"GRADIENT ERROR: Loss at epoch {epoch_num} minibatch {mb} does not require gradient. Computation graph detached?")
        
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


def find_LR(model, model_name, scaler, x, y, x_valid, y_valid, minibatch_examples, accumulated_minibatches, noise, device,
            min_epochs, max_epochs, dataname, timesteps, loss_fn, score_fn, distr_error_fn, weight_decay, initial_lr,
            verbosity=1, seed=None, run_validation=False):  # Set the seed for reproducibility
    # TODO: Modify my approach. I should only use live-analysis to detect when to stop. Then return the complete results, which I will apply front-to-back analyses on to identify the points of significance (steepest point on cliff, bottom and top of cliff)
    
    # assert(data.check_leakage([(x, y, x_valid, y_valid)]))
    
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    total_samples = x.size(0)
    steps_per_epoch = ceildiv(total_samples, minibatch_examples * accumulated_minibatches)
    
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=weight_decay)
    # manager = epoch_managers.DivergenceManager(memory=0.95, threshold=0.025, mode="rel_start", min_epochs=min_epochs*steps_per_epoch, max_epochs=max_epochs*steps_per_epoch)
    manager = epoch_managers.ConvergenceManager(memory=0.95, threshold=10.0, mode="rel",
                                                min_epochs=min_epochs * steps_per_epoch,
                                                max_epochs=max_epochs * steps_per_epoch)
    base_scheduler = lr_schedule.ExponentialLR(optimizer, epoch_lr_factor=100.0,
                                               steps_per_epoch=2 * steps_per_epoch)  # multiplying by 2 as a cheap way to say I want the LR to increase by epoch_lr_factor after 4 actual epochs (the names here are misleading, the manager is actually managing minibatches and calling them epochs)
    scheduler = lr_schedule.LRScheduler(base_scheduler, initial_lr=initial_lr)
    
    # filepath_out_incremental = f'results/logs/{model_name}_{dataname}_LRRangeSearch.csv'
    old_lr = scheduler.get_last_lr()
    train_examples_seen = 0
    smoothed_loss_opt = Optimum('manager_metric', 'min')
    last_opt = Optimum(metric=None)
    
    total_samples = x.size(0)
    minibatches = ceildiv(total_samples, minibatch_examples)
    current_index = 0
    
    model.train()
    optimizer.zero_grad()
    
    stream_interval = 1  # minibatches
    
    total_loss = 0
    total_penalty = 0
    stream_loss = 0
    stream_penalty = 0
    stream_examples = 0
    
    done = False
    while not done:
        x_batch, y_batch, current_index = data.get_batch(x, y, t, minibatch_examples, current_index, noise_level_x=noise, noise_level_y=noise)
        if current_index >= total_samples:
            current_index = 0
            x, y = data.shuffle_data(x, y)
        
        y_pred = eval_model(model, x_batch, timesteps)[-1]
        y_pred = y_pred.to(device)
        loss = loss_fn(y_pred, y_batch) / accumulated_minibatches
        scaler.scale(loss).backward()
        
        distr_error = distr_error_fn(y_pred)
        total_loss += loss.item() * x_batch.size(0)
        total_penalty += distr_error.item() * x_batch.size(0)
        train_examples_seen += x_batch.size(0)
        
        if (manager.epoch + 1) % accumulated_minibatches == 0:
            scaler.step(optimizer)
            scaler.update()
            scheduler.batch_step()
            optimizer.zero_grad()
        
        stream_loss += total_loss
        stream_penalty += total_penalty
        stream_examples += x_batch.size(0)
        
        scheduler.epoch_step(loss)
        
        # quick and dirty way to get these in dictinoary format. I should refactor the above code so that the variables are always in a dictionary instead.
        dict = {"epoch": manager.epoch, "trn_loss": loss, "lr": old_lr, "manager_metric": manager.get_metric()}
        
        # TODO: Fix bug, manager.get_metric will be behind by 1 epoch, truncating final values
        smoothed_loss_opt.track_best(dict)
        last_opt.track_best(dict)
        
        old_lr = scheduler.get_last_lr()
        
        stop = manager.should_stop(last_opt)
        metric = manager.get_metric()
        if metric is not None:
            metric = metric.item()
        
        if manager.epoch % stream_interval == 0:
            # VALIDATION
            if run_validation:
                l_val, l_dki_val, p_val = validate_epoch(model, x_valid, y_valid, minibatch_examples, timesteps,
                                                         loss_fn, score_fn,
                                                         distr_error_fn, device)
                plotstream.plot(f"Raw LRRS for {model_name}", "log( Learning Rate )", "loss",
                                [f"{model_name}, wd:{weight_decay}, init_lr:{initial_lr} - Val",
                                 f"{model_name}, wd:{weight_decay}, init_lr:{initial_lr} - Trn"],
                                scheduler.get_last_lr(),
                                [l_val, loss.item()], add_point=False, x_log=True)
            else:
                plotstream.plot_single(f"Raw LRRS for {model_name}", "log( Learning Rate )", "Raw Loss",
                                       f"{model_name}, wd:{weight_decay}, init_lr:{initial_lr}",
                                       scheduler.get_last_lr(), loss.item(), add_point=False, x_log=True)
            
            # plotstream.plot_single("LRRS Loss vs Minibatches", "Minibatches", "Smoothed Loss", model_name,
            #                    manager.epoch, manager.get_metric().item(), add_point=False)
            # plotstream.plot_single(f"LRRS metric for {model_name}", "log( Learning Rate )", "Smoothed Loss", f"{model_name}, wd:{weight_decay}, init_lr:{initial_lr}",
            #                    scheduler.get_last_lr(), metric, add_point=False, x_log=True)
        
        stream_loss = 0
        stream_penalty = 0
        stream_examples = 0
        
        if stop:
            break
    
    print(f"Last LR: {last_opt.lr}")
    print(f"Best LR: {smoothed_loss_opt.lr}")
    
    # TODO: Track the SLOPE of the smoothed loss, and use a mix of that LR and the "best" LR. The minimum is occurring when it has stopped improving, so using a point after that is causing me to overestimate the point at which "slight divergence" occurs.
    # TODO: Draw the point that is chosen onto the LR Finder curve plot. I'll need to add a new method for that.
    # TODO: Tune the LR Finder / OneCycle params. I'm not really getting good performance right now. It might be too few epochs.
    # TODO: Most importantly, I need to do some fast hyperparameter searching on each model using the LR Finder as metric (in particular WD).
    # alpha = 0.9
    # diverge_lr = math.exp(alpha * math.log(smoothed_loss_opt.lr) + (1.0 - alpha) * math.log(last_opt.lr))
    diverge_lr = smoothed_loss_opt.lr  # * 0.5
    diverge_loss = smoothed_loss_opt.trn_loss
    diverge_metric = smoothed_loss_opt.manager_metric
    
    # TODO: if averaging along the logarithmic scale (to find halfway point between 1e-3 and 1e-2 for eample), do the geometric mean sqrt(a*b). We want to use this to find e.g. the point between two optima measurements. If we need to weight that geometric mean, it's exp(alpha * log(a) + (1-alpha) * log(b))
    
    # plotstream.plot_point(f"LRRS metric for {model_name}", f"{model_name}, wd:{weight_decay}", diverge_lr, diverge_metric.item(), symbol="*")
    plotstream.plot_point(f"Raw LRRS for {model_name}", f"{model_name}, wd:{weight_decay}", diverge_lr,
                          diverge_loss.item(), symbol="*")
    
    print(f"Peak LR: {diverge_lr}")
    return diverge_lr


# Search for hyperparameters by identifying the values that lead to the highest learning rate before divergence
# Based on notes found here: https://sgugger.github.io/the-1cycle-policy.html
def hyperparameter_search_with_LRfinder(model_constr, model_args, model_name, scaler, data_folded, minibatch_examples,
                                        accumulated_minibatches,
                                        device, min_epochs, max_epochs, dataname, timesteps, loss_fn, score_fn,
                                        distr_error_fn,
                                        threshold_proportion=0.9, verbosity=1, seed=None, run_validation=False):
    x, y, x_valid, y_valid = data_folded[0]
    
    # weight_decay_values = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0] # SLPReplicator
    # weight_decay_values = [1.0, 3.3, 10] #[1e-1, 0.33, 1e0, 3.3, 1e1, 33] # cNODE2
    weight_decay_values = [0.0, 1e-3, 1e-2, 1e-1, 0.32, 1e0, 3.2]
    # weight_decay_values = [1e-1, 0.33, 1e0, 3.3, 1e1]
    initial_lr_values = [0.03]
    
    highest_lr = -np.inf
    lr_results = {}
    
    model_base = model_constr(model_args)
    
    # Perform the hyperparameter search
    for wd in weight_decay_values:
        for initial_lr in initial_lr_values:
            model = model_constr(model_args)
            model.load_state_dict(model_base.state_dict())
            
            diverge_lr = find_LR(model, model_name, scaler, x, y, x_valid, y_valid, minibatch_examples,
                                 accumulated_minibatches,
                                 device, min_epochs, max_epochs, dataname, timesteps, loss_fn, score_fn, distr_error_fn,
                                 wd, initial_lr,
                                 verbosity, seed=seed, run_validation=True)
            lr_results[wd] = diverge_lr
            plotstream.plot_single(f"WD vs divergence LR", "WD", "Divergence LR", model_name, wd, diverge_lr,
                                   False, y_log=True, x_log=True)
            
            if diverge_lr > highest_lr:
                highest_lr = diverge_lr
    
    # Determine the valid weight_decay values within the threshold proportion
    valid_weight_decays = [wd for wd, lr in lr_results.items() if lr >= threshold_proportion * highest_lr]
    
    # Select the largest valid weight_decay
    optimal_weight_decay = max(valid_weight_decays)
    optimal_lr = lr_results[optimal_weight_decay]
    
    plotstream.plot_point(f"WD vs divergence LR", model_name, optimal_weight_decay, optimal_lr, symbol="*")
    
    return optimal_weight_decay, optimal_lr


def run_epochs(model, optimizer, scheduler, manager, minibatch_examples, accumulated_minibatches, noise, interpolate, scaler, data_train, data_valid, t,
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
    l_val, score_val, p_val = validate_epoch(model, data_valid, minibatch_examples, t, loss_fn, score_fn,
                                             distr_error_fn, device)
    l_trn, score_trn, p_trn = validate_epoch(model, data_train, minibatch_examples, t, loss_fn, score_fn,
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
                          "Peak RAM (GB)", cpuRam  / (1024 ** 3),
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
        l_trn, p_trn, train_examples_seen = train_epoch(model, data_train, minibatch_examples,
                                                        accumulated_minibatches, noise, interpolate, optimizer, scheduler, scaler, t,
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
        
        l_val, score_val, p_val = validate_epoch(model, data_valid, minibatch_examples, t, loss_fn, score_fn,
                                                 distr_error_fn, device)
        if reeval_train:
            l_trn, score_trn, p_trn = validate_epoch(model, data_train, minibatch_examples, t, loss_fn, score_fn,
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
        
        training_curve.append({"fold": fold, "epoch": manager.epoch, "trn_loss": l_trn, "val_loss": l_val, "time": elapsed_time})
        
        old_lr = new_lr
        
        # check if we should continue
        if manager.should_stop(last_opt):
            break
    
    # TODO: Check if this is the best model of a given name, and if so, save the weights and logs to a separate folder for that model name
    # TODO: could also try to save the source code, but would need to copy it at time of execution and then rename it if it gets the best score.
    
    return val_opt, valscore_opt, trn_opt, trnscore_opt, last_opt, training_curve


def crossvalidate_model(LR, scaler, accumulated_minibatches, data_folded, noise, interpolate, device, early_stop, patience, kfolds,
                        min_epochs, max_epochs,
                        minibatch_examples, model_constr, model_args, model_name, dataname, timesteps, loss_fn,
                        score_fn, distr_error_fn, weight_decay, verbosity=1, reptile_rewind=0.0, reeval_train=False,
                        whichfold=-1, jobstring=""):
    filepath_out_fold = f'results/folds/{model_name}_{dataname}{jobstring}_folds.csv'
    
    # LR_start_factor = 0.1 # OneCycle
    LR_start_factor = 1.0  # everything else
    
    val_loss_optims = []
    val_score_optims = []
    trn_loss_optims = []
    trn_score_optims = []
    final_optims = []
    val_loss_curves = []
    for fold_num, data_fold in enumerate(data_folded):
        if whichfold >= 0:
            fold_num = whichfold
        model = model_constr(model_args).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR * LR_start_factor, weight_decay=weight_decay)
        # optimizer = torch.optim.SGD(model.parameters(), lr=LR*LR_start_factor, weight_decay=weight_decay)
        # manager = epoch_managers.FixedManager(max_epochs=min_epochs)
        manager = epoch_managers.ExplosionManager(memory=0.5, threshold=1.0, mode="rel", max_epochs=max_epochs)
        # manager = epoch_managers.ConvergenceManager(memory=0.1, threshold=0.001, mode="const", min_epochs=min_epochs, max_epochs=max_epochs)
        
        # x_train, y_train, x_valid, y_valid = data_fold
        requires_condensed = getattr(model, 'USES_CONDENSED', False)
        if not requires_condensed:
            data_train = [data_fold[0][0], data_fold[1][0]]
            data_valid = [data_fold[0][1], data_fold[1][1]]
        else:
            data_train = [data_fold[2][0], data_fold[3][0], data_fold[4][0]]
            data_valid = [data_fold[2][1], data_fold[3][1], data_fold[4][1]]

        
        steps_per_epoch = ceildiv(data_train[0].size(0), minibatch_examples * accumulated_minibatches)
        base_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=patience // 2, cooldown=patience,
                                           threshold_mode='rel', threshold=0.01)
        # base_scheduler = OneCycleLR(
        #     optimizer, max_lr=LR, epochs=min_epochs, steps_per_epoch=steps_per_epoch, div_factor=1.0/LR_start_factor,
        #     final_div_factor=1.0/(LR_start_factor*0.1), three_phase=True, pct_start=0.4, anneal_strategy='cos')
        scheduler = lr_schedule.LRScheduler(base_scheduler, initial_lr=LR * LR_start_factor)
        
        print(f"Fold {fold_num + 1}/{kfolds}")
        
        val_opt, valscore_opt, trn_opt, trnscore_opt, last_opt, training_curve = run_epochs(model, optimizer, scheduler, manager,
                                                                            minibatch_examples, accumulated_minibatches, noise, interpolate,
                                                                            scaler, data_train, data_valid, 
                                                                            timesteps, model_name,
                                                                            dataname, fold_num, loss_fn, score_fn,
                                                                            distr_error_fn, device,
                                                                            outputs_per_epoch=10,
                                                                            verbosity=verbosity - 1,
                                                                            reptile_rate=reptile_rewind,
                                                                            reeval_train=reeval_train,
                                                                            jobstring=jobstring)
        

        # Below is temporarily commented out - fix it for the new dataset formats
        # Print output of model on a batch of test examples
        # DEBUG_OUTPUT = False  # TO DO: make this an actual parameter
        # DEBUG_OUT_NUM = 4
        
        # if DEBUG_OUTPUT:
        #     DEBUG_OUT_CSV = f"./analysis/debug_outputs/{model_name}_{dataname}{jobstring}_predictions.csv"
            
        #     model.eval()
        #     with torch.no_grad():
        #         # Get the model output for the first minibatch
        #         debug_ys = eval_model(model, x_valid[:DEBUG_OUT_NUM].to(device), timesteps)
                
        #         # Get the corresponding y_valid batch
        #         y_valid_batch = y_valid[0:DEBUG_OUT_NUM].to(device)
                
        #         print(f"Example output of model {model_name} on first test batch")
                
        #         csv_data = []
        #         # Iterate row by row
        #         for i in range(DEBUG_OUT_NUM):
        #             y_valid_row = y_valid_batch[i].cpu().numpy()  # Move to CPU and convert to numpy for easy processing
                    
                    
        #             # For console printing
        #             print(f"Row {i}:")
        #             print("time: -1, y_valid:")
        #             print(y_valid_row)
                    
        #             # Iterate through each timestep to append debug_ys for the current batch position
        #             for t_idx, debug_y_row in enumerate(debug_ys[:, i].cpu().numpy()):
        #                 timestep = timesteps[t_idx]
                        
        #                 # Add debug_ys row with the current timestep
        #                 csv_data.append(
        #                     [timestep.item()] + list(debug_y_row))  # Add debug_y for this batch at current timestep
                        
        #                 # For console printing
        #                 print(f"time: {timestep.item()}, debug_y for batch {i}:")
        #                 print(debug_y_row)

        #             # Add correct answer at the end with time = -1
        #             csv_data.append([-1] + list(y_valid_row))  # Add y_valid for this batch
                    
        #             print('-' * 50)  # Separator between batch items
                
        #         # Create a DataFrame for exporting to CSV
        #         df = pd.DataFrame(csv_data)
            
        #     # Export to CSV
        #     df.to_csv(DEBUG_OUT_CSV, index=False, header=False)
        
        val_loss_optims.append(val_opt)
        val_score_optims.append(valscore_opt)
        trn_loss_optims.append(trn_opt)
        trn_score_optims.append(trnscore_opt)
        final_optims.append(last_opt)
        
        # To Do: refactor this, we don't need all these variables
        val_loss = val_opt.val_loss
        val_score = val_opt.val_score
        val_trn_loss = val_opt.trn_loss
        val_epoch = val_opt.epoch
        val_time = val_opt.time
        trn_loss = trn_opt.trn_loss
        trn_score = trn_opt.trn_score
        trn_val_loss = trn_opt.val_loss
        trn_epoch = trn_opt.epoch
        trn_time = trn_opt.time
        
        stream.stream_results(filepath_out_fold, verbosity > 0, verbosity > 0, verbosity > -1,
                              "fold", fold_num + 1,
                              "Validation Loss", val_loss,
                              "Validation Score", val_score,
                              "Val @ epoch0s", val_epoch + 1,
                              "Val @ time", val_time,
                              "Val @ training loss", val_trn_loss,
                              "Training Loss", trn_loss,
                              "Training Score", trn_score,
                              "Trn @ epochs", trn_epoch + 1,
                              "Trn @ time", trn_time,
                              "Trn @ validation loss", trn_val_loss,
                              prefix="\n========================================FOLD=========================================\n",
                              suffix="\n=====================================================================================\n")
    
    val_loss_curves.append(training_curve)
    
    return val_loss_optims, val_score_optims, trn_loss_optims, trn_score_optims, final_optims, val_loss_curves


def unrolldict(d):
    unrolled_items = list(itertools.chain(*(d.items())))
    return unrolled_items


def main():
    # device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    
    # Data
    
    # dataname = "P"
    # dataname = "waimea"
    dataname = "waimea-std"
    # dataname = "waimea-condensed"
    # dataname = "cNODE-paper-ocean"
    # dataname = "cNODE-paper-ocean-std"
    # dataname = "cNODE-paper-human-gut"
    # dataname = "cNODE-paper-human-oral"
    # dataname = "cNODE-paper-drosophila"
    # dataname = "cNODE-paper-soil-vitro"
    # dataname = "cNODE-paper-soil-vivo"
    # dataname = "dki-synth"
    # dataname = "dki-real"
    
    # data folding params
    kfolds = 5  # -1 for leave-one-out, > 1 for k-folds
    
    # whichfold = -1
    whichfold = 0
    
    jobid = "-1"  # -1 for no job id
    
    # experiment hyperparameters
    hp = dicy()
    
    hp.modelnames = "cNODE-hourglass"
    
    hp.solver = os.getenv("SOLVER")
    
    hp.min_epochs = 2
    hp.max_epochs = 1
    hp.patience = 1100
    hp.early_stop = True
    
    hp.ode_timesteps = 15
    
    # optimization hyperparameters
    hp.minibatch_examples = 20
    hp.accumulated_minibatches = 1
    hp.LR = 0.1
    hp.reptile_lr = 1.0
    hp.WD_factor = 0.0
    hp.noise = 0.075
    hp.interpolate = True
    
    # model shape hyperparameters
    hp.hidden_dim = 8
    hp.attend_dim_per_head = 4
    hp.num_heads = 2
    hp.depth = 6
    hp.ffn_dim_multiplier = 4.0
    hp.dropout = 0.5
    
    # command-line arguments
    parser = argparse.ArgumentParser(description='run')
    parser.add_argument('--models', default=hp.modelnames, help='models to run, comma-separated list')
    parser.add_argument('--dataname', default=dataname, help='dataset name')
    parser.add_argument('--kfolds', default=kfolds, help='how many data folds, -1 for leave-one-out')
    parser.add_argument('--whichfold', default=whichfold, help='which fold to run, -1 for all')
    parser.add_argument("--jobid", default=jobid, help="job id")
    parser.add_argument("--headless", action="store_true", help="run without plotting")
    parser.add_argument('--epochs', default=hp.max_epochs, help='maximum number of epochs')
    parser.add_argument('--mb', default=hp.minibatch_examples, help='minibatch size')
    parser.add_argument('--lr', default=hp.LR, help='learning rate')
    parser.add_argument('--reptile_rate', default=hp.reptile_lr, help='reptile outer-loop learning rate')
    parser.add_argument('--wd_factor', default=hp.WD_factor, help='weight decay factor (multiple of LR)')
    parser.add_argument('--noise', default=hp.noise, help='noise level')
    parser.add_argument('--ode_steps', default=hp.ode_timesteps, help='number of ODE timesteps')
    parser.add_argument('--interpolate', default=int(hp.interpolate), help='whether or not to use supervised interpolation steps, 0 or 1')
    parser.add_argument('--num_heads', default=hp.num_heads, help='number of attention heads')
    parser.add_argument('--hidden_dim', default=hp.hidden_dim, help='hidden dimension')
    parser.add_argument('--attend_dim_per_head', default=hp.attend_dim_per_head, help='attention dimension')
    parser.add_argument('--depth', default=hp.depth, help='depth of model')
    parser.add_argument('--ffn_dim_multiplier', default=hp.ffn_dim_multiplier, help='multiplier for ffn dimension')
    parser.add_argument('--dropout', default=hp.dropout, help='dropout rate')
    
    
    args = parser.parse_args()
    hp.modelnames = args.models
    dataname = args.dataname
    whichfold = int(args.whichfold)
    kfolds = int(args.kfolds)
    hp.max_epochs = int(args.epochs)
    hp.minibatch_examples = int(args.mb)
    hp.LR = float(args.lr)
    hp.reptile_lr = float(args.reptile_rate)
    hp.WD_factor = float(args.wd_factor)
    hp.noise = float(args.noise)
    hp.ode_timesteps = int(args.ode_steps)
    hp.interpolate = bool(int(args.interpolate))
    hp.num_heads = int(args.num_heads)
    hp.hidden_dim = int(args.hidden_dim)
    hp.attend_dim_per_head = int(args.attend_dim_per_head)
    hp.depth = int(args.depth)
    hp.ffn_dim_multiplier = float(args.ffn_dim_multiplier)
    hp.dropout = float(args.dropout)
    
    
    # load data
    filepath_train = f'data/{dataname}_train.csv'
    filepath_train_pos = f'data/{dataname}_train-pos.csv'
    filepath_train_val = f'data/{dataname}_train-val.csv'
    x, y, xcon, ycon, idcon = data.load_data(filepath_train, filepath_train_pos, filepath_train_val, device)
    data_folded = data.fold_data([x, y, xcon, ycon, idcon], kfolds)  # shape is (kfolds, datasets (x,y,xcon,...), train vs valid, n, d)
    data_folded = [data_folded[whichfold]] if whichfold >= 0 else data_folded  # only run a single fold based on args
    assert (data.check_leakage(data_folded))
    
    print('dataset:', filepath_train)
    print(f'data shape: {x.shape}')
    print(f'training data shape: {data_folded[0][0][0].shape}')
    print(f'validation data shape: {data_folded[0][0][1].shape}')
    print(f'condensed training shape: {data_folded[0][2][0].shape}')
    print(f'condensed validation shape: {data_folded[0][2][1].shape}')
    
    
    # computed hyperparams
    _, hp.data_dim = x.shape
    hp.WD = hp.LR * hp.WD_factor
    hp.attend_dim = hp.attend_dim_per_head * hp.num_heads
    
    jobid = int(args.jobid.split('_')[0])
    jobstring = f"_job{jobid}" if jobid >= 0 else ""
    
    if args.headless:
        plotstream.set_headless()

    assert hp.attend_dim % hp.num_heads == 0, "attend_dim must be divisible by num_heads"
    
    
    reeval_train = False # hp.interpolate or hp.noise > 0.0 # This isn't a general rule, you might want to do this for other reasons. But it's an easy way to make sure training loss curves are readable.
    
    # Specify model(s) for experiment
    # Note that each must be a constructor function that takes a dictionary args. Lamda is recommended.
    models = {
        # most useful models
        'baseline-constShaped': lambda args: models_baseline.ConstOutputFilteredNormalized(args.data_dim),
        'baseline-SLPMultShaped': lambda args: models_baseline.SLPMultFilteredNormalized(args.data_dim, args.hidden_dim),
        'cNODE1': lambda args: models_cnode.cNODE1(args.data_dim),
        'cNODE2': lambda args: models_cnode.cNODE2(args.data_dim),
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
            ffn_dim_multiplier=args.ffn_dim_multiplier, fitness_qk_dim=args.attend_dim, dropout=args.dropout
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
        'baseline-cNODE1-1step': lambda args: models_baseline.cNODE1_singlestep(args.data_dim),
        'baseline-cAttend-1step': lambda args: models_embedded.cAttend_simple(args.data_dim, args.attend_dim, args.attend_dim),
        'baseline-SLP-ODE': lambda args: models_baseline.SLPODE(args.data_dim, args.hidden_dim),
        'baseline-cNODE2-width1': lambda args: models_cnode.cNODE_HourglassFitness(
            data_dim=args.data_dim, hidden_dim=1, depth=3
        ),
        'baseline-cNODE2-width2': lambda args: models_cnode.cNODE_HourglassFitness(
            data_dim=args.data_dim, hidden_dim=2, depth=3
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
        'cNODE1-GenFn': lambda args: models_cnode.cNODE2_ExternalFitnessFn(args.data_dim), # for testing, identical to cNODE1
        'cNODE2_DKI': lambda args: models_cnode.cNODE2_DKI(args.data_dim), # sanity test, this is the same as cNODE2 but less optimized
        'cNODE2-Gen': lambda args: models_cnode.cNODEGen_ConstructedFitness(lambda: nn.Sequential(nn.Linear(args.data_dim, args.data_dim), nn.Linear(args.data_dim, args.data_dim))),  # sanity test, this is the same as cNODE2 but generated at runtime
        "cNODE2-static": lambda args: models_cnode.cNODE2_ExternalFitness(args.data_dim), # sanity test
        "cNODE2-FnFitness": lambda args: models_cnode.cNODE2_FnFitness(args.data_dim), # sanity test, this is the same as cNODE2 but testing externally-supplied fitness functions
    }
    
    modelnames_to_run = hp.modelnames.split(',')
    models_to_run = {name: models[name] for name in modelnames_to_run}
    
    # specify loss function
    loss_fn = loss_bc
    # avg_richness = x.count_nonzero()/x.size(0)
    # loss_fn = lambda y_pred, y_true: loss_bc_unbounded(y_pred, y_true, avg_richness)
    score_fn = loss_bc
    # loss_fn = lambda y_pred,y_true: loss_bc(y_pred, y_true) + distribution_error(y_pred)
    
    distr_error_fn = distribution_error
    
    # scaler = torch.amp.GradScaler(device)
    scaler = torch.cuda.amp.GradScaler()
    
    # time step "data"
    ode_timemax = 1.0
    ode_stepsize = ode_timemax / hp.ode_timesteps
    timesteps = torch.arange(0.0, ode_timemax + 0.1*ode_stepsize, ode_stepsize).to(device)
    
    # TODO: Experiment dictionary. Model, data set, hyperparam override(s).
    # Model dictionary. Hyperparam override(s)
    
    # Establish baseline performance and add to plots
    trivial_model = models_baseline.ReturnInput()
    trivial_loss, trivial_score, trivial_distro_error = validate_epoch(trivial_model, [x, y], hp.minibatch_examples,
                                                                       timesteps, loss_fn, score_fn, distr_error_fn,
                                                                       device)
    print(f"\n\nTRIVIAL MODEL loss: {trivial_loss}, score: {trivial_score}\n\n")
    plotstream.plot_horizontal_line(f"loss {dataname}", trivial_loss, f"Trivial (in=out)")
    plotstream.plot_horizontal_line(f"score {dataname}", trivial_score, f"Trivial (in=out)")
    # trivial_model2 = models_baseline.ReturnZeros()
    # trivial_loss2 = validate_epoch(trivial_model2, x, y, hp.minibatch_examples, timesteps, loss_fn, score_fn, distr_error_fn, device)[0]
    # plotstream.plot_horizontal_line(dataname, trivial_loss2, f"Trivial (zeros)")
    
    filepath_out_expt = f'results/expt/{dataname}{jobstring}_experiments.csv'
    seed = int(time.time())  # currently only used to set the data shuffle seed in find_LR
    print(f"Seed: {seed}")
    for model_name, model_constr in models_to_run.items():
        
        # # TODO remove this: it's just to resume from where we were previously
        # if ((attend_dim == 4 or attend_dim == 16) and model_name == 'canODE-transformer-d6' and num_heads == 4):
        #     continue
        
        # try:
        print(f"\nRunning model: {model_name}")
        
        # test construction and print parameter count
        model = model_constr(hp)
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Number of parameters in model: {num_params}")
        
        # find optimal LR
        # hp.WD, hp.LR = hyperparameter_search_with_LRfinder(
        #     model_constr, hp, model_name, scaler, data_folded, hp.minibatch_examples, hp.accumulated_minibatches,
        #     device, 3, 3, dataname, timesteps, loss_fn, score_fn, distr_error_fn, verbosity=1, seed=seed)
        # print(f"LR:{hp.LR}, WD:{hp.WD}")
        
        # hp = ui.ask(hp, keys=["LR", "WD"])
        
        # print hyperparams
        for key, value in hp.items():
            print(f"{key}: {value}")
        
        # Just for the sake of logging experiments before cross validation...
        optdict = {"epoch": -1, "trn_loss": -1.0, "trn_score": -1.0, "val_loss": -1.0,
                "val_score": -1.0, "lr": -1.0, "time": -1.0, "gpu_memory": -1.0,
                "metric": -1.0}
        val_loss_optims = [Optimum('val_loss', 'min', dict=optdict)]
        trn_loss_optims = [Optimum('trn_loss', 'min', dict=optdict)]
        val_score_optims = [Optimum('val_score', 'min', dict=optdict)]
        trn_score_optims = [Optimum('trn_score', 'min', dict=optdict)]
        stream.stream_scores(filepath_out_expt, True, True, True,
                             "model", model_name,
                             "dataset", dataname,
                             "mean_val_loss", -1,
                             "mean_val_loss @ epoch", -1,
                             "mean_val_loss @ time", -1,
                             "mean_val_loss @ trn_loss", -1,
                             "model parameters", num_params,
                             "fold", -1,
                             "k-folds", kfolds,
                             "device", device,
                             *unrolldict(hp),  # unroll the hyperparams dictionary
                             *unrolloptims(val_loss_optims[0], val_score_optims[0], trn_loss_optims[0],
                                           trn_score_optims[0]),
                             prefix="\n=======================================================EXPERIMENT========================================================\n",
                             suffix="\n=========================================================================================================================\n")
        
        
        # train and test the model across multiple folds
        val_loss_optims, val_score_optims, trn_loss_optims, trn_score_optims, final_optims, training_curves = crossvalidate_model(
            hp.LR, scaler, hp.accumulated_minibatches, data_folded, hp.noise, hp.interpolate, device, hp.early_stop, hp.patience,
            kfolds, hp.min_epochs, hp.max_epochs, hp.minibatch_examples, model_constr, hp,
            model_name, dataname, timesteps, loss_fn, score_fn, distr_error_fn, hp.WD, verbosity=1,
            reptile_rewind=(1.0 - hp.reptile_lr), reeval_train=reeval_train, whichfold=whichfold, jobstring=jobstring
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
                                 "model", model_name,
                                 "dataset", dataname,
                                 "mean_val_loss", best_epoch_metrics["val_loss"],
                                 "mean_val_loss @ epoch", best_epoch_metrics["epoch"],
                                 "mean_val_loss @ time", best_epoch_metrics["time"],
                                 "mean_val_loss @ trn_loss", best_epoch_metrics["trn_loss"],
                                 "model parameters", num_params,
                                 "fold", i if whichfold < 0 else whichfold,
                                 "k-folds", kfolds,
                                 "device", device,
                                 *unrolldict(hp),  # unroll the hyperparams dictionary
                                 *unrolloptims(val_loss_optims[i], val_score_optims[i], trn_loss_optims[i],
                                               trn_score_optims[i]),
                                 prefix="\n=======================================================EXPERIMENT========================================================\n",
                                 suffix="\n=========================================================================================================================\n")
        
        # except Exception as e:
        #     stream.stream_scores(filepath_out_expt, True, True, True,
        #         "model", model_name,
        #         "model parameters", -1,
        #         "Validation Score", -1,
        #         "Validation Loss", -1,
        #         "Val @ Epoch", -1,
        #         "Val @ Time", -1,
        #         "Val @ Trn Loss", -1,
        #         "Train Score", -1,
        #         "Train Loss", -1,
        #         "Trn @ Epoch", -1,
        #         "Trn @ Time", -1,
        #         "Trn @ Val Loss", -1,
        #         "k-folds", kfolds,
        #         "timesteps", ode_timesteps,
        #         *list(itertools.chain(*(hp.items()))), # unroll the hyperparams dictionary
        #         prefix="\n=======================================================EXPERIMENT========================================================\n",
        #         suffix="\n=========================================================================================================================\n")
        #     print(f"Model {model_name} failed with error:\n{e}")
    
    print("\n\nDONE")
    plotstream.wait_for_plot_exit()


# main
if __name__ == "__main__":
    main()

# TODO: Set seed for reproducible shuffling
# TODO: in particular we want to ensure that LR/WD search is using the same model+data on each trial, so we aren't just selecting which initialization was randomly more tolerant of high LR
# TODO: hyperparameter optimization
# TODO: time limit and/or time based early stopping
# TODO: hyperparameters optimization based on loss change rate against clock time, somehow
# TODO: Confirm that changes of t shape do not alter performance, then remove t argument to model forward for non-ODEfuncs. The t can be generated in forward to pass to odeint.
# TODO: realtime visualization
# TODO: Add param count to filename of logs
# TODO: Instead of my lambda model constructors, try base_model = torchdistx.deferred_init.deferred_init(MyModel, param1, param2, ...)
# TODO: Try muP for hyperparameter scaling: https://github.com/microsoft/mup

# TODO: Try transfer learning with shared ODE but separate embed/unembed - espcially x-shaped conjoined networks for joint learning. Or Reptile for similar metalearning.
# TODO: Create a parameterized generalized version of the canODE models so I can explore the model architectures as hyperparameters
# TODO: Attention layers in a DEQ (deep equilibrium model) similar to the nODE to produce F(x) for the ODE
# TODO: (Probably not a good idea) As an alternative to attention, try condensing into a high enough space that we can still use channels as IDs (minimum size would be the largest summed assemblage in the dataset) and then use a vanilla (but smaller) cNODE. The difficulty here is that the embed mapping needs to be dynamic because many input channels will map to the same embedded channels and some of those could occur at the same time. There effectively needs to be the ability to decide on the fly that "A should go in M but that's already occupied by B, so A will go in N instead, which has the same properties as M" ... which means the embedding needs to have redundancies. I guess I could hardcode there being a few redundant copies of each channel somehow (shared weights?) but I don't like that idea. In general this seems much weaker than using attention to divorce the species representations from the preferred basis, allowing them to share dynamics to exactly the extent that is helpful via independent subspaces.
# TODO: Test baseline ODE model that does not learn F(x) at all - either it's just torch.ones() or it's a torch.random() at initialization or it's a torch.random() in the forward call.
# TODO: Test multi-layer attention-based models that do not utilize ODEs at all.

# TODO: Try all of the small models with higher parameter counts. We don't need to use such small embedding dimension.

# cNODE-based models seem to get nearly unlimited ability to handle high LR the lower WD is (at least during LRRS tests), BUT the models become very stiff and therefore torchdiffeq solving is slow. Would it be better to tune our LR based on optimizing Loss vs Time, rather than Loss vs Samples?
# ...Turns out that there is a bit of divergence after a brief optimum, but then it plateaus and usually doesn't diverge further until absurdly high LR levels. But the exponential average was masking that behavior.

# TODO: I think LRRS arguments should specify the number of iterations as a hyperparameter instead of epochs, and compute epochs. Because the logic around processing and analyzing the results is very step-centric. We would get much higher end loss for low step sizes, but should still be able to identify optimal LR which is the real goal.

# TODO: Try data augmentation by removing features and renormalizing
# TODO: Try self-supervised task by masking out features (and not renormalizing) then predict the masked embedding (no need to unembed unless we have a real classification task). Allowing to learn how species interact by what is missing.
# ACTUALLY isn't that a more direct way of estimating keystoneness?

# TODO: Try using proportion as magnitude, multiplied against the species embedding, instead of concatenated or added.

# TODO: Settings for training and validating on less than a full batch. Like, max number of data samples per train/val "epoch". But I would need to make sure shuffling only occurs after full epoch. Useful for training on large datasets, but even more useful for LRRS. (The latter can probably switch entirely to a number of samples config instead of epoch-based config.)
#  Problem: K-Fold paradigm superficially seems to conflict with this. In particular the validation since we would want to use the same samples for each time step. In that case, what we really want is a "Hold X out" fold rather than a K-fold. And we can use a subsample of that for LRRS validation.
#  One option to reduce config complexity: when K is negative, it is the number of samples to hold out for validation. When K is positive, it is the number of folds. But I will have to compute the number of folds (last of which will usually be smaller) from the number of held out samples.