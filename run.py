import itertools
import math
import time

import numpy as np
import torch
import torch.nn as nn
from dotsy import dicy, ency
from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau

import data
import epoch_managers
import lr_schedule
import models
import models_baseline
import models_condensed
import stream
import user_confirmation as ui


# TODO: everywhere that I'm reporting loss, I should use the DKI implementation for comparison, even if that's not what I'm optimizing.
def loss_bc_dki(y_pred, y_true):
    return torch.sum(torch.abs(y_pred - y_true)) / torch.sum(torch.abs(y_pred + y_true))   # DKI implementation

def loss_bc(y_pred, y_true):  # Bray-Curtis Dissimilarity
    return torch.sum(torch.abs(y_pred - y_true)) / torch.sum(torch.abs(y_pred) + torch.abs(y_true))  # more robust implementation?
    # return torch.sum(torch.abs(y_pred - y_true)) / (2.0 * y_pred.shape[0])   # simplified by assuming every vector is a species composition that sums to 1 ... but some models may violate that constraint so maybe don't use this

def distribution_error(x):  # penalties for invalid distributions
    a = 1.0
    b = 1.0
    feature_penalty = torch.sum(torch.clamp(torch.abs(x - 0.5) - 0.5, min=0.0)) / x.shape[0]  # each feature penalized for distance from range [0,1]
    sum_penalty = torch.sum(torch.abs(torch.sum(x, dim=1) - 1.0)) / x.shape[0]  # sum penalized for distance from 1.0
    return a*feature_penalty + b*sum_penalty


def ceildiv(a, b):
    return -(a // -b)


# TODO: switch most of this functionality to dict. Optimum just needs to accept a key name and info about how to evaluate that key, then in each step it gets the dict as an argument. It just conditionally copies the dict's contents into its internal dict.
# Can the internal dict copy be empty during construction, and get filled during the first update call? That way we don't have to worry about default values.
# Can we make this an Ency for convenience? (and to finally test Ency)
class Optimum(ency):
    def __init__(self, metric, metric_type='min', dict=None):
        self.metric_name = metric
        self.metric_type = metric_type
        
        if dict is not None:
            self.dict = dict.copy()
        else:
            self.dict = {}

        if metric_type == 'min':
            self.best_metric = float('inf')
        elif metric_type == 'max':
            self.best_metric = -float('inf')
            
        super().__init__(["dict"])  # initialize data for ency dot-access

    def track_best(self, dict):
        if self.metric_name is None or self.metric_name not in self.dict:
            best = True
        else:
            current_metric = dict[self.metric_name]
            last_metric = self.dict[self.metric_name]
            if current_metric is None:
                best = False
            elif last_metric is None:
                best = True
            else:
                if self.metric_type == 'min':
                    best = current_metric < last_metric
                else:
                    best = current_metric > last_metric

        if best:
            self.dict = dict.copy()
            
        return best


def validate_epoch(model, x_val, y_val, minibatch_examples, t, loss_fn, distr_error_fn, device):
    model.eval()
    
    total_loss = 0.0
    total_dki_loss = 0.0
    total_distr_error = 0.0
    total_samples = x_val.size(0)
    minibatches = ceildiv(total_samples, minibatch_examples)
    current_index = 0  # Initialize current index to start of dataset

    for mb in range(minibatches):
        x, y, current_index = data.get_batch(x_val, y_val, minibatch_examples, current_index)
        mb_examples = x.size(0)

        with torch.no_grad():
            y_pred = model(t, x).to(device)

            loss = loss_fn(y_pred, y)
            dki_loss = loss_bc_dki(y_pred, y)
            distr_error = distr_error_fn(y_pred)
            total_loss += loss.item() * mb_examples  # Multiply loss by batch size
            total_dki_loss += dki_loss.item() * mb_examples
            total_distr_error += distr_error.item() * mb_examples

    avg_loss = total_loss / total_samples
    avg_dki_loss = total_dki_loss / total_samples
    avg_penalty = total_distr_error / total_samples
    return avg_loss, avg_dki_loss, avg_penalty


def train_epoch(model, x_train, y_train, minibatch_examples, accumulated_minibatches, optimizer, scheduler, scaler, t, outputs_per_epoch,
                prev_examples, fold, epoch_num, model_name, dataname, loss_fn, distr_error_fn, device, filepath_out_incremental, lr_plot=None, loss_plot=None, lr_loss_plot=None, verbosity=1):
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

        x, y, current_index = data.get_batch(x_train, y_train, minibatch_examples, current_index) #
        mb_examples = x.size(0)
        
        if current_index >= total_samples:
            current_index = 0  # Reset index if end of dataset is reached
            x, y = data.shuffle_data(x, y)
        
        y_pred = model(t, x).to(device)

        loss = loss_fn(y_pred, y)
        actual_loss = loss.item() * mb_examples
        loss = loss / accumulated_minibatches # Normalize the loss by the number of accumulated minibatches, since loss function can't normalize by this

        scaler.scale(loss).backward()

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
                "Examples per second", examples_per_second,
                "Learning Rate", scheduler.get_last_lr(),
                )
            if lr_plot:
                stream.plot_single(lr_plot, "epochs", "LR", f"{model_name} fold {fold}", epoch_num + mb/minibatches, scheduler.get_last_lr(), False, y_log=True)
            if loss_plot:
                stream.plot_loss(loss_plot, f"{model_name} fold {fold}", epoch_num + mb/minibatches, stream_loss / stream_examples, None, add_point=False)
            if lr_loss_plot:
                stream.plot_single(lr_loss_plot, "log( Learning Rate )", "Loss", f"{model_name} fold {fold}", scheduler.get_last_lr(), stream_loss / stream_examples, False, x_log=True)
            stream_loss = 0
            stream_penalty = 0
            prev_time = end_time
            stream_examples = 0

        if ((mb + 1) % accumulated_minibatches == 0) or (mb == minibatches - 1):
            scaler.step(optimizer)
            scaler.update()
            scheduler.batch_step() # TODO: Add accum_loss metric in case I ever want to do ReduceLROnPlateau with batch_step mode
            optimizer.zero_grad()

        #del x, y


    avg_loss = total_loss / total_samples
    avg_penalty = total_penalty / total_samples
    new_total_examples = prev_examples + new_examples
    return avg_loss, avg_penalty, new_total_examples


def find_LR(model, model_name, scaler, x, y, minibatch_examples, accumulated_minibatches, device, min_epochs, max_epochs, dataname, timesteps, loss_fn, distr_error_fn, weight_decay, verbosity=1, seed=None):    # Set the seed for reproducibility
    # TODO: Modify my approach. I should only use live-analysis to detect when to stop. Then return the complete results, which I will apply front-to-back analyses on to identify the points of significance (steepest point on cliff, bottom and top of cliff)
    
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    total_samples = x.size(0)
    steps_per_epoch = ceildiv(total_samples, minibatch_examples * accumulated_minibatches)
    
    initial_lr = 1e-3
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=weight_decay)
    # manager = epoch_managers.DivergenceManager(memory=0.95, threshold=0.025, mode="rel_start", min_epochs=min_epochs*steps_per_epoch, max_epochs=max_epochs*steps_per_epoch)
    manager = epoch_managers.ConvergenceManager(memory=0.95, threshold=10.0, mode="rel", min_epochs=min_epochs*steps_per_epoch, max_epochs=max_epochs*steps_per_epoch)
    base_scheduler = lr_schedule.ExponentialLR(optimizer, epoch_lr_factor=100.0, steps_per_epoch=4*steps_per_epoch) # multiplying by 4 as a cheap way to say I want the LR to increase by epoch_lr_factor after 4 actual epochs (the names here are misleading, the manager is actually managing minibatches and calling them epochs)
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
    
    stream_interval = 1#minibatches
    
    total_loss = 0
    total_penalty = 0
    stream_loss = 0
    stream_penalty = 0
    stream_examples = 0
    
    done = False
    while not done:
        x_batch, y_batch, current_index = data.get_batch(x, y, minibatch_examples, current_index)
        if current_index >= total_samples:
            current_index = 0
            x, y = data.shuffle_data(x, y)
        
        y_pred = model(timesteps, x_batch).to(device)
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
            # stream.plot_single("LRRS Loss vs Minibatches", "Minibatches", "Smoothed Loss", model_name,
            #                    manager.epoch, manager.get_metric().item(), add_point=False)
            stream.plot_single(f"LRRS for {model_name}", "log( Learning Rate )", "Smoothed Loss", f"{model_name}, wd:{weight_decay}",
                               scheduler.get_last_lr(), metric, add_point=False, x_log=True)
            stream.plot_single(f"Raw LRRS for {model_name}", "log( Learning Rate )", "Raw Loss", f"{model_name}, wd:{weight_decay}",
                               scheduler.get_last_lr(), loss.item(), add_point=False, x_log=True)
        
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
    diverge_lr = smoothed_loss_opt.lr # * 0.5
    diverge_loss = smoothed_loss_opt.trn_loss
    diverge_metric = smoothed_loss_opt.manager_metric
    
    # TODO: if averaging along the logarithmic scale (to find halfway point between 1e-3 and 1e-2 for eample), do the geometric mean sqrt(a*b). We want to use this to find e.g. the point between two optima measurements. If we need to weight that geometric mean, it's exp(alpha * log(a) + (1-alpha) * log(b))
    
    stream.plot_point(f"LRRS for {model_name}", f"{model_name}, wd:{weight_decay}", diverge_lr, diverge_metric.item(), symbol="*")
    stream.plot_point(f"Raw LRRS for {model_name}", f"{model_name}, wd:{weight_decay}", diverge_lr, diverge_loss.item(), symbol="*")
    
    print(f"Peak LR: {diverge_lr}")
    return diverge_lr


# Search for hyperparameters by identifying the values that lead to the highest learning rate before divergence
# Based on notes found here: https://sgugger.github.io/the-1cycle-policy.html
def hyperparameter_search_with_LRfinder(model_constr, model_args, model_name, scaler, x, y, minibatch_examples, accumulated_minibatches,
                                       device, min_epochs, max_epochs, dataname, timesteps, loss_fn, distr_error_fn,
                                       threshold_proportion=0.9, verbosity=1, seed=None):
    # weight_decay_values = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0] # SLPReplicator
    # weight_decay_values = [1.0, 3.3, 10] #[1e-1, 0.33, 1e0, 3.3, 1e1, 33] # cNODE2
    weight_decay_values = [1e-2, 1e-1, 0.33, 1e0, 3.3]
    
    highest_lr = -np.inf
    lr_results = {}
    
    model_base = model_constr(model_args)

    # Perform the hyperparameter search
    for wd in weight_decay_values:
        model = model_constr(model_args)
        model.load_state_dict(model_base.state_dict())
        
        diverge_lr = find_LR(model, model_name, scaler, x, y, minibatch_examples, accumulated_minibatches,
                             device, min_epochs, max_epochs, dataname, timesteps, loss_fn, distr_error_fn, wd, verbosity, seed=seed)
        lr_results[wd] = diverge_lr
        stream.plot_single(f"WD vs divergence LR", "WD", "Divergence LR", model_name, wd, diverge_lr,
                           False, y_log=True, x_log=True)
        
        if diverge_lr > highest_lr:
            highest_lr = diverge_lr

    # Determine the valid weight_decay values within the threshold proportion
    valid_weight_decays = [wd for wd, lr in lr_results.items() if lr >= threshold_proportion * highest_lr]

    # Select the largest valid weight_decay
    optimal_weight_decay = max(valid_weight_decays)
    optimal_lr = lr_results[optimal_weight_decay]
    
    stream.plot_point(f"WD vs divergence LR", model_name, optimal_weight_decay, optimal_lr, symbol="*")

    return optimal_weight_decay, optimal_lr


def run_epochs(model, optimizer, scheduler, manager, minibatch_examples, accumulated_minibatches, scaler, x_train, y_train, x_valid, y_valid, t,
               model_name, dataname, fold, loss_fn, distr_error_fn, device, outputs_per_epoch=10, verbosity=1):
    # track stats at various definitions of the "best" epoch
    val_opt = Optimum('val_loss', 'min')
    trn_opt = Optimum('trn_loss', 'min')
    last_opt = Optimum(metric=None) # metric None to update it every time. metric="epoch" would do the same
    
    old_lr = scheduler.get_last_lr()
    
    filepath_out_epoch = f'results/logs/{model_name}_{dataname}_epochs.csv'
    # filepath_out_model = f'results/logs/{model_name}_{dataname}_model.pth'
    filepath_out_incremental = f'results/logs/{model_name}_{dataname}_incremental.csv'
    
    # initial validation benchmark
    l_val, l_dki_val, p_val = validate_epoch(model, x_valid, y_valid, minibatch_examples, t, loss_fn, distr_error_fn, device)
    stream.stream_results(filepath_out_epoch, verbosity > 0,
        "fold", fold + 1,
        "epoch", 0,
        "training examples", 0,
        "Avg Training Loss", -1.0,
        "Avg DKI Trn Loss", -1.0,
        "Avg Training Distr Error", -1.0,
        "Avg Validation Loss", l_val,
        "Avg DKI Val Loss", l_dki_val,
        "Avg Validation Distr Error", p_val,
        "Learning Rate", old_lr,
        "Elapsed Time", 0.0,
        "GPU Footprint (MB)", -1.0,
        prefix="================PRE-VALIDATION===============\n",
        suffix="\n=============================================\n")
    stream.plot_loss(dataname, f"{model_name} fold {fold}", 0, None, l_val, add_point=False)
    # stream.plot(dataname, "epoch", "loss", [f"{model_name} fold {fold} - Val", f"{model_name} fold {fold} - Trn", f"{model_name} fold {fold} - DKI Val", f"{model_name} fold {fold} - DKI Trn"], 0, [l_val, None, l_dki_val, None], add_point=False)
    
    train_examples_seen = 0
    start_time = time.time()
    
    while True:
        l_trn, p_trn, train_examples_seen = train_epoch(model, x_train, y_train, minibatch_examples,
            accumulated_minibatches, optimizer, scheduler, scaler, t, outputs_per_epoch, train_examples_seen,
            fold, manager.epoch, model_name, dataname, loss_fn, distr_error_fn, device, filepath_out_incremental, lr_plot="Learning Rate", verbosity=verbosity - 1)
        l_val, l_dki_val, p_val = validate_epoch(model, x_valid, y_valid, minibatch_examples, t, loss_fn, distr_error_fn, device)
        l_trn, l_dki_trn, p_trn = validate_epoch(model, x_train, y_train, minibatch_examples, t, loss_fn, distr_error_fn, device)
        
        # Update learning rate based on validation loss
        scheduler.epoch_step(l_trn)
        new_lr = scheduler.get_last_lr()
        lr_changed = not np.isclose(new_lr, old_lr)
        add_point = lr_changed and isinstance(scheduler.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
        
        current_time = time.time()
        elapsed_time = current_time - start_time
        gpu_memory_reserved = torch.cuda.memory_reserved(device)
        
        stream.stream_results(filepath_out_epoch, verbosity > 0,
            "fold", fold + 1,
            "epoch", manager.epoch + 1,
            "training examples", train_examples_seen,
            "Avg Training Loss", l_trn,
            "Avg DKI Trn Loss", -1.0,
            "Avg Training Distr Error", p_trn,
            "Avg Validation Loss", l_val,
            "Avg DKI Val Loss", l_dki_val,
            "Avg Validation Distr Error", p_val,
            "Learning Rate", old_lr, # should I track average LR in the epoch? Max and min LR?
            "Elapsed Time", elapsed_time,
            "GPU Footprint (MB)", gpu_memory_reserved / (1024 ** 2),
            prefix="==================VALIDATION=================\n",
            suffix="\n=============================================\n")
        stream.plot_loss(dataname, f"{model_name} fold {fold}", manager.epoch + 1, l_trn, l_val, add_point=add_point)
        # stream.plot(dataname, "epoch", "loss", [f"{model_name} fold {fold} - Val", f"{model_name} fold {fold} - Trn", f"{model_name} fold {fold} - DKI Val", f"{model_name} fold {fold} - DKI Trn"], manager.epoch + 1, [l_val, l_trn, l_dki_val, l_dki_trn], add_point=add_point)
        if l_val != l_dki_val:
            print("WARNING: CURRENT LOSS METRIC DISAGREES WITH DKI LOSS METRIC")
        
        # TODO: replace this quick and dirty dict packing. They should have always been in a dict.
        dict = {"epoch": manager.epoch, "trn_loss": l_trn, "val_loss": l_val, "lr": old_lr, "time": elapsed_time, "gpu_memory": gpu_memory_reserved, "metric": p_val}
        # track best validation
        val_opt.track_best(dict)
        # if val_opt.track_best(manager.epoch, l_trn, l_val, old_lr, elapsed_time):
        #     torch.save(model.state_dict(), filepath_out_model)

        # track best training
        trn_opt.track_best(dict)
        
        # track newest
        last_opt.track_best(dict)
        
        old_lr = new_lr
        
        # check if we should continue
        if manager.should_stop(last_opt):
            break
    
    # TODO: Check if this is the best model of a given name, and if so, save the weights and logs to a separate folder for that model name
    # TODO: could also try to save the source code, but would need to copy it at time of execution and then rename it if it gets the best score.
    
    return val_opt, trn_opt, last_opt
    


def crossvalidate_model(LR, scaler, accumulated_minibatches, data_folded, device, early_stop, patience, kfolds, min_epochs, max_epochs,
                        minibatch_examples, model_constr, model_args, model_name, dataname, timesteps, loss_fn, distr_error_fn, weight_decay, verbosity=1):
    
    filepath_out_fold = f'results/logs/{model_name}_{dataname}_folds.csv'
    
    LR_start_factor = 0.1
    
    val_losses = []
    val_trn_losses = []
    val_epochs = []
    val_times = []
    trn_losses = []
    trn_val_losses = []
    trn_epochs = []
    trn_times = []
    for fold_num, data_fold in enumerate(data_folded):
        model = model_constr(model_args).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR*LR_start_factor, weight_decay=weight_decay)
        manager = epoch_managers.FixedManager(max_epochs=min_epochs)
        # manager = epoch_managers.ConvergenceManager(memory=0.1, threshold=0.001, mode="const", min_epochs=min_epochs, max_epochs=max_epochs)
        
        x_train, y_train, x_valid, y_valid = data_fold
        
        steps_per_epoch = ceildiv(x_train.size(0), minibatch_examples * accumulated_minibatches)
        # base_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience = patience // 2, cooldown = patience, threshold_mode='rel', threshold=0.01)
        base_scheduler = OneCycleLR(
            optimizer, max_lr=LR, epochs=min_epochs, steps_per_epoch=steps_per_epoch, div_factor=1.0/LR_start_factor,
            final_div_factor=1.0/(LR_start_factor*0.1), three_phase=True, pct_start=0.4)
        scheduler = lr_schedule.LRScheduler(base_scheduler, initial_lr=LR*LR_start_factor)
        
        
        print(f"Fold {fold_num + 1}/{kfolds}")
        
        val_opt, trn_opt, last_opt = run_epochs(model, optimizer, scheduler, manager, minibatch_examples, accumulated_minibatches, scaler, x_train, y_train,
                x_valid, y_valid, timesteps, model_name, dataname, fold_num, loss_fn, distr_error_fn, device, outputs_per_epoch=10, verbosity=verbosity - 1)
        
        val_loss = val_opt.val_loss
        val_trn_loss = val_opt.trn_loss
        val_epoch = val_opt.epoch
        val_time = val_opt.time
        trn_loss = trn_opt.trn_loss
        trn_val_loss = trn_opt.val_loss
        trn_epoch = trn_opt.epoch
        trn_time = trn_opt.time
        
        val_losses.append(val_loss)
        val_epochs.append(val_epoch)
        val_trn_losses.append(val_trn_loss)
        val_times.append(val_time)
        trn_losses.append(trn_loss)
        trn_epochs.append(trn_epoch)
        trn_val_losses.append(trn_val_loss)
        trn_times.append(trn_time)
        
        stream.stream_results(filepath_out_fold, verbosity > 0,
            "fold", fold_num+1,
            "Validation Loss", val_loss,
            "Val @ epochs", val_epoch+1,
            "Val @ time", val_time,
            "Val @ training loss", val_trn_loss,
            "Training Loss", trn_loss,
            "Trn @ epochs", trn_epoch+1,
            "Trn @ time", trn_time,
            "Trn @ validation loss", trn_val_loss,
            prefix="\n========================================FOLD=========================================\n",
            suffix="\n=====================================================================================\n")
        
    return val_losses, val_epochs, val_times, val_trn_losses, trn_losses, trn_epochs, trn_times, trn_val_losses


def pessimistic_summary(fold_losses):
    model_score = np.max([np.mean(fold_losses), np.median(fold_losses)])
    return model_score


def main():
    
    # device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    
    # Data
    
    dataname = "waimea"
    # dataname = "waimea-condensed"
    # dataname = "cNODE-paper-ocean"
    # dataname = "cNODE-paper-human-gut"
    # dataname = "cNODE-paper-human-oral"
    # dataname = "cNODE-paper-drosophila"
    # dataname = "cNODE-paper-soil-vitro"
    # dataname = "cNODE-paper-soil-vivo"
    # dataname = "dki-synth"
    # dataname = "dki-real"
    
    # data folding params
    kfolds = 7
    DEBUG_SINGLE_FOLD = True
    
    # load data
    filepath_train = f'data/{dataname}_train.csv'
    x, y = data.load_data(filepath_train, device)
    x, y = data.shuffle_data(x, y)
    data_folded = data.fold_data(x, y, kfolds)
    if DEBUG_SINGLE_FOLD:
        data_folded = [data_folded[0]]
    
    print('dataset:', filepath_train)
    print(f'training data shape: {data_folded[0][0].shape}')
    print(f'validation data shape: {data_folded[0][2].shape}')
    
    # experiment hyperparameters
    hp = dicy()
    hp.min_epochs = 8
    hp.max_epochs = 10
    hp.patience = 1
    hp.early_stop = True
    
    # optimization hyperparameters
    hp.minibatch_examples = 256
    hp.accumulated_minibatches = 1
    hp.LR = 0.0316
    hp.WD = 0.01

    # model shape hyperparameters
    _, hp.data_dim = x.shape
    hp.hidden_dim = math.isqrt(hp.data_dim)
    hp.attend_dim = 16 # math.isqrt(hidden_dim)
    hp.num_heads = 4
    hp.depth = 2
    hp.ffn_dim_multiplier = 0.5
    assert hp.attend_dim % hp.num_heads == 0, "attend_dim must be divisible by num_heads"
    
    # Specify model(s) for experiment
    # Note that each must be a constructor function that takes a dictionary args. Lamda is recommended.
    models_to_test = {
        # 'baseline-SLP': lambda args: models_baseline.SingleLayerPerceptron(hp.data_dim),
        # 'baseline-SLPMult': lambda args: models_baseline.SingleLayerMultiplied(hp.data_dim),
        # 'baseline-SLPSum': lambda args: models_baseline.SingleLayerSummed(hp.data_dim),
        # 'baseline-SLPMultSum': lambda args: models_baseline.SingleLayerMultipliedSummed(hp.data_dim),
        'baseline-SLPReplicator': lambda args: models_baseline.SingleLayerReplicator(hp.data_dim),
        # 'baseline-cNODE0': lambda args: models_baseline.cNODE0(hp.data_dim),
        # LRRS range: 1e-2...1e0.5
        # WD range: 1e-6...1e0
        # LR:0.5994842503189424, WD:0.33
        # LR:0.8799225435691093, WD:0.33
        # 'baseline-cNODE2-width1': lambda args: models.cNODEGen_ConstructedFitness(lambda: nn.Sequential(
        #     nn.Linear(hp.data_dim, 1),
        #     nn.Linear(1, hp.data_dim))),
        #
        # 'cNODE2-custom': lambda args: models.cNODEGen_ConstructedFitness(lambda: nn.Sequential(
        #     nn.Linear(hp.data_dim, args["hidden_dim"]),
        #     nn.Linear(args["hidden_dim"], hp.data_dim))),
        # 'cNODE2-custom-nl': lambda args: models.cNODEGen_ConstructedFitness(lambda: nn.Sequential(
        #     nn.Linear(hp.data_dim, args["hidden_dim"]),
        #     nn.ReLU(),
        #     nn.Linear(args["hidden_dim"], hp.data_dim))),
        # 'cNODE-deep3': lambda args: models.cNODEGen_ConstructedFitness(lambda: nn.Sequential(
        #     nn.Linear(hp.data_dim, args["hidden_dim"]),
        #     nn.Linear(args["hidden_dim"], args["hidden_dim"]),
        #     nn.Linear(args["hidden_dim"], hp.data_dim))),
        # 'cNODE-deep3-nl': lambda args: models.cNODEGen_ConstructedFitness(lambda: nn.Sequential(
        #     nn.Linear(hp.data_dim, args["hidden_dim"]),
        #     nn.ReLU(),
        #     nn.Linear(args["hidden_dim"], args["hidden_dim"]),
        #     nn.ReLU(),
        #     nn.Linear(args["hidden_dim"], hp.data_dim))),
        # 'cNODE-deep4-flat': lambda args: models.cNODEGen_ConstructedFitness(lambda: nn.Sequential(
        #     nn.Linear(hp.data_dim, args["hidden_dim"]),
        #     nn.Linear(args["hidden_dim"], args["hidden_dim"]),
        #     nn.Linear(args["hidden_dim"], args["hidden_dim"]),
        #     nn.Linear(args["hidden_dim"], hp.data_dim))),
        # 'cNODE-deep4-flat-nl': lambda args: models.cNODEGen_ConstructedFitness(lambda: nn.Sequential(
        #     nn.Linear(hp.data_dim, args["hidden_dim"]),
        #     nn.ReLU(),
        #     nn.Linear(args["hidden_dim"], args["hidden_dim"]),
        #     nn.ReLU(),
        #     nn.Linear(args["hidden_dim"], args["hidden_dim"]),
        #     nn.ReLU(),
        #     nn.Linear(args["hidden_dim"], hp.data_dim))),
        # 'cNODE1': lambda args: models.cNODE1(hp.data_dim),
        # 'cNODE2': lambda args: models.cNODE2(hp.data_dim),
        # LR: 0.03, WD: 3.3
        
        # 'canODE-noValue': lambda args: models_condensed.canODE_attentionNoValue(hp.data_dim, args["attend_dim"], args["attend_dim"]),
        # # 'canODE-noValue-static': lambda args: models_condensed.canODE_attentionNoValue_static(hp.data_dim, args["attend_dim"], args["attend_dim"]),
        # 'canODE': lambda args: models_condensed.canODE_attention(hp.data_dim, args["attend_dim"], args["attend_dim"]),
        # 'canODE-multihead': lambda args: models_condensed.canODE_attentionMultihead(hp.data_dim, args["attend_dim"], args["num_heads"]),
        # 'canODE-singlehead': lambda args: models_condensed.canODE_attentionMultihead(hp.data_dim, args["attend_dim"], 1),
        # 'canODE-transformer': lambda args: models_condensed.canODE_transformer(hp.data_dim, args["attend_dim"], args["num_heads"], args["depth"], args["ffn_dim_multiplier"]),
        # 'canODE-transformer-d2': lambda args: models_condensed.canODE_transformer(hp.data_dim, args["attend_dim"], args["num_heads"], 2, args["ffn_dim_multiplier"]),
        # 'canODE-transformer-d6': lambda args: models_condensed.canODE_transformer(hp.data_dim, args["attend_dim"], args["num_heads"], 6, args["ffn_dim_multiplier"]),
        # 'canODE-transformer-d6-old': lambda args: models_condensed.canODE_transformer(hp.data_dim, args["attend_dim"], 4, 6, args["ffn_dim_multiplier"]),
        # 'canODE-transformer-d3-a8-h2-f0.5': lambda args: models_condensed.canODE_transformer(hp.data_dim, 8, 2, 3, 0.5),
        # 'canODE-transformer-d3-med': lambda args: models_condensed.canODE_transformer(hp.data_dim, 32, 4, 3, 1.0),
        # 'canODE-transformer-d3-big': lambda args: models_condensed.canODE_transformer(hp.data_dim, 64, 16, 3, 2.0),
        
        # 'cAttend-simple': lambda args: models_condensed.cAttend_simple(hp.data_dim, args["attend_dim"], args["attend_dim"]),
        # 'Embedded-cNODE2': lambda args: models.Embedded_cNODE2(hp.data_dim, args["hidden_dim"]),  # this model is not good
        # 'cNODE2_DKI': lambda args: models.cNODE2_DKI(hp.data_dim), # sanity test, this is the same as cNODE2 but less optimized
        # 'cNODE2-Gen': lambda args: models.cNODEGen_ConstructedFitness(lambda: nn.Sequential(nn.Linear(hp.data_dim, hp.data_dim), nn.Linear(hp.data_dim, hp.data_dim))),  # sanity test, this is the same as cNODE2 but generated at runtime
        # "cNODE2-static": lambda args: models.cNODE2_ExternalFitness(hp.data_dim),
        # "cNODE2-FnFitness": lambda args: models.cNODE2_FnFitness(hp.data_dim), # sanity test, this is the same as cNODE2 but testing externally-supplied fitness functions
    }
    
    
    # specify loss function
    loss_fn = loss_bc
    # loss_fn = lambda y_pred,y_true: loss_bc(y_pred, y_true) + distribution_error(y_pred)
    
    distr_error_fn = distribution_error
    
    # scaler = torch.amp.GradScaler(device)
    scaler = torch.cuda.amp.GradScaler()
    
    # time step "data"
    ode_timesteps = 5  # must be at least 2. TODO: run this through hyperparameter opt to verify that it doesn't impact performance
    ode_timemax = 1.0
    timesteps = torch.arange(0.0, ode_timemax, ode_timemax / (ode_timesteps - 1)).to(device)
    
    # # START of hacky hyperparam search - remove
    # minibatch_examples_list = [1, 8, 16, 32]
    # accumulated_minibatches_list = [1, 4, 8]
    # LR_list = [0.0001, 0.002, 0.04, 0.8, 10.0]
    # WD_list = [0.0, 0.0001, 0.001, 0.01]
    # for minibatch_examples in minibatch_examples_list:
    #     for accumulated_minibatches in accumulated_minibatches_list:
    #         for LR in LR_list:
    #             for WD in WD_list:
    #                 #     # END of hacky hyperparam search - remove
    
    # # START of hacky hyperparam search - remove
    # LR_list = [0.00004, 0.0001, 0.0004, 0.001, 0.004] #, 0.01, 0.04] #, 0.1, 0.4, 1.0]
    # WD_list = [0.0] #, 0.000001, 0.00001, 0.0001]
    # for LR in LR_list:
    #     for WD in WD_list:
    #         # END of hacky hyperparam search - remove
    
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
    
    filepath_out_expt = f'results/{dataname}_experiments.csv'
    seed = int(time.time()) # currently only used to set the data shuffle seed in find_LR
    print(f"Seed: {seed}")
    for model_name, model_constr in models_to_test.items():
    
        model_args = {"hidden_dim": hp.hidden_dim, "attend_dim": hp.attend_dim, "num_heads": hp.num_heads, "depth": hp.depth, "ffn_dim_multiplier": hp.ffn_dim_multiplier}
        
        # # TODO remove this: it's just to resume from where we were previously
        # if ((attend_dim == 4 or attend_dim == 16) and model_name == 'canODE-transformer-d6' and num_heads == 4):
        #     continue
        
        # try:
        print(f"\nRunning model: {model_name}")
        
        # test construction and print parameter count
        model = model_constr(model_args)
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Number of parameters in model: {num_params}")
        
        # find optimal LR
        hp.WD, hp.LR = hyperparameter_search_with_LRfinder(
            model_constr, model_args, model_name, scaler, x, y, hp.minibatch_examples, hp.accumulated_minibatches,
            device, 6, 6, dataname, timesteps, loss_fn, distr_error_fn, verbosity=1, seed=seed)
        print(f"LR:{hp.LR}, WD:{hp.WD}")
        # TODO: remove, hardcoded values for cNODE2 (steepest point found in LRFinder)
        # hp.WD = 0.33
        # hp.LR = 0.03
        
        hp = ui.ask(hp, keys=["LR", "WD"])
        
        # print hyperparams
        for key, value in hp.items():
            print(f"{key}: {value}")
        
        # train and test the model across multiple folds
        val_losses, val_epochs, val_times, val_trn_losses, trn_losses, trn_epochs, trn_times, trn_val_losses = crossvalidate_model(
            hp.LR, scaler, hp.accumulated_minibatches, data_folded, device, hp.early_stop, hp.patience,
            kfolds, hp.min_epochs, hp.max_epochs, hp.minibatch_examples, model_constr, model_args,
            model_name, dataname, timesteps, loss_fn, distr_error_fn, hp.WD, verbosity=0
        )
        
        
        
        print(f"Val Losses: {val_losses}")
        print(f"Epochs: {val_epochs}")
        print(f"Durations: {val_times}")
        print(f"Trn Losses: {val_trn_losses}")
        
        for i in range(len(val_losses)):
            stream.stream_scores(filepath_out_expt, True,
                "model", model_name,
                "model parameters", num_params,
                "Validation Score", val_losses[i],
                "Val @ Epoch", val_epochs[i],
                "Val @ Time", val_times[i],
                "Val @ Trn Loss", val_trn_losses[i],
                "Train Score", trn_losses[i],
                "Trn @ Epoch", trn_epochs[i],
                "Trn @ Time", trn_times[i],
                "Trn @ Val Loss", trn_val_losses[i],
                "k-folds", kfolds,
                "timesteps", ode_timesteps,
                *list(itertools.chain(*(hp.items()))), # unroll the hyperparams dictionary
                prefix="\n=======================================================EXPERIMENT========================================================\n",
                suffix="\n=========================================================================================================================\n")
        
        # summary score for each result vector
        model_score = pessimistic_summary(val_losses)
        model_epoch = pessimistic_summary(val_epochs)
        model_time = pessimistic_summary(val_times)
        model_trn_loss = pessimistic_summary(val_trn_losses)
        
        print(f"Avg Val Losses: {model_score}")
        print(f"Avg Epochs: {model_epoch}")
        print(f"Avg Durations: {model_time}")
        print(f"Avg Trn Losses: {model_trn_loss}")
        
        # except Exception as e:
        #     stream.stream_scores(filepath_out_expt, True,
        #         "model", model_name,
        #         "model parameters", -1,
        #         "Validation Score", -1,
        #         "Val @ Epoch", -1,
        #         "Val @ Time", -1,
        #         "Val @ Trn Loss", -1,
        #         "Train Score", -1,
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
    stream.keep_plots_open()


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