
import itertools
import os
import traceback
import numpy as np
import pandas as pd
import chunked_dataset
import epoch_managers
import lr_schedule
from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau
import data
import torch
import models_baseline
import stream
from optimum import Optimum, summarize, unrolloptims
import stream_plot as plotstream
import time


import copy
import time

import torch


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


def ceildiv(a, b):
    return -(a // -b)


def validate_epoch(model, requires_condensed, data_val, minibatch_examples, t, loss_fn, score_fn, distr_error_fn, device):
    model.eval()
    
    total_loss = 0.0
    total_score = 0.0
    total_distr_error = 0.0
    # total_samples = data_val[chunked_dataset.DK_X].size(0)
    # minibatches = ceildiv(total_samples, minibatch_examples)
    # current_index = 0  # Initialize current index to start of dataset

    total_samples = 0

    for mb in data_val:
        x, y, x_sparse, y_sparse, ids = mb[chunked_dataset.DK_X], mb[chunked_dataset.DK_Y], mb[chunked_dataset.DK_XSPARSE], mb[chunked_dataset.DK_YSPARSE], mb[chunked_dataset.DK_IDS]
        
        if requires_condensed:
            x = x_sparse.to(device)
            y = y_sparse.to(device)
            ids = ids.to(device)
        else:
            x = x.to(device)
            y = y.to(device)

        # TODO: Currently hacked to only work with the Identity model. Need to generalize it.

        # z, ids, current_index = data.get_batch_raw(data_val, t, minibatch_examples, current_index, noise_level_x=0.0, noise_level_y=0.0, requires_timesteps=False)
        # x, y = z[0], z[-1]
        mb_examples = x.size(0)
        total_samples += mb_examples

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


def run_test(model, requires_condensed, model_name, model_config, epoch, mini_epoch, minibatch_examples, data_test, t, fold, loss_fn, score_fn, distr_error_fn, device, filepath_out_test, gpu_memory_reserved, cpuRam, elapsed_time, train_loss=-1.0, val_loss=-1.0):
    l_test, score_test, p_test = validate_epoch(model, requires_condensed, data_test, minibatch_examples, t, loss_fn, score_fn, distr_error_fn, device)
    stream.stream_results(filepath_out_test, True, True, True,
                            "model_name", model_name,
                            "model_config", model_config,
                            "fold", fold,
                            "epoch", epoch,
                            "mini-epoch", mini_epoch,
                            "Avg Test Loss", l_test,
                            "Avg DKI Test Loss", score_test,
                            "Avg Test Distr Error", p_test,
                            "Avg Train Loss", train_loss,
                            "Avg Validation Loss", val_loss,
                            "Elapsed Time", elapsed_time,
                            "VRAM (GB)", gpu_memory_reserved / (1024 ** 3),
                            "Peak RAM (GB)", cpuRam / (1024 ** 3),
                            prefix="==================TESTING=================\n",
                            suffix="\n=============================================\n")
    return l_test

def is_finite_number(number):
    return torch.all(torch.isfinite(number)) and not torch.any(torch.isnan(number))

def train_mini_epoch(model, requires_condensed, epoch_data_iterator, data_train, mini_epoch_size, full_epoch_size, minibatch_examples, accumulated_minibatches, noise, interpolate, interpolate_noise, optimizer, scheduler, scaler, t,
                outputs_per_mini_epoch, 
                prev_examples, prev_updates, fold, epoch_num, mini_epoch_num, model_config, dataname, loss_fn, score_fn, distr_error_fn, device,
                filepath_out_incremental, lr_plot=None, loss_plot=None, lr_loss_plot=None, verbosity=1, supervised_timesteps=1):
    model.train()
    
    total_loss = 0
    total_penalty = 0

    if mini_epoch_size > 0:
        minibatches = ceildiv(mini_epoch_size, minibatch_examples)
        loop_batches = True
    else:
        minibatches = ceildiv(full_epoch_size, minibatch_examples)
        loop_batches = False
    
    new_examples = 0
    new_updates = 0
    
    stream_interval = max(1, minibatches // outputs_per_mini_epoch)
    
    optimizer.zero_grad()
    
    # set up metrics for streaming
    prev_time = time.time()
    stream_loss = 0
    stream_penalty = 0
    stream_examples = 0
    
    for mb in range(minibatches):
        try: 
            batch = next(epoch_data_iterator)
        except StopIteration:
            epoch_num += 1
            epoch_data_iterator = iter(data_train)
            batch = next(epoch_data_iterator)

        model_requires_timesteps = getattr(model, 'USES_ODEINT', False)
        supervise_steps = interpolate and model_requires_timesteps
        
        # z, ids, current_index, epoch_num = data.get_batch(data_train, t, minibatch_examples, current_index, total_samples, epoch_num, noise_level_x=noise, noise_level_y=noise, interpolate_noise=interpolate_noise, requires_timesteps=supervise_steps, loop=loop_batches, shuffle=True)
        x, y, x_sparse, y_sparse, ids = batch[chunked_dataset.DK_X], batch[chunked_dataset.DK_Y], batch[chunked_dataset.DK_XSPARSE], batch[chunked_dataset.DK_YSPARSE], batch[chunked_dataset.DK_IDS]
        if requires_condensed:
            x = x_sparse.to(device)
            y = y_sparse.to(device)
            ids = ids.to(device)
        else:
            x = x.to(device)
            y = y.to(device)

        if noise > 0.0:
            # print(f"x: \n{x}")
            x = data.noisy_x0(x, noise)
            # print(f"noisy x: \n{x}")

        # mb_examples = z.shape[-2]
        mb_examples = x.size(0)
        
        y_pred = eval_model(model, x, t, ids)
        if supervise_steps:
            y_pred = y_pred[1:]
            y_true = z[1:]  # TODO: Fix: implement supervised interpolation with new data source
        else:
            y_pred = y_pred[-1:]
            y_true = y.unsqueeze(0)  # TODO: may not need this unsqueeze if we stop supporting supervised interpolation
        y_pred = y_pred.to(device)

        loss = loss_fn(y_pred, y_true)
        actual_loss = loss.item() * mb_examples
        loss = loss / accumulated_minibatches  # Normalize the loss by the number of accumulated minibatches, since loss function can't normalize by this
        
        scaled_loss = scaler.scale(loss)
        if scaled_loss.requires_grad and scaled_loss.grad_fn is not None:
            scaled_loss.backward()
        else:
            print(f"GRADIENT ERROR: Loss at epoch {epoch_num} mini-epoch {mini_epoch_num} minibatch {mb} does not require gradient. Computation graph detached?")
        
        distr_error = distr_error_fn(y_pred)
        actual_penalty = distr_error.item() * mb_examples
        
        # del y_pred, loss, distr_error
        
        total_loss += actual_loss
        total_penalty += actual_penalty
        new_examples += mb_examples
        new_updates += 1
        
        stream_loss += actual_loss
        stream_penalty += actual_penalty
        stream_examples += mb_examples
        
        if (mb + 1) % stream_interval == 0:
            end_time = time.time()
            examples_per_second = stream_examples / max(end_time - prev_time,
                                                        0.0001)  # TODO: Find a better way to handle div by zero, or at least a more appropriate nonzero value
            stream.stream_results(filepath_out_incremental, verbosity > 0, verbosity > 0, verbosity > -1,
                                  "fold", fold,
                                  "epoch", epoch_num,
                                  "mini-epoch", mini_epoch_num,
                                  "minibatch", mb,
                                  "total examples seen", prev_examples + new_examples,
                                  "total updates", prev_updates + new_updates,
                                  "Avg Loss", stream_loss / stream_examples,
                                  "Avg Distr Error", stream_penalty / stream_examples,
                                  "Examples per second", examples_per_second,
                                  "Learning Rate", scheduler.get_last_lr(),
                                  )
            if lr_plot:
                plotstream.plot_single(lr_plot, "mini-epochs", "LR", f"{model_config} fold {fold}",
                                       mini_epoch_num + mb / minibatches, scheduler.get_last_lr(), False, y_log=False)
            if loss_plot:
                plotstream.plot_loss(loss_plot, f"{model_config} fold {fold}", mini_epoch_num + mb / minibatches,
                                     stream_loss / stream_examples, None, add_point=False)
            if lr_loss_plot:
                plotstream.plot_single(lr_loss_plot, "log( Learning Rate )", "Loss", f"{model_config} fold {fold}",
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
    


    avg_loss = total_loss / new_examples  # This used to be divided by total_samples which is dataset size, that's wrong isn't it?
    avg_penalty = total_penalty / new_examples
    new_total_examples = prev_examples + new_examples
    new_total_updates = prev_updates + new_updates
    return avg_loss, avg_penalty, new_total_examples, new_total_updates, epoch_num, epoch_data_iterator


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


def run_epochs(model, requires_condensed, optimizer, scheduler, manager, minibatch_examples, accumulated_minibatches, noise, interpolate, interpolate_noise, scaler, data_train, data_valid, data_test, t,
               model_name, model_config, dataname, fold, loss_fn, score_fn, distr_error_fn, device, mini_epoch_size, full_epoch_size, outputs_per_epoch=10, verbosity=1,
               preeval_training_set=True, reeval_train_epoch=False, reeval_train_final=True, jobstring="", use_best_model=True):
    # assert(data.check_leakage([(x_train, y_train, x_valid, y_valid)]))

    epoch = 0
    current_sample_index = 0
    
    # track stats at various definitions of the "best" epoch
    val_opt = Optimum('val_loss', 'min')
    trn_opt = Optimum('trn_loss', 'min')
    valscore_opt = Optimum('val_score', 'min')
    trnscore_opt = Optimum('trn_score', 'min')
    last_opt = Optimum(metric=None)  # metric None to update it every time. metric="epoch" would do the same
    
    old_lr = scheduler.get_last_lr()
    
    filepath_out_epoch = f'results/epochs/{model_name}_{dataname}{jobstring}_epochs.csv'
    filepath_out_test = f'results/tests/{dataname}{jobstring}_tests.csv'
    # filepath_out_model = f'results/logs/{model_config}_{dataname}_model.pth'
    filepath_out_incremental = f'results/incr/{model_config}_{dataname}{jobstring}_incremental.csv'
    
    # initial validation benchmark
    print("Evaluating initial validation score")
    l_val, score_val, p_val = validate_epoch(model, requires_condensed, data_valid, minibatch_examples, t, loss_fn, score_fn,
                                             distr_error_fn, device)
    if preeval_training_set:
        print("Evaluating initial training score")
        l_trn, score_trn, p_trn = validate_epoch(model, requires_condensed, data_train, minibatch_examples, t, loss_fn, score_fn,
                                             distr_error_fn, device)
    else:
        l_trn = None
        score_trn = None
        p_trn = None

    gpu_memory_reserved = torch.cuda.memory_reserved(device)
    _, cpuRam = tracemalloc.get_traced_memory()
    stream.stream_results(filepath_out_epoch, verbosity > 0, verbosity > 0, verbosity > -1,
                          "fold", fold,
                          "epoch", 0,
                          "mini-epoch", 0,
                          "training examples", 0,
                          "update steps", 0,
                          "Avg Training Loss", l_trn if l_trn is not None else -1.0,
                          "Avg DKI Trn Loss", score_trn if score_trn is not None else -1.0,
                          "Avg Training Distr Error", p_trn if p_trn is not None else -1.0,
                          "Avg Validation Loss", l_val,
                          "Avg DKI Val Loss", score_val,
                          "Avg Validation Distr Error", p_val,
                          "Learning Rate", old_lr,
                          "Elapsed Time", 0.0,
                          "VRAM (GB)", gpu_memory_reserved / (1024 ** 3),
                          "Peak RAM (GB)", cpuRam  / (1024 ** 3),
                          prefix="================PRE-VALIDATION===============\n",
                          suffix="\n=============================================\n")
    plotstream.plot_loss(f"loss {dataname}", f"{model_config} fold {fold}", 0, l_trn, l_val, add_point=False)
    # plotstream.plot_loss(f"score {dataname}", f"{model_config} fold {fold}", 0, score_trn, score_val, add_point=False)
    plotstream.plot(f"stopmetric {dataname}", "mini-epoch", "metric", [f"metric {model_config} fold {fold}", f"threshold {model_config} fold {fold}"], manager.epoch, [manager.get_metric(), manager.get_threshold()], add_point=False)
    # plotstream.plot(f"Validation Loss EMA {dataname}", "mini-epoch", "metric", [f"val_EMA {model_config} fold {fold}"], manager.epoch, [manager.get_supplemental()["val_EMA"]], add_point=False) # Commented because constant epoch manager doesn't have an EMA
    # plotstream.plot(dataname, "epoch", "loss", [f"{model_config} fold {fold} - Val", f"{model_config} fold {fold} - Trn", f"{model_config} fold {fold} - DKI Val", f"{model_config} fold {fold} - DKI Trn"], 0, [l_val, None, score_val, None], add_point=False)
    
    train_examples_seen = 0
    update_steps = 0
    start_time = time.time()
    
    # if reptile_rate > 0.0:
    #     # Create a copy of the model to serve as the meta-model
    #     meta_model = copy.deepcopy(model)
    #     outer_optimizer = type(optimizer)(meta_model.parameters())
    #     outer_optimizer.load_state_dict(optimizer.state_dict())
    #     outer_optimizer.lr = reptile_rate
    
    training_curve = []
    has_backup = False

    # TODO: replace this quick and dirty dict packing. They should have always been in a dict.
    dict = {"epoch": epoch, "mini_epoch": manager.epoch, "trn_loss": l_trn, "trn_score": score_trn, "val_loss": l_val,
            "val_score": score_val, "lr": old_lr, "time": start_time, "gpu_memory": gpu_memory_reserved,
            "metric": p_val, "stop_metric": manager.get_metric(), "stop_threshold": manager.get_threshold()}
    # track various optima
    model_path = f'results/models/{model_config}_{dataname}_fold{fold}_job{jobstring}.pt'
    if use_best_model:
        val_opt.track_best(dict, model=model, model_path=model_path)
    else:
        val_opt.track_best(dict)
    valscore_opt.track_best(dict)
    trn_opt.track_best(dict)
    trnscore_opt.track_best(dict)
    last_opt.track_best(dict)
    manager.set_baseline(last_opt)

    epoch_data_iterator = iter(data_train)

    print("Starting training")

    while True:
        l_trn, p_trn, train_examples_seen, update_steps, epoch, epoch_data_iterator = train_mini_epoch(
            model, requires_condensed, epoch_data_iterator, data_train, mini_epoch_size, full_epoch_size, minibatch_examples,
            accumulated_minibatches, noise, interpolate, interpolate_noise, optimizer, scheduler, scaler, t,
            outputs_per_epoch, train_examples_seen, update_steps, 
            fold, epoch, manager.epoch, model_config, dataname, loss_fn, score_fn,
            distr_error_fn, device, filepath_out_incremental,
            lr_plot="Learning Rate", verbosity=verbosity - 1
        )
        # if reptile_rate > 0.0:
        #     # Meta-update logic
        #     with torch.no_grad():
        #         # Apply the difference as pseudo-gradients to the outer optimizer
        #         meta_weights = {name: param.clone() for name, param in meta_model.state_dict().items()}
        #         for name, param in meta_model.named_parameters():
        #             param.grad = (meta_weights[name] - model.state_dict()[name])
                
        #         # Use the outer optimizer to step with these gradients
        #         outer_optimizer.step()
                
        #         # Synchronize the inner model with the updated meta-model weights
        #         model.load_state_dict(meta_model.state_dict())
        
        l_val, score_val, p_val = validate_epoch(model, requires_condensed, data_valid, minibatch_examples, t, loss_fn, score_fn,
                                                 distr_error_fn, device)
        
        if reeval_train_epoch:
            print("Re-evaluating training score")
            l_trn, score_trn, p_trn = validate_epoch(model, requires_condensed, data_train, minibatch_examples, t, loss_fn, score_fn,
                                                     distr_error_fn, device)
        else:
            score_trn = -1.0

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
        
        # TODO: replace this quick and dirty dict packing. They should have always been in a dict.
        dict = {"epoch": epoch, "mini_epoch": manager.epoch, "trn_loss": l_trn, "trn_score": score_trn, "val_loss": l_val,
                "val_score": score_val, "lr": old_lr, "time": elapsed_time, "gpu_memory": gpu_memory_reserved,
                "metric": p_val, "stop_metric": manager.get_metric(), "stop_threshold": manager.get_threshold()}
        # track various optima
        model_path = f'results/models/{model_config}_{dataname}_fold{fold}_job{jobstring}.pt'
        val_opt.track_best(dict, model=model, model_path=model_path)
        valscore_opt.track_best(dict)
        trn_opt.track_best(dict)
        trnscore_opt.track_best(dict)
        last_opt.track_best(dict)
        
        training_curve.append({"fold": fold, "epoch":epoch, "mini_epoch": manager.epoch, "trn_loss": l_trn, "val_loss": l_val, "time": elapsed_time})
        
        old_lr = new_lr
        
        # update epoch manager, which decides if we should stop or continue
        should_stop = manager.should_stop(last_opt)
        
        # log results (after updating manager so we can log stats from the manager itself)
        stream.stream_results(filepath_out_epoch, verbosity > 0, verbosity > 0, verbosity > -1,
                              "fold", fold,
                              "epoch", epoch,
                              "mini-epoch", manager.epoch,
                              "training examples", train_examples_seen,
                              "update steps", update_steps,
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
        
        plotstream.plot_loss(f"loss {dataname}", f"{model_config} fold {fold}", manager.epoch if mini_epoch_size <= 0 else update_steps, l_trn, l_val,
                             add_point=add_point)
        # plotstream.plot_loss(f"score {dataname}", f"{model_config} fold {fold}", manager.epoch, score_trn, score_val, add_point=add_point)
        plotstream.plot(f"stopmetric {dataname}", "mini-epoch", "metric", [f"metric {model_config} fold {fold}", f"threshold {model_config} fold {fold}"], manager.epoch, [manager.get_metric(), manager.get_threshold()], add_point=False)
        # plotstream.plot(f"Validation Loss EMA {dataname}", "mini-epoch", "metric", [f"val_EMA {model_config} fold {fold}"], manager.epoch, [manager.get_supplemental()["val_EMA"]], add_point=False) # Commented because constant epoch manager doesn't have an EMA
        # plotstream.plot(dataname, "epoch", "loss", [f"{model_config} fold {fold} - Val", f"{model_config} fold {fold} - Trn", f"{model_config} fold {fold} - DKI Val", f"{model_config} fold {fold} - DKI Trn"], manager.epoch + 1, [l_val, l_trn, score_val, score_trn], add_point=add_point)
        # if l_val != score_val:
        #     print("WARNING: CURRENT LOSS METRIC DISAGREES WITH DKI LOSS METRIC")

        # time to stop: optionally load model and run test.
        test_score = None
        if should_stop:
            print("===Stopping training===")
            if data_test or reeval_train_final:
                if use_best_model:
                    model.load_state_dict(torch.load(model_path, weights_only=True))
                if reeval_train_final:
                    print("Re-evaluating final training score")
                    l_trn, score_trn, p_trn = validate_epoch(model, requires_condensed, data_train, minibatch_examples, t, loss_fn, score_fn, distr_error_fn, device)
                    print(f"Final Training Loss: {l_trn}, Final Validation Loss: {l_val}")
                if data_test:
                    print("Running test")
                    test_score = run_test(model, requires_condensed, model_name, model_config, val_opt.epoch, val_opt.mini_epoch, minibatch_examples, data_test, t, fold, loss_fn, score_fn, distr_error_fn, device, filepath_out_test, gpu_memory_reserved, cpuRam, elapsed_time, train_loss=l_trn, val_loss=l_val)
            break
    
    return val_opt, valscore_opt, trn_opt, trnscore_opt, last_opt, training_curve, test_score


def crossvalidate_model(LR, scaler, accumulated_minibatches, data_folded, data_test, total_train_samples, noise, interpolate, interpolate_noise, device, early_stop, patience, kfolds,
                        min_epochs, max_epochs, mini_epoch_size, 
                        minibatch_examples, model_constr, epoch_manager_constr, model_args, model_name, model_config, dataname, timesteps, loss_fn,
                        score_fn, distr_error_fn, weight_decay, verbosity=1, preeval_training_set=True, reeval_train_epoch=False, reeval_train_final=True,
                        whichfold=-1, jobstring="", use_best_model=True):
    filepath_out_fold = f'results/folds/{model_config}_{dataname}{jobstring}_folds.csv'
    
    # LR_start_factor = 0.1 # OneCycle
    # LR_start_factor = 1.0  # constantLR
    LR_start_factor = 0.0 # warm up
    
    val_loss_optims = []
    val_score_optims = []
    trn_loss_optims = []
    trn_score_optims = []
    final_optims = []
    test_scores = []
    val_loss_curves = []
    for fold_num, data_fold in enumerate(data_folded):
        if whichfold >= 0:
            fold_num = whichfold
        model = model_constr(model_args).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR * LR_start_factor, weight_decay=weight_decay)

        manager = epoch_manager_constr(model_args)
        # optimizer = torch.optim.SGD(model.parameters(), lr=LR*LR_start_factor, weight_decay=weight_decay)
        # manager = epoch_managers.FixedManager(max_epochs=min_epochs)
        # manager = epoch_managers.ExplosionManager(memory=0.5, threshold=1.0, mode="rel", max_epochs=max_epochs)
        # manager = epoch_managers.EarlyStopManager(memory=0.0, threshold=0.0, mode="rel", max_epochs=max_epochs)
        # manager = epoch_managers.ConvergenceManager(memory=0.1, threshold=0.001, mode="const", min_epochs=min_epochs, max_epochs=max_epochs)
        
        # x_train, y_train, x_valid, y_valid = data_fold
        requires_condensed = getattr(model, 'USES_CONDENSED', False)
        data_train = data_fold[0]
        data_valid = data_fold[1]
        # if not requires_condensed:
        #     data_train = [data_fold[0][0], data_fold[1][0]]
        #     data_valid = [data_fold[0][1], data_fold[1][1]]
        #     if testdata:
        #         data_test = [testdata[0], testdata[1]]
        #     else:
        #         data_test = None
        # else:
        #     data_train = [data_fold[2][0], data_fold[3][0], data_fold[4][0]]
        #     data_valid = [data_fold[2][1], data_fold[3][1], data_fold[4][1]]
        #     if testdata:
        #         data_test = [testdata[2], testdata[3], testdata[4]]
        #     else:
        #         data_test = None

        
        steps_per_epoch = ceildiv(mini_epoch_size if mini_epoch_size > 0 else total_train_samples, minibatch_examples * accumulated_minibatches)
        # print(f"Steps per epoch: {steps_per_epoch}")
        # base_scheduler = lr_schedule.ConstantLR(optimizer)
        # base_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=patience // 2, cooldown=patience,
        #                                    threshold_mode='rel', threshold=0.01)
        # base_scheduler = OneCycleLR(
        #     optimizer, max_lr=LR, epochs=min_epochs, steps_per_epoch=steps_per_epoch, div_factor=1.0/LR_start_factor,
        #     final_div_factor=1.0/(LR_start_factor*0.1), three_phase=True, pct_start=0.4, anneal_strategy='cos')
        update_steps = max_epochs*steps_per_epoch
        base_scheduler = lr_schedule.DirectToZero(optimizer, peak_lr=LR, update_steps=update_steps, warmup_proportion=0.1)
        scheduler = lr_schedule.LRScheduler(base_scheduler, initial_lr=LR * LR_start_factor)
        
        print(f"Fold {fold_num + 1}/{kfolds}")
        
        val_opt, valscore_opt, trn_opt, trnscore_opt, last_opt, training_curve, test_score = run_epochs(
            model, requires_condensed, optimizer, scheduler, manager, minibatch_examples, accumulated_minibatches, 
            noise, interpolate, interpolate_noise, scaler, data_train, data_valid, data_test,
            timesteps, model_name, model_config, dataname, fold_num, loss_fn, score_fn,
            distr_error_fn, device, mini_epoch_size, total_train_samples, 
            outputs_per_epoch=10, verbosity=verbosity - 1,
            # reptile_rate=reptile_rewind,
            preeval_training_set=preeval_training_set, reeval_train_epoch=reeval_train_epoch, reeval_train_final=reeval_train_final,
            jobstring=jobstring, 
            use_best_model=True
        )
        

        # Below is temporarily commented out - fix it for the new dataset formats
        # Print output of model on a batch of test examples
        # DEBUG_OUTPUT = False  # TO DO: make this an actual parameter
        # DEBUG_OUT_NUM = 4
        
        # if DEBUG_OUTPUT:
        #     DEBUG_OUT_CSV = f"./analysis/debug_outputs/{model_config}_{dataname}{jobstring}_predictions.csv"
            
        #     model.eval()
        #     with torch.no_grad():
        #         # Get the model output for the first minibatch
        #         debug_ys = eval_model(model, x_valid[:DEBUG_OUT_NUM].to(device), timesteps)
                
        #         # Get the corresponding y_valid batch
        #         y_valid_batch = y_valid[0:DEBUG_OUT_NUM].to(device)
                
        #         print(f"Example output of model {model_config} on first test batch")
                
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
        test_scores.append(test_score)
        
        stream.stream_results(filepath_out_fold, verbosity > 0, verbosity > 0, verbosity > -1,
                              "fold", fold_num,
                              "Validation Loss", val_opt.val_loss,
                              "Validation Score", val_opt.val_score,
                              "Val @ epochs", val_opt.epoch,
                              "Val @ mini-epochs", val_opt.mini_epoch,
                              "Val @ time", val_opt.time,
                              "Val @ training loss", val_opt.trn_loss,
                              "Training Loss", trn_opt.trn_loss,
                              "Training Score", trn_opt.trn_score,
                              "Trn @ epochs", trn_opt.epoch,
                              "Trn @ mini-epochs", trn_opt.mini_epoch,
                              "Trn @ time", trn_opt.time,
                              "Trn @ validation loss", trn_opt.val_loss,
                              "Test Loss", test_score,
                              prefix="\n========================================FOLD=========================================\n",
                              suffix="\n=====================================================================================\n")
    
    val_loss_curves.append(training_curve)
    
    return val_loss_optims, val_score_optims, trn_loss_optims, trn_score_optims, final_optims, val_loss_curves, test_scores



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



def run_experiment(cp, dp, hp, data_folded, testdata, device, models, epoch_mngr_constructors, loss_fn, score_fn, distr_error_fn, identity_loss, identity_score, dense_columns, sparse_columns):
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
                            "test loss", -1,
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
        val_loss_optims, val_score_optims, trn_loss_optims, trn_score_optims, final_optims, training_curves, test_scores = crossvalidate_model(
            hp.lr, scaler, hp.accumulated_minibatches, data_folded, testdata, dp.total_train_samples, hp.noise, hp.interpolate, hp.interpolate_noise, device, hp.early_stop, hp.patience,
            dp.kfolds, hp.min_epochs, hp.adjusted_epochs, hp.mini_epoch_size, dp.minibatch_examples, model_constr, epoch_manager_constr, hp,
            hp.model_name, hp.model_config, dp.y_dataset, timesteps, loss_fn, score_fn, distr_error_fn, hp.wd, verbosity=1,
            preeval_training_set=hp.preeval_training_set, reeval_train_epoch=hp.reeval_training_set_epoch, reeval_train_final=hp.reeval_training_set_final, 
            whichfold=dp.whichfold, jobstring=jobstring, use_best_model=hp.use_best_model
        )
        
        # print all folds
        print(f'Val Loss optimums: \n{[f'Fold {num}: {opt}\n' for num, opt in enumerate(val_loss_optims)]}\n')
        print(f'Val Score optimums: \n{[f'Fold {num}: {opt}\n' for num, opt in enumerate(val_score_optims)]}\n')
        print(f'Trn Loss optimums: \n{[f'Fold {num}: {opt}\n' for num, opt in enumerate(trn_loss_optims)]}\n')
        print(f'Trn Score optimums: \n{[f'Fold {num}: {opt}\n' for num, opt in enumerate(trn_score_optims)]}\n')
        print(f'Final optimums: \n{[f'Fold {num}: {opt}\n' for num, opt in enumerate(final_optims)]}\n')
        print(f'Test scores: \n{[f'Fold {num}: {score}\n' for num, score in enumerate(test_scores)]}\n')
        
        # calculate fold summaries
        avg_val_loss_optim = summarize(val_loss_optims)
        avg_val_score_optim = summarize(val_score_optims)
        avg_trn_loss_optim = summarize(trn_loss_optims)
        avg_trn_score_optim = summarize(trn_score_optims)
        avg_final_optim = summarize(final_optims)

        # mean of test_scores, which is a list of either numbers or Nones
        avg_test_score = np.nanmean([score for score in test_scores if score is not None])
        
        # print summaries
        print(f'Avg Val Loss optimum: {avg_val_loss_optim}')
        print(f'Avg Val Score optimum: {avg_val_score_optim}')
        print(f'Avg Trn Loss optimum: {avg_trn_loss_optim}')
        print(f'Avg Trn Score optimum: {avg_trn_score_optim}')
        print(f'Avg Final optimum: {avg_final_optim}')
        print(f'Avg Test score: {avg_test_score}')
        
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
                                "test loss", test_scores[i],
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
                        "test loss", -1,
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



def test_identity_model(dp, data_folded, device, loss_fn, score_fn, distr_error_fn):

    # Establish baseline performance and add to plots
    identity_model = models_baseline.ReturnInput()
    identity_loss, identity_score, identity_distro_error = validate_epoch(identity_model, False, data_folded[0][1], 100,  # using 100 instead of hp.minibatch_examples, because this model doesn't learn so the only concern is computational throughput.
                                                                    [0.0, 1.0], loss_fn, score_fn, distr_error_fn,
                                                                    device)
    print(f"\n\nIDENTITY MODEL loss: {identity_loss}, score: {identity_score}\n\n")
    plotstream.plot_horizontal_line(f"loss {dp.y_dataset}", identity_loss, f"Identity")
    # plotstream.plot_horizontal_line(f"score {dp.y_dataset}", identity_score, f"Identity")

    return identity_loss, identity_score




def process_config_params(cp):
    for key, value in cp.items():
        print(f"{key}: {value}")


def process_data_params(dp):
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

    if dp.whichfold >= 0:
        data_folded = [data_folded[dp.whichfold]]
        print(f"Using ONLY fold {dp.whichfold} of {len(data_folded)}")

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

    return data_folded, testdata, dense_columns, sparse_columns