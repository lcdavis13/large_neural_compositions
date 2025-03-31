
import numpy as np
import epoch_managers
import lr_schedule
from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau
import data
import torch
import stream
from optimum import Optimum, summarize, unrolloptims
from stream_plot import plotstream
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


def validate_epoch(model, data_val, minibatch_examples, t, loss_fn, score_fn, distr_error_fn, device):
    model.eval()
    
    total_loss = 0.0
    total_score = 0.0
    total_distr_error = 0.0
    total_samples = data_val[0].size(0)
    minibatches = ceildiv(total_samples, minibatch_examples)
    current_index = 0  # Initialize current index to start of dataset
    
    for mb in range(minibatches):
        z, ids, current_index = data.get_batch_raw(data_val, t, minibatch_examples, current_index, noise_level_x=0.0, noise_level_y=0.0, requires_timesteps=False)
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


def run_test(model, model_name, model_config, epoch, mini_epoch, minibatch_examples, data_test, t, fold, loss_fn, score_fn, distr_error_fn, device, filepath_out_test, gpu_memory_reserved, cpuRam, elapsed_time):
    l_test, score_test, p_test = validate_epoch(model, data_test, minibatch_examples, t, loss_fn, score_fn, distr_error_fn, device)
    stream.stream_results(filepath_out_test, True, True, True,
                            "model_name", model_name,
                            "model_config", model_config,
                            "fold", fold,
                            "epoch", epoch,
                            "mini-epoch", mini_epoch,
                            "Avg Test Loss", l_test,
                            "Avg DKI Test Loss", score_test,
                            "Avg Test Distr Error", p_test,
                            "Elapsed Time", elapsed_time,
                            "VRAM (GB)", gpu_memory_reserved / (1024 ** 3),
                            "Peak RAM (GB)", cpuRam / (1024 ** 3),
                            prefix="==================TESTING=================\n",
                            suffix="\n=============================================\n")

def is_finite_number(number):
    return torch.all(torch.isfinite(number)) and not torch.any(torch.isnan(number))

def train_mini_epoch(model, data_train, minibatch_examples, accumulated_minibatches, noise, interpolate, interpolate_noise, optimizer, scheduler, scaler, t,
                outputs_per_mini_epoch, 
                prev_examples, fold, epoch_num, mini_epoch_num, current_index, model_config, dataname, loss_fn, score_fn, distr_error_fn, device,
                filepath_out_incremental, lr_plot=None, loss_plot=None, lr_loss_plot=None, verbosity=1, supervised_timesteps=1, mini_epoch_size=-1):
    model.train()
    
    total_loss = 0
    total_penalty = 0
    total_samples = data_train[0].size(0)

    if mini_epoch_size > 0:
        minibatches = ceildiv(mini_epoch_size, minibatch_examples)
        loop_batches = True
    else:
        minibatches = ceildiv(total_samples, minibatch_examples)
        loop_batches = False
    
    new_examples = 0
    
    stream_interval = max(1, minibatches // outputs_per_mini_epoch)
    
    optimizer.zero_grad()
    
    # set up metrics for streaming
    prev_time = time.time()
    stream_loss = 0
    stream_penalty = 0
    stream_examples = 0
    
    if lr_plot:
        plotstream.plot_single(lr_plot, "mini-epochs", "LR", f"{model_config} fold {fold}",
                                mini_epoch_num, scheduler.get_last_lr(), False, y_log=True)
    
    for mb in range(minibatches):
        model_requires_timesteps = getattr(model, 'USES_ODEINT', False)
        supervise_steps = interpolate and model_requires_timesteps
        
        z, ids, current_index, epoch_num = data.get_batch(data_train, t, minibatch_examples, current_index, total_samples, epoch_num, noise_level_x=noise, noise_level_y=noise, interpolate_noise=interpolate_noise, requires_timesteps=supervise_steps, loop=loop_batches, shuffle=True)
        
        mb_examples = z.shape[-2]
        
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
            print(f"GRADIENT ERROR: Loss at epoch {epoch_num} mini-epoch {mini_epoch_num} minibatch {mb} does not require gradient. Computation graph detached?")
        
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
                                  "epoch", epoch_num,
                                  "mini-epoch", mini_epoch_num,
                                  "minibatch", mb,
                                  "total examples seen", prev_examples + new_examples,
                                  "Avg Loss", stream_loss / stream_examples,
                                  "Avg Distr Error", stream_penalty / stream_examples,
                                  "Examples per second", examples_per_second,
                                  "Learning Rate", scheduler.get_last_lr(),
                                  )
            if lr_plot:
                plotstream.plot_single(lr_plot, "mini-epochs", "LR", f"{model_config} fold {fold}",
                                       mini_epoch_num + mb / minibatches, scheduler.get_last_lr(), False, y_log=True)
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
    
    avg_loss = total_loss / total_samples
    avg_penalty = total_penalty / total_samples
    new_total_examples = prev_examples + new_examples
    return avg_loss, avg_penalty, new_total_examples, epoch_num, current_index


def run_epochs(model, optimizer, scheduler, manager, hp, scaler, data_train, data_valid, data_test, t,
               dataname, fold, loss_fn, score_fn, distr_error_fn, device, outputs_per_epoch=10, verbosity=1,
               reptile_rate=0.0, reeval_train=False, jobstring=""):
    # assert(data.check_leakage([(x_train, y_train, x_valid, y_valid)]))

    epoch = 0
    current_sample_index = 0
    
    # track stats at various definitions of the "best" epoch
    val_opt = Optimum('val_loss', 'min')
    trn_opt = Optimum('trn_loss', 'min')
    valscore_opt = Optimum('val_score', 'min')
    trnscore_opt = Optimum('trn_score', 'min')
    last_opt = Optimum(metric=None)  # metric None to update it every time. metric="epoch" would do the same
    
    scheduler.start()
    old_lr = scheduler.get_last_lr()
    
    filepath_out_epoch = f'results/epochs/{hp.model_name}_{dataname}{jobstring}_epochs.csv'
    filepath_out_test = f'results/tests/{dataname}{jobstring}_tests.csv'
    # filepath_out_model = f'results/logs/{model_config}_{dataname}_model.pth'
    filepath_out_incremental = f'results/incr/{hp.model_config}_{dataname}{jobstring}_incremental.csv'
    
    # initial validation benchmark
    l_val, score_val, p_val = validate_epoch(model, data_valid, hp.minibatch_examples, t, loss_fn, score_fn,
                                             distr_error_fn, device)
    l_trn, score_trn, p_trn = validate_epoch(model, data_train, hp.minibatch_examples, t, loss_fn, score_fn,
                                             distr_error_fn, device)
    
    gpu_memory_reserved = torch.cuda.memory_reserved(device)
    _, cpuRam = tracemalloc.get_traced_memory()
    stream.stream_results(filepath_out_epoch, verbosity > 0, verbosity > 0, verbosity > -1,
                          "fold", fold,
                          "epoch", 0,
                          "mini-epoch", 0,
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
    plotstream.plot_loss(f"loss {dataname}", f"{hp.model_config} fold {fold}", 0, l_trn, l_val, add_point=False)
    # plotstream.plot_loss(f"score {dataname}", f"{model_config} fold {fold}", 0, score_trn, score_val, add_point=False)
    plotstream.plot(f"stopmetric {dataname}", "mini-epoch", "metric", [f"metric {hp.model_config} fold {fold}", f"threshold {hp.model_config} fold {fold}"], manager.epoch, [manager.get_metric(), manager.get_threshold()], add_point=False)
    plotstream.plot(f"Validation Loss EMA {dataname}", "mini-epoch", "metric", [f"val_EMA {hp.model_config} fold {fold}"], manager.epoch, [manager.get_supplemental()["val_EMA"]], add_point=False)
    # plotstream.plot(dataname, "epoch", "loss", [f"{model_config} fold {fold} - Val", f"{model_config} fold {fold} - Trn", f"{model_config} fold {fold} - DKI Val", f"{model_config} fold {fold} - DKI Trn"], 0, [l_val, None, score_val, None], add_point=False)
    
    train_examples_seen = 0
    start_time = time.time()
    
    if reptile_rate > 0.0:
        # Create a copy of the model to serve as the meta-model
        meta_model = copy.deepcopy(model)
        outer_optimizer = type(optimizer)(meta_model.parameters())
        outer_optimizer.load_state_dict(optimizer.state_dict())
        outer_optimizer.lr = reptile_rate
    
    training_curve = []
    has_backup = False

    # TODO: replace this quick and dirty dict packing. They should have always been in a dict.
    dict = {"epoch": epoch, "mini_epoch": manager.epoch, "trn_loss": l_trn, "trn_score": score_trn, "val_loss": l_val,
            "val_score": score_val, "lr": old_lr, "time": start_time, "gpu_memory": gpu_memory_reserved,
            "metric": p_val, "stop_metric": manager.get_metric(), "stop_threshold": manager.get_threshold()}
    # track various optima
    model_path = f'results/models/{hp.model_config}_{dataname}_fold{fold}_job{jobstring}.pt'
    val_opt.track_best(dict, model=model, model_path=model_path)
    valscore_opt.track_best(dict)
    trn_opt.track_best(dict)
    trnscore_opt.track_best(dict)
    last_opt.track_best(dict)
    manager.set_baseline(last_opt)

    while True:
        l_trn, p_trn, train_examples_seen, epoch, current_sample_index = train_mini_epoch(model, data_train, hp.minibatch_examples,
                                                        hp.accumulated_minibatches, hp.noise, hp.interpolate, hp.interpolate_noise, optimizer, scheduler, scaler, t,
                                                        outputs_per_epoch, train_examples_seen,
                                                        fold, epoch, manager.epoch, current_sample_index, hp.model_config, dataname, loss_fn, score_fn,
                                                        distr_error_fn, device, filepath_out_incremental,
                                                        lr_plot="Learning Rate", verbosity=verbosity - 1, mini_epoch_size=hp.mini_epoch_size)
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
        
        l_val, score_val, p_val = validate_epoch(model, data_valid, hp.minibatch_examples, t, loss_fn, score_fn,
                                                 distr_error_fn, device)
        if reeval_train:
            l_trn, score_trn, p_trn = validate_epoch(model, data_train, hp.minibatch_examples, t, loss_fn, score_fn,
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
        
        # TODO: replace this quick and dirty dict packing. They should have always been in a dict.
        dict = {"epoch": epoch, "mini_epoch": manager.epoch, "trn_loss": l_trn, "trn_score": score_trn, "val_loss": l_val,
                "val_score": score_val, "lr": old_lr, "time": elapsed_time, "gpu_memory": gpu_memory_reserved,
                "metric": p_val, "stop_metric": manager.get_metric(), "stop_threshold": manager.get_threshold()}
        # track various optima
        model_path = f'results/models/{hp.model_config}_{dataname}_fold{fold}_job{jobstring}.pt'
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
        plotstream.plot_loss(f"loss {dataname}", f"{hp.model_config} fold {fold}", manager.epoch, l_trn, l_val,
                             add_point=add_point)
        # plotstream.plot_loss(f"score {dataname}", f"{model_config} fold {fold}", manager.epoch, score_trn, score_val, add_point=add_point)
        plotstream.plot(f"stopmetric {dataname}", "mini-epoch", "metric", [f"metric {hp.model_config} fold {fold}", f"threshold {hp.model_config} fold {fold}"], manager.epoch, [manager.get_metric(), manager.get_threshold()], add_point=False)
        plotstream.plot(f"Validation Loss EMA {dataname}", "mini-epoch", "metric", [f"val_EMA {hp.model_config} fold {fold}"], manager.epoch, [manager.get_supplemental()["val_EMA"]], add_point=False)
        # plotstream.plot(dataname, "epoch", "loss", [f"{model_config} fold {fold} - Val", f"{model_config} fold {fold} - Trn", f"{model_config} fold {fold} - DKI Val", f"{model_config} fold {fold} - DKI Trn"], manager.epoch + 1, [l_val, l_trn, score_val, score_trn], add_point=add_point)
        # if l_val != score_val:
        #     print("WARNING: CURRENT LOSS METRIC DISAGREES WITH DKI LOSS METRIC")

        # time to stop: optionally load model and run test.
        if should_stop:
            if data_test:
                model.load_state_dict(torch.load(model_path, weights_only=True))
                run_test(model, hp.model_name, hp.model_config, val_opt.epoch, val_opt.mini_epoch, hp.minibatch_examples, data_test, t, fold, loss_fn, score_fn, distr_error_fn, device, filepath_out_test, gpu_memory_reserved, cpuRam, elapsed_time)
            break
    
    return val_opt, valscore_opt, trn_opt, trnscore_opt, last_opt, training_curve


def crossvalidate_model(hp, dp, scaler, data_folded, testdata, device, 
                        model_constr, epoch_manager_constr, timesteps, loss_fn,
                        score_fn, distr_error_fn, verbosity=1, reptile_rewind=0.0, reeval_train=False,
                        whichfold=-1, jobstring=""):
    filepath_out_fold = f'results/folds/{hp.model_config}_{dp.dataset}{jobstring}_folds.csv'
    
    # LR_start_factor = 0.1 # OneCycle
    # LR_start_factor = 1.0  # everything else
    
    val_loss_optims = []
    val_score_optims = []
    trn_loss_optims = []
    trn_score_optims = []
    final_optims = []
    val_loss_curves = []
    for fold_num, data_fold in enumerate(data_folded):
        if whichfold >= 0:
            fold_num = whichfold
        model = model_constr(hp).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=hp.lr, weight_decay=hp.wd)

        manager = epoch_manager_constr(hp)
        # optimizer = torch.optim.SGD(model.parameters(), lr=hp.lr*LR_start_factor, weight_decay=hp.wd)
        # manager = epoch_managers.FixedManager(max_epochs=min_epochs)
        # manager = epoch_managers.ExplosionManager(memory=0.5, threshold=1.0, mode="rel", max_epochs=max_epochs)
        # manager = epoch_managers.EarlyStopManager(memory=0.0, threshold=0.0, mode="rel", max_epochs=max_epochs)
        # manager = epoch_managers.ConvergenceManager(memory=0.1, threshold=0.001, mode="const", min_epochs=min_epochs, max_epochs=max_epochs)
        
        # x_train, y_train, x_valid, y_valid = data_fold
        requires_condensed = getattr(model, 'USES_CONDENSED', False)
        if not requires_condensed:
            data_train = [data_fold[0][0], data_fold[1][0]]
            data_valid = [data_fold[0][1], data_fold[1][1]]
            if testdata:
                data_test = [testdata[0], testdata[1]]
            else:
                data_test = None
        else:
            data_train = [data_fold[2][0], data_fold[3][0], data_fold[4][0]]
            data_valid = [data_fold[2][1], data_fold[3][1], data_fold[4][1]]
            if testdata:
                data_test = [testdata[2], testdata[3], testdata[4]]
            else:
                data_test = None

        
        steps_per_epoch = ceildiv(hp.mini_epoch_size, hp.minibatch_examples * hp.accumulated_minibatches)
        # base_scheduler = lr_schedule.ConstantLR(optimizer)
        
        # base_scheduler = lr_schedule.TransformerLR(optimizer, steps_per_epoch*warmup_epochs, hp.lr)
        # base_scheduler = lr_schedule.WarmupDecayPlateauLR(optimizer, steps_per_epoch*warmup_epochs, 1e-9, hp.lr, peak_lr=peak_lr, decay_halflife_steps=steps_per_epoch*warmup_epochs/2.0)
        base_scheduler = lr_schedule.DirectToZeroLR(optimizer, hp.lr, warmup_steps=hp.warmup_epochs*steps_per_epoch, decay_steps=(hp.min_epochs - hp.warmup_epochs)*steps_per_epoch)
        # base_scheduler = lr_schedule.RationalLR(optimizer, hp.lr, warmup_steps=hp.warmup_epochs*steps_per_epoch, wd=hp.wd)
        # base_scheduler = OneCycleLR(
        #     optimizer, max_lr=peak_lr, total_steps=steps_per_epoch*(hp.min_epochs+1), div_factor=peak_lr/hp.lr,
        #     final_div_factor=peak_lr/hp.lr/10.0, three_phase=False, pct_start=0.3, anneal_strategy='cos')
        scheduler = lr_schedule.LRScheduler(base_scheduler)
        
        print(f"Fold {fold_num + 1}/{dp.kfolds}")
        
        val_opt, valscore_opt, trn_opt, trnscore_opt, last_opt, training_curve = run_epochs(model, optimizer, scheduler, manager,
                                                                            hp, 
                                                                            scaler, data_train, data_valid, data_test,
                                                                            timesteps, 
                                                                            dp.dataset, fold_num, loss_fn, score_fn,
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
                              prefix="\n========================================FOLD=========================================\n",
                              suffix="\n=====================================================================================\n")
    
    val_loss_curves.append(training_curve)
    
    return val_loss_optims, val_score_optims, trn_loss_optims, trn_score_optims, final_optims, val_loss_curves

