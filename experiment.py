
import itertools
import math
import os
import traceback
import numpy as np
import pandas as pd
import chunked_dataset
from introspection import construct
import lr_schedule
import data
import torch
import models
import models_fitted
import stream
from optimum import Optimum, unrolloptims_dict
import stream_plot as plotstream
import time


import tracemalloc
tracemalloc.start()


def eval_model(model, x, timesteps, ids, max_dropped_timesteps=0):
    # evaluates models whether they require ODE timesteps or not
    requires_timesteps = getattr(model, 'USES_ODEINT', False)
    requires_condensed = getattr(model, 'USES_CONDENSED', False)

    # randomly drop some final timesteps if allowed
    if max_dropped_timesteps > 0 and requires_timesteps:
        #randomly select number between one and max_dropped_timesteps
        early_stop = np.random.randint(1, max_dropped_timesteps+1)
        timesteps = timesteps[:-early_stop]

    if requires_timesteps and timesteps is not None:
        # Call models that require the timesteps argument and return dydt in addition to y
        if requires_condensed:
            y_steps = model(timesteps, x, ids)
        else:
            y_steps = model(timesteps, x)
    else:
        # Call models that do not require the timesteps argument
        dydt = None
        if requires_condensed:
            y = model(x, ids) 
        else:
            y = model(x)
        y_steps = [[y, None]]
    
    return y_steps


def ceildiv(a, b):
    return -(a // -b)


def validate_epoch(model, requires_condensed, data, t, hp, loss_fn, score_fns, device):
    model.eval()

    # score_fns_2 = score_fns.copy()
    # score_fns_2["loss"] = loss_fn

    totals = {name: 0.0 for name in score_fns}
    total_samples = 0

    for mb in data:
        x, y, x_sparse, y_sparse, ids = mb[chunked_dataset.DK_X], mb[chunked_dataset.DK_Y], mb[chunked_dataset.DK_XSPARSE], mb[chunked_dataset.DK_YSPARSE], mb[chunked_dataset.DK_IDS]
        
        if requires_condensed:
            x = x_sparse.to(device)
            y = y_sparse.to(device)
            ids = ids.to(device)
        else:
            x = x.to(device)
            y = y.to(device)

        mb_examples = x.size(0)
        total_samples += mb_examples

        with torch.no_grad():
            y_pred, dydt = eval_model(model, x, t, ids)[-1]
            y_pred = y_pred.to(device)
            dydt = dydt.to(device) if dydt is not None else None

            for name, fn in score_fns.items():
                val = fn(y_pred, y, hp=hp, dydt=dydt)
                totals[name] += val.item() * mb_examples

    avg_scores = {name: total / total_samples for name, total in totals.items()}
    return avg_scores



def run_test(model, requires_condensed, hp, model_name, model_config, epoch, mini_epoch, data_test, t, fold, loss_fn, score_fns, device, filepath_out_test, gpu_memory_reserved, cpuRam, elapsed_time, train_score=-1.0, val_score=-1.0):
    test_scores = validate_epoch(model=model, requires_condensed=requires_condensed, data=data_test, t=t, hp=hp, loss_fn=loss_fn, score_fns=score_fns, device=device)
    stream.stream_results(filepath_out_test, True, True, True,
                            "model_name", model_name,
                            "model_config", model_config,
                            "fold", fold,
                            "epoch", epoch,
                            "mini-epoch", mini_epoch,
                            *unrolldict(test_scores),
                            "Avg Train Score", train_score,
                            "Avg Validation Score", val_score,
                            "Elapsed Time", elapsed_time,
                            "VRAM (GB)", gpu_memory_reserved / (1024 ** 3),
                            "Peak RAM (GB)", cpuRam / (1024 ** 3),
                            prefix="==================TESTING=================\n",
                            suffix="\n=============================================\n")
    return test_scores

def is_finite_number(number):
    return torch.all(torch.isfinite(number)) and not torch.any(torch.isnan(number))

def train_mini_epoch(model, requires_condensed, epoch_data_iterator, data_train, hp, mini_epoch_size, full_epoch_size, minibatch_examples, accumulated_minibatches, noise, interpolate, interpolate_noise, optimizer, scheduler, wd_scheduler, scaler, t,
                outputs_per_mini_epoch, 
                prev_examples, prev_updates, fold, epoch_num, mini_epoch_num, config_label, dataname, loss_fn, device,
                filepath_out_incremental, number_converged_timesteps, lr_plot=None, loss_plot=None, lr_loss_plot=None, verbosity=1, supervised_timesteps=1):
    model.train()
    
    total_loss = 0

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
        
        y_pred, dydt = eval_model(model, x, t, ids, max_dropped_timesteps=0)[-1] 
        #number_converged_timesteps)
        # if supervise_steps:
        #     y_pred = y_pred[1:]
        #     y_true = z[1:]  # TODO: Fix: implement supervised interpolation with new data source
        # else:
        #     y_pred = y_pred[-1:]
        #     y_true = y.unsqueeze(0)  # TODO: may not need this unsqueeze if we stop supporting supervised interpolation
        # y_pred = y_preds[-number_converged_timesteps:]
        # y_true = y.unsqueeze(0).expand(number_converged_timesteps, -1, -1)
        y_true = y
        # print(f"y_pred shape: {y_pred.shape}, y_true shape: {y_true.shape}")
        y_pred = y_pred.to(device)
        dydt = dydt.to(device) if dydt is not None else None

        loss = loss_fn(y_pred, y_true, hp=hp, dydt=dydt)
        actual_loss = loss.item() * mb_examples
        loss = loss / accumulated_minibatches  # Normalize the loss by the number of accumulated minibatches, since loss function can't normalize by this
        
        scaled_loss = scaler.scale(loss)
        if scaled_loss.requires_grad and scaled_loss.grad_fn is not None:
            scaled_loss.backward()
        else:
            print(f"GRADIENT ERROR: Loss at epoch {epoch_num} mini-epoch {mini_epoch_num} minibatch {mb} does not require gradient. Computation graph detached?")
        
        # del y_pred, loss, distr_error
        
        total_loss += actual_loss
        new_examples += mb_examples
        new_updates += 1
        
        stream_loss += actual_loss
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
                                  "Examples per second", examples_per_second,
                                  "Learning Rate", scheduler.get_last_lr(),
                                  "Weight Decay", wd_scheduler.get_last_wd(),
                                  )
            if lr_plot:
                plotstream.plot_single(lr_plot, "mini-epochs", "LR", f"{config_label} fold {fold}",
                                       mini_epoch_num + mb / minibatches, scheduler.get_last_lr(), False, y_log=False)
            if loss_plot:
                plotstream.plot_loss(loss_plot, f"{config_label} fold {fold}", mini_epoch_num + mb / minibatches,
                                     stream_loss / stream_examples, None, add_point=False, xlabel='Update Steps', ylabel='Bray-Curtis Loss')
            if lr_loss_plot:
                plotstream.plot_single(lr_loss_plot, "log( Learning Rate )", "Loss", f"{config_label} fold {fold}",
                                       scheduler.get_last_lr(), stream_loss / stream_examples, False, x_log=True)
            stream_loss = 0
            prev_time = end_time
            stream_examples = 0
        
        if ((mb + 1) % accumulated_minibatches == 0) or (mb == minibatches - 1):
            scaler.step(optimizer)
            scaler.update()
            scheduler.batch_step()  # TODO: Add accum_loss metric in case I ever want to do ReduceLROnPlateau with batch_step mode
            wd_scheduler.batch_step()
            optimizer.zero_grad()
        
        # del x, y
    


    avg_loss = total_loss / new_examples  # This used to be divided by total_samples which is dataset size, that's wrong isn't it?
    new_total_examples = prev_examples + new_examples
    new_total_updates = prev_updates + new_updates
    return avg_loss, new_total_examples, new_total_updates, epoch_num, epoch_data_iterator


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


def run_epochs(model, requires_condensed, optimizer, scheduler, wd_scheduler, manager, hp, minibatch_examples, accumulated_minibatches, noise, interpolate, interpolate_noise, scaler, data_train, data_valid, data_test, t,
               model_name, model_config, config_label, dataname, fold, loss_fn, score_fns, device, mini_epoch_size, full_epoch_size, outputs_per_epoch,
               preeval_training_set, reeval_train_epoch, reeval_train_final, jobstring, use_best_model, export_cnode, number_converged_timesteps, verbosity=1, plot_all_scores=True):
    # assert(data.check_leakage([(x_train, y_train, x_valid, y_valid)]))

    epoch = 0
    current_sample_index = 0

    empty_scores = {key: None for key in score_fns}
    empty_scores["loss"] = None
    
    # track stats at various definitions of the "best" epoch
    val_opt = Optimum('val_score', 'min')
    trn_opt = Optimum('trn_score', 'min')
    
    old_lr = scheduler.get_last_lr()
    
    filepath_out_epoch = f'results/epochs/{model_name}_{dataname}{jobstring}_epochs.csv'
    filepath_out_test = f'results/tests/{dataname}{jobstring}_tests.csv'
    # filepath_out_model = f'results/logs/{model_config}_{dataname}_model.pth'
    filepath_out_incremental = f'results/incr/{model_config}_{dataname}{jobstring}_incremental.csv'
    
    # initial validation benchmark
    print("Evaluating initial validation score")
    val_scores = validate_epoch(model=model, requires_condensed=requires_condensed, data=data_valid, t=t, hp=hp, loss_fn=loss_fn, score_fns=score_fns, device=device)
    if preeval_training_set:
        print("Evaluating initial training score")
        trn_scores = validate_epoch(model=model, requires_condensed=requires_condensed, data=data_train, t=t, hp=hp, loss_fn=loss_fn, score_fns=score_fns, device=device)
    else:
        trn_scores = empty_scores

    val_and_trn_scores = flatten_dict({"val": val_scores, "trn": trn_scores})

    if device.type == 'cuda':
        gpu_memory_reserved = torch.cuda.memory_reserved(device)
    else:
        gpu_memory_reserved = 0
    _, cpuRam = tracemalloc.get_traced_memory()
    stream.stream_results(filepath_out_epoch, verbosity > 0, verbosity > 0, verbosity > -1,
                          "fold", fold,
                          "epoch", 0,
                          "mini-epoch", 0,
                          "training examples", 0,
                          "update steps", 0,
                          *unrolldict(val_and_trn_scores),
                          "Learning Rate", old_lr,
                          "Elapsed Time", 0.0,
                          "VRAM (GB)", gpu_memory_reserved / (1024 ** 3),
                          "Peak RAM (GB)", cpuRam  / (1024 ** 3),
                          prefix="================PRE-VALIDATION===============\n",
                          suffix="\n=============================================\n")
    
    if plot_all_scores:
        for score_name in score_fns:
            plotstream.plot_loss(f"{score_name} on dataset {dataname}", f"{model_config} fold {fold}", 0, trn_scores[score_name], val_scores[score_name], add_point=False, y_log=True)
    else:
        plotstream.plot_loss(f"Loss on dataset {dataname}", f"{model_config} fold {fold}", 0, trn_scores["loss"], val_scores["loss"], add_point=False, y_log=True)
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
    
    # training_curve = []
    # has_backup = False

    # TODO: replace this quick and dirty dict packing. They should have always been in a dict.
    dict = {"epoch": epoch, "mini_epoch": manager.epoch, 
            "lr": old_lr, "time": start_time, "gpu_memory": gpu_memory_reserved,
            "stop_metric": manager.get_metric(), "stop_threshold": manager.get_threshold()}
    dict.update(val_and_trn_scores)
    # track various optima
    model_path = f'results/models/{model_config}_{dataname}_fold{fold}_job{jobstring}.pt'
    if use_best_model:
        val_opt.track_best(dict, model=model, model_path=model_path)
    else:
        val_opt.track_best(dict)
    trn_opt.track_best(dict)
    manager.set_baseline(dict)

    epoch_data_iterator = iter(data_train)

    print("Starting training")

    while True:
        trn_loss, train_examples_seen, update_steps, epoch, epoch_data_iterator = train_mini_epoch(
            model=model, requires_condensed=requires_condensed, epoch_data_iterator=epoch_data_iterator, data_train=data_train, hp=hp, mini_epoch_size=mini_epoch_size, full_epoch_size=full_epoch_size, minibatch_examples=minibatch_examples,
            accumulated_minibatches=accumulated_minibatches, noise=noise, interpolate=interpolate, interpolate_noise=interpolate_noise, optimizer=optimizer, scheduler=scheduler, wd_scheduler=wd_scheduler, scaler=scaler, t=t,
            outputs_per_mini_epoch=outputs_per_epoch, prev_examples=train_examples_seen, prev_updates=update_steps, 
            fold=fold, epoch_num=epoch, mini_epoch_num=manager.epoch, config_label=config_label, dataname=dataname, loss_fn=loss_fn, device=device, filepath_out_incremental=filepath_out_incremental,
            lr_plot=f"Learning Rate {dataname}", verbosity=verbosity - 1, number_converged_timesteps=number_converged_timesteps
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
        
        val_scores = validate_epoch(model=model, requires_condensed=requires_condensed, data=data_valid, t=t, hp=hp, loss_fn=loss_fn, score_fns=score_fns, device=device)
        
        if reeval_train_epoch:
            # print("Re-evaluating training score")
            trn_scores = validate_epoch(model=model, requires_condensed=requires_condensed, data=data_train, t=t, hp=hp, loss_fn=loss_fn, score_fns=score_fns, device=device)
        else:
            trn_scores = empty_scores
            trn_scores["loss"] = trn_loss
        
        val_and_trn_scores = flatten_dict({"val": val_scores, "trn": trn_scores})  
        
        # Update learning rate based on loss
        scheduler.epoch_step()  #(l_trn)
        wd_scheduler.epoch_step()
        new_lr = scheduler.get_last_lr()
        lr_changed = not np.isclose(new_lr, old_lr)
        add_point = lr_changed and isinstance(scheduler.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
        
        current_time = time.time()
        elapsed_time = current_time - start_time
        if device.type == 'cuda':
            gpu_memory_reserved = torch.cuda.memory_reserved(device)
        else:
            gpu_memory_reserved = 0
        _, cpuRam = tracemalloc.get_traced_memory()
        
        # TODO: replace this quick and dirty dict packing. They should have always been in a dict.
        dict = {"epoch": epoch, "mini_epoch": manager.epoch, 
                "lr": old_lr, "time": start_time, "gpu_memory": gpu_memory_reserved,
                "stop_metric": manager.get_metric(), "stop_threshold": manager.get_threshold()}
        dict.update(val_and_trn_scores)
        # track various optima
        model_path = f'results/models/{model_config}_{dataname}_fold{fold}_job{jobstring}.pt'
        if use_best_model:
            val_opt.track_best(dict, model=model, model_path=model_path)
        else:
            val_opt.track_best(dict)
        trn_opt.track_best(dict)
        
        # training_curve.append({"fold": fold, "epoch":epoch, "mini_epoch": manager.epoch, "trn_loss": l_trn, "val_loss": l_val, "time": elapsed_time})
        
        old_lr = new_lr
        
        # update epoch manager, which decides if we should stop or continue
        should_stop = manager.should_stop(dict)
        
        # log results (after updating manager so we can log stats from the manager itself)
        stream.stream_results(filepath_out_epoch, verbosity > 0, verbosity > 0, verbosity > -1,
                              "fold", fold,
                              "epoch", epoch,
                              "mini-epoch", manager.epoch,
                              "training examples", train_examples_seen,
                              "update steps", update_steps,
                              *unrolldict(val_and_trn_scores),
                              "Learning Rate", old_lr,  # should I track average LR in the epoch? Max and min LR?
                              "Elapsed Time", elapsed_time,
                              "VRAM (GB)", gpu_memory_reserved / (1024 ** 3),
                              "Peak RAM (GB)", cpuRam / (1024 ** 3),
                              prefix="==================VALIDATION=================\n",
                              suffix="\n=============================================\n")
        
        if plot_all_scores:
            for score_name in score_fns:
                plotstream.plot_loss(f"{score_name} on dataset {dataname}", f"{model_config} fold {fold}", manager.epoch if mini_epoch_size <= 0 else update_steps, trn_scores[score_name], val_scores[score_name],
                                    add_point=add_point, xlabel='Epoch' if mini_epoch_size <= 0 else 'Update Steps', ylabel=score_name, y_log=True)
        else:
            plotstream.plot_loss(f"Loss on dataset {dataname}", f"{model_config} fold {fold}", manager.epoch if mini_epoch_size <= 0 else update_steps, trn_scores["loss"], val_scores["loss"],
                                add_point=add_point, xlabel='Epoch' if mini_epoch_size <= 0 else 'Update Steps', ylabel='Bray-Curtis Loss', y_log=True)
        # plotstream.plot_loss(f"score {dataname}", f"{model_config} fold {fold}", manager.epoch, score_trn, score_val, add_point=add_point)
        plotstream.plot(f"stopmetric {dataname}", "mini-epoch", "metric", [f"metric {model_config} fold {fold}", f"threshold {model_config} fold {fold}"], manager.epoch, [manager.get_metric(), manager.get_threshold()], add_point=False)
        # plotstream.plot(f"Validation Loss EMA {dataname}", "mini-epoch", "metric", [f"val_EMA {model_config} fold {fold}"], manager.epoch, [manager.get_supplemental()["val_EMA"]], add_point=False) # Commented because constant epoch manager doesn't have an EMA
        # plotstream.plot(dataname, "epoch", "loss", [f"{model_config} fold {fold} - Val", f"{model_config} fold {fold} - Trn", f"{model_config} fold {fold} - DKI Val", f"{model_config} fold {fold} - DKI Trn"], manager.epoch + 1, [l_val, l_trn, score_val, score_trn], add_point=add_point)
        # if l_val != score_val:
        #     print("WARNING: CURRENT LOSS METRIC DISAGREES WITH DKI LOSS METRIC")

        # time to stop: optionally load model and run test.
        test_scores = empty_scores
        if should_stop:
            print("===Stopping training===")
            if data_test or reeval_train_final:
                if use_best_model:
                    model.load_state_dict(torch.load(model_path, weights_only=True))
                if export_cnode and model_name == "cNODE1":
                    print("Exporting trained cNODE parameters")
                    weight = model.func.fcc1.weight.detach().cpu().numpy()
                    bias = model.func.fcc1.bias
                    if bias is not None:
                        bias = bias.detach().cpu().numpy()
                    else:
                        bias = np.zeros(weight.shape[0])
                    f = f'results/models/{model_config}_{dataname}_fold{fold}_job{jobstring}_cnode_weights.csv'
                    f2 = f'results/models/{model_config}_{dataname}_fold{fold}_job{jobstring}_cnode_bias.csv'
                    print(f"Exporting cNODE weights to {f} and bias to {f2}")
                    pd.DataFrame(weight).to_csv(f, index=False, header=False)
                    pd.DataFrame(bias).to_csv(f2, index=False, header=False)
                if reeval_train_final:
                    print("Re-evaluating final training score")
                    trn_scores = validate_epoch(model=model, requires_condensed=requires_condensed, data=data_train, t=t, hp=hp, loss_fn=loss_fn, score_fns=score_fns, device=device)
                    print(f"Final Training Loss: {trn_scores["loss"]}, Final Validation Loss: {val_scores["loss"]}")
                if data_test:
                    print("Running test")
                    test_scores = run_test(model=model, requires_condensed=requires_condensed, hp=hp, model_name=model_name, model_config=model_config, epoch=val_opt.epoch, mini_epoch=val_opt.mini_epoch, data_test=data_test, t=t, fold=fold, loss_fn=loss_fn, score_fns=score_fns, device=device, filepath_out_test=filepath_out_test, gpu_memory_reserved=gpu_memory_reserved, cpuRam=cpuRam, elapsed_time=elapsed_time, train_score=trn_scores["loss"], val_score=val_scores["loss"])
            break
    
    all_scores = flatten_dict({"val": val_scores, "trn": trn_scores, "test": test_scores})

    final_dict = {"epoch": epoch, "mini_epoch": manager.epoch, 
            "lr": old_lr, "time": start_time, "gpu_memory": gpu_memory_reserved,
            "stop_metric": manager.get_metric(), "stop_threshold": manager.get_threshold()}
    final_dict.update(all_scores)

    return final_dict, val_opt, trn_opt


def train_model(data_train, data_valid, data_test,
                cp, dp, hp, 
                fold_num, 
                model_class, epoch_manager_constr, model_args, 
                scaler, device, 
                timesteps, loss_fn, score_fns,
                jobstring, verbosity=1):

    # (data_train, data_valid, data_test, 
    # LR, scaler, accumulated_minibatches, total_train_samples, noise, interpolate, interpolate_noise, device, kfolds,
    # max_epochs, mini_epoch_size, 
    # minibatch_examples, model_constr, epoch_manager_constr, model_args, model_name, model_config, dataname, timesteps, loss_fn,
    # score_fn, distr_error_fn, weight_decay, preeval_training_set, reeval_train_epoch, reeval_train_final,
    # jobstring, use_best_model, verbosity=1):

        
    epoch_size = hp.mini_epoch_size if hp.mini_epoch_size > 0 else dp.total_train_samples
    stepsize = dp.minibatch_examples * hp.accumulated_minibatches
    steps_per_epoch = ceildiv(epoch_size, stepsize)
    final_epoch_minsamples = (hp.epochs % 1.0) * epoch_size 
    final_epoch_minibatches = ceildiv(final_epoch_minsamples, dp.minibatch_examples)
    update_steps = (hp.epochs // 1.0)*steps_per_epoch + final_epoch_minibatches
    
    # print("==========================")
    # print(f"total_train_samples: {epoch_size}")
    # print(f"step size: {stepsize}")
    # print(f"steps per epoch: {steps_per_epoch}")
    # print(f"final epoch min samples: {final_epoch_minsamples}")
    # print(f"final epoch minibatches: {final_epoch_minibatches}")
    # print(f"update steps: {update_steps}")
    # print("==========================")

    
    # LR_start_factor = 0.1 # OneCycle
    # LR_start_factor = 1.0  # constantLR
    LR_start_factor = 0.0 # warm up

    model = construct(model_class, model_args).to(device)
    start_wd = hp.wd if hp.wd_during_warmup else 0.0
    optimizer = torch.optim.AdamW(model.parameters(), lr=hp.lr * LR_start_factor, weight_decay=start_wd)

    manager = epoch_manager_constr(model_args)
    # optimizer = torch.optim.SGD(model.parameters(), lr=LR*LR_start_factor, weight_decay=weight_decay)
    # manager = epoch_managers.FixedManager(max_epochs=min_epochs)
    # manager = epoch_managers.ExplosionManager(memory=0.5, threshold=1.0, mode="rel", max_epochs=max_epochs)
    # manager = epoch_managers.EarlyStopManager(memory=0.0, threshold=0.0, mode="rel", max_epochs=max_epochs)
    # manager = epoch_managers.ConvergenceManager(memory=0.1, threshold=0.001, mode="const", min_epochs=min_epochs, max_epochs=max_epochs)
    
    # x_train, y_train, x_valid, y_valid = data_fold
    requires_condensed = getattr(model, 'USES_CONDENSED', False)
    

    # print(f"Steps per epoch: {steps_per_epoch}")
    # base_scheduler = lr_schedule.ConstantLR(optimizer)
    # base_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=patience // 2, cooldown=patience,
    #                                    threshold_mode='rel', threshold=0.01)
    # base_scheduler = OneCycleLR(
    #     optimizer, max_lr=LR, epochs=min_epochs, steps_per_epoch=steps_per_epoch, div_factor=1.0/LR_start_factor,
    #     final_div_factor=1.0/(LR_start_factor*0.1), three_phase=True, pct_start=0.4, anneal_strategy='cos')
    base_scheduler = lr_schedule.DirectToZero(optimizer, peak_lr=hp.lr, update_steps=update_steps, warmup_proportion=0.1)
    scheduler = lr_schedule.LRScheduler(base_scheduler, initial_lr=hp.lr * LR_start_factor)

    if hp.wd_during_warmup:
        base_wd_scheduler = lr_schedule.WDConstant(optimizer, wd=hp.wd)
    else:
        base_wd_scheduler = lr_schedule.WDZeroWarmup(optimizer, wd=hp.wd, update_steps=update_steps, warmup_proportion=0.1)
    wd_scheduler = lr_schedule.WDScheduler(base_wd_scheduler)

    print(f"Fold {fold_num + 1}/{dp.kfolds}")
    
    final_dict, val_opt, trn_opt = run_epochs(
        model=model, requires_condensed=requires_condensed, optimizer=optimizer, scheduler=scheduler, wd_scheduler=wd_scheduler, manager=manager, hp=hp, minibatch_examples=dp.minibatch_examples, accumulated_minibatches=hp.accumulated_minibatches,
        noise=hp.noise, interpolate=hp.interpolate, interpolate_noise=hp.interpolate_noise, scaler=scaler, data_train=data_train, data_valid=data_valid, data_test=data_test,
        t=timesteps, model_name=hp.model_name, model_config=hp.model_config, config_label=hp.config_label, dataname=dp.y_dataset, fold=fold_num, loss_fn=loss_fn, score_fns=score_fns,
        device=device, mini_epoch_size=hp.mini_epoch_size, full_epoch_size=dp.total_train_samples, 
        outputs_per_epoch=10, verbosity=verbosity - 1,
        # reptile_rate=reptile_rewind,
        preeval_training_set=hp.preeval_training_set, reeval_train_epoch=hp.reeval_training_set_epoch, reeval_train_final=hp.reeval_training_set_final,
        jobstring=jobstring, 
        use_best_model=hp.use_best_model, 
        export_cnode=hp.export_cnode, 
        number_converged_timesteps=hp.number_converged_timesteps, 
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


    fold_stats_dict = unrolloptims_dict(val_opt, trn_opt)
    fold_stats_dict.update(final_dict)

    other_stats_dict = {}

    return fold_stats_dict, other_stats_dict
    # can't return training_curves in this format or it will try to print to results stream. Do we want to return it at all? 
    # purpose was to find the truly optimal epoch across folds, but averaging the epoch across folds might be better for avoiding overfitting anyway.
    # return val_opt, valscore_opt, trn_opt, trnscore_opt, last_opt, training_curve, test_score


def crossvalidate(fit_and_evaluate_fn, data_folded, data_test, score_fns, whichfold, filepath_out_fold, filepath_out_expt, out_rowinfo_dict, required_scores=None, verbosity=1):
    """
    Run cross-validation on the given data folds. This is fully agnostic to the model, training regime, etc.
    
    fit_and_evaluate_fn: function to fit and evaluate the model.
        - Accepts data_train, data_valid, data_test, info_dict, and verbosity as arguments. Other args should be passed using lambdas in the calling context.
        - Returns a tuple of dictionaries: fold_stats_dict, other_stats_dict.
            - fold_stats_dict: statistics that should be averaged across folds (e.g. loss, score, epochs).
    format_name: optional suffix for the output file names. Use this if there are different possible result formats returned from fit_and_evaluate_fn.
        - e.g. directly fitted models (linear regression etc) might report fewer statistics than trained models.
    out_keys_dict: dictionary of keys to be output in the results. This is used to ensure that the same keys are used across all folds.
    """
    
    
    fold_stat_dicts = []
    other_stat_dicts = []
    fold_valid = []
    valid_folds = 0

    for fold_num, data_fold in enumerate(data_folded):
        if whichfold >= 0:
            fold_num = whichfold
        
        data_train = data_fold[0]
        data_valid = data_fold[1]

        fold_stats_dict, other_stats_dict = fit_and_evaluate_fn(data_train=data_train, data_valid=data_valid, data_test=data_test, 
                                                                fold_num=fold_num, score_fns=score_fns, verbosity=verbosity-1)
        
        fold_stat_dicts.append(fold_stats_dict)
        other_stat_dicts.append(other_stats_dict)

        # check validity of all values in fold_stats_dict
        valid = True
        if required_scores is not None:
            for k in required_scores:
                v = fold_stats_dict.get(k)
                if not isinstance(v, (int, float)):
                    print(f"WARNING: Fold {fold_num}, key '{k}': invalid type ({type(v).__name__})")
                    valid = False
                elif not math.isfinite(v):
                    print(f"WARNING: Fold {fold_num}, key '{k}': non-finite value ({v})")
                    valid = False
        fold_valid.append(valid)
        if valid:
            valid_folds += 1
        else:
            print(f"WARNING: Fold {fold_num} is invalid. Not including in mean and stddev.")

        
        stream.stream_results(filepath_out_fold, verbosity-1 > 0, verbosity-1 > 0, verbosity-1 > -1,
                              "fold_num", fold_num, 
                              "fold_valid", valid,
                              *unrolldict(out_rowinfo_dict),
                              *unrolldict(fold_stats_dict),
                              *unrolldict(other_stats_dict),
                              prefix="\n========================================FOLD=========================================\n",
                              suffix="\n=====================================================================================\n")

    # compute stats across folds (excluding folds with invalid values for any required keys, and excluding individual invalid values in non-required keys)
    mean_dict = {}
    std_dict = {}
    for key in fold_stat_dicts[0].keys():
        valid_values = [d[key] for d, valid in zip(fold_stat_dicts, fold_valid) if valid]
        valid_values = [v for v in valid_values if v is not None]  # Remove None values. The above filter removes entire folds based on certain required keys, but this filters on an element-by-element basis in case there are Nones in a non-required key.
        mean_key = f"mean_{key}"
        std_key = f"std_{key}"
        mean_dict[mean_key] = np.nanmean(valid_values)
        std_dict[std_key] = np.nanstd(valid_values)
        
    stream.stream_results(filepath_out_expt, verbosity > 0, verbosity > 0, verbosity > -1,
                            *unrolldict(out_rowinfo_dict),
                            "valid_folds", valid_folds,
                            *unrolldict(mean_dict),
                            *unrolldict(std_dict),
                            prefix="\n=====================================EXPERIMENT======================================\n",
                            suffix="\n=====================================================================================\n")
    
    
    # return val_loss_optims, val_score_optims, trn_loss_optims, trn_score_optims, final_optims, val_loss_curves, test_scores
    return mean_dict, std_dict, fold_stat_dicts, other_stat_dicts



def flatten_dict(d, parent_key='', sep='_'):
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else str(k)
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items


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



def run_experiment(cp, dp, hp, data_folded, testdata, device, model_classes, epoch_mngr_constructors, loss_fn, score_fns, benchmark_losses, dense_columns, sparse_columns, plot_all_scores=True):
    # things that are needed for reporting an exception, so they go before the try block
    jobid_substring = int(cp.jobid.split('_')[0])
    jobstring = f"_job{jobid_substring}" if jobid_substring >= 0 else ""
    filepath_out_expt = f'results/expt/{dp.y_dataset}{jobstring}_experiments.csv'
    filepath_out_fold = f'results/folds/{dp.y_dataset}{jobstring}_folds.csv'
    filepath_out_hyperparams = f'results/hp/{dp.y_dataset}{jobstring}_hyperparams.csv'
    num_params = -1

    # if True:
    try:
        if plot_all_scores:
            for score_name in score_fns:
                plotstream.plot_horizontal_lines(f"{score_name} on dataset {dp.y_dataset}", benchmark_losses[score_name])
        else:
            plotstream.plot_horizontal_lines(f"loss on dataset {dp.y_dataset}", benchmark_losses)

        # computed hyperparams
        hp.data_dim = dense_columns
        hp.sparse_data_dim = sparse_columns
        # hp.WD = hp.lr * hp.wd_factor
        # hp.attend_dim = hp.attend_dim_per_head * hp.num_heads
        hp.config_label = f"{hp.model_name}: {hp.config}"
        hp.model_config = f"{hp.model_name} hp-{cp.config_configid}-{dp.data_configid}-{hp.configid}"

        # conditionally adjust epochs to compensate for subset size
        if hp.subset_increases_epochs:
            if hp.base_data_subset > 0:
                hp.adjusted_epochs = int(hp.epochs // (dp.data_subset / hp.base_data_subset))
            else:            
                hp.adjusted_epochs = int(hp.epochs // dp.data_fraction)
            print(f"Adjusted epochs from {hp.epochs} to {hp.adjusted_epochs} to compensate for subset size")
        else:
            hp.adjusted_epochs = hp.epochs

        # assert hp.attend_dim % hp.num_heads == 0, "attend_dim must be divisible by num_heads"
        
        model_class = model_classes[hp.model_name]
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
        
        # Test construction. If using parameter_target, this will find a model configuration with approximately correct parameter count and return the specific config hyperparameters.
        print(f"\nModel construction test for: {hp.model_config}")
        model, config_overrides = models.construct_model_parameterized(model_class, hp.parameter_target, hp.width_depth_tradeoff, hp)
        if config_overrides:
            print(f"Model configuration overrides: {config_overrides}")
            hp.update(config_overrides) 
            # NOTE: This mutates hp in the calling context. Currently that's fine because we only send hp to this function and then construct a new hp on each step through the loop, but if things change, that could be problematic.
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Number of parameters in model: {num_params}")
        
        # print hyperparams
        for key, value in hp.items():
            print(f"{key}: {value}")
        
        # Just for the sake of logging experiments before cross validation...
        stream.stream_scores(filepath_out_hyperparams, True, True, True,
                            "model parameters", num_params,
                            "device", device,
                            "solver", os.environ["SOLVER"],
                            *unrolldict(cp),  # unroll the data params dictionary
                            *unrolldict(dp),  # unroll the data params dictionary
                            *unrolldict(hp),  # unroll the hyperparams dictionary
                            prefix="\n=======================================================HYPERPARAMS=======================================================\n",
                            suffix="\n=========================================================================================================================\n")
        
        
        # train and test the model across multiple folds
        train_model_lambda = lambda data_train, data_valid, data_test, fold_num, score_fns, verbosity: train_model(
            data_train=data_train, data_valid=data_valid, data_test=data_test, 
            fold_num=fold_num, 
            cp=cp, dp=dp, hp=hp, 
            model_class=model_class, epoch_manager_constr=epoch_manager_constr, model_args=hp, 
            scaler=scaler, device=device, 
            timesteps=timesteps, loss_fn=loss_fn, score_fns=score_fns,
            jobstring=jobstring, verbosity=verbosity
        )

        verbosity = 2
        out_rowinfo_dict = {
            "model_name": hp.model_name,
            "model_config": hp.model_config,
            "dataname": dp.y_dataset,
            "data_subset": dp.data_subset,
            "jobid": jobid_substring,
        }
        mean_dict, std_dict, fold_stat_dicts, other_stat_dicts = crossvalidate(fit_and_evaluate_fn=train_model_lambda, 
                                                                               data_folded=data_folded, data_test=testdata, 
                                                                               whichfold=dp.whichfold, score_fns=score_fns, 
                                                                               filepath_out_fold=filepath_out_fold, 
                                                                               filepath_out_expt=filepath_out_expt, 
                                                                               out_rowinfo_dict=out_rowinfo_dict,
                                                                               required_scores=["val_loss", "trn_loss", "val_score", "trn_score"],
                                                                               verbosity=verbosity)
        

        
        # # print all folds
        # print(f'Val Loss optimums: \n{[f'Fold {num}: {opt}\n' for num, opt in enumerate(val_loss_optims)]}\n')
        # print(f'Val Score optimums: \n{[f'Fold {num}: {opt}\n' for num, opt in enumerate(val_score_optims)]}\n')
        # print(f'Trn Loss optimums: \n{[f'Fold {num}: {opt}\n' for num, opt in enumerate(trn_loss_optims)]}\n')
        # print(f'Trn Score optimums: \n{[f'Fold {num}: {opt}\n' for num, opt in enumerate(trn_score_optims)]}\n')
        # print(f'Final optimums: \n{[f'Fold {num}: {opt}\n' for num, opt in enumerate(final_optims)]}\n')
        # print(f'Test scores: \n{[f'Fold {num}: {score}\n' for num, score in enumerate(test_scores)]}\n')
        
        # # calculate fold summaries
        # avg_val_loss_optim = summarize(val_loss_optims)
        # avg_val_score_optim = summarize(val_score_optims)
        # avg_trn_loss_optim = summarize(trn_loss_optims)
        # avg_trn_score_optim = summarize(trn_score_optims)
        # avg_final_optim = summarize(final_optims)

        # # mean of test_scores, which is a list of either numbers or Nones
        # avg_test_score = np.nanmean([score for score in test_scores if score is not None])
        
        # # print summaries
        # print(f'Avg Val Loss optimum: {avg_val_loss_optim}')
        # print(f'Avg Val Score optimum: {avg_val_score_optim}')
        # print(f'Avg Trn Loss optimum: {avg_trn_loss_optim}')
        # print(f'Avg Trn Score optimum: {avg_trn_score_optim}')
        # print(f'Avg Final optimum: {avg_final_optim}')
        # print(f'Avg Test score: {avg_test_score}')
        
        # # find optimal mini-epoch
        # # training_curves is a list of dictionaries, convert to a dataframe
        # all_data = [entry for fold in training_curves for entry in fold]
        # df = pd.DataFrame(all_data)
        # df_clean = df.dropna(subset=['val_loss'])
        # # Check if df_clean is not empty
        # if not df_clean.empty:
        #     average_metrics = df_clean.groupby('mini_epoch').mean(numeric_only=True).reset_index()
        #     min_val_loss_epoch = average_metrics.loc[average_metrics['val_loss'].idxmin()]
        #     best_epoch_metrics = min_val_loss_epoch.to_dict()
        # else:
        #     min_val_loss_epoch = None  # or handle the empty case as needed
        #     best_epoch_metrics = {"epoch": -1, "mini_epoch": -1, "val_loss": -1.0, "trn_loss": -1.0, "val_score": -1.0, "trn_score": -1.0, "time": -1.0}
        #     test_scores = [-1.0] * len(val_loss_optims)
        
        # # write folds to log file
        # for i in range(len(val_loss_optims)):
        #     stream.stream_scores(filepath_out_expt, True, True, True,.
        #                         "optimal early stop val_loss", best_epoch_metrics["val_loss"],
        #                         "optimal early stop epoch", best_epoch_metrics["epoch"],
        #                         "optimal early stop mini-epoch", best_epoch_metrics["mini_epoch"],
        #                         "optimal early stop time", best_epoch_metrics["time"],
        #                         "optimal early stop trn_loss", best_epoch_metrics["trn_loss"],
        #                         "test loss", test_scores[i],
        #                         "model parameters", num_params,
        #                         "fold", i if dp.whichfold < 0 else dp.whichfold,
        #                         "device", device,
        #                         "solver", os.environ["SOLVER"],
        #                         *unrolldict(benchmark_losses),
        #                         *unrolldict(cp),  # unroll the data params dictionary
        #                         *unrolldict(dp),  # unroll the data params dictionary
        #                         *unrolldict(hp),  # unroll the hyperparams dictionary
        #                         *unrolloptims(val_loss_optims[i], val_score_optims[i], trn_loss_optims[i],
        #                                     trn_score_optims[i], final_optims[i]),
        #                         prefix="\n=======================================================EXPERIMENT========================================================\n",
        #                         suffix="\n=========================================================================================================================\n")
        
    except Exception as e:
        # stream.stream_scores(filepath_out_expt, True, True, True,
        #                 "mean_val_loss", -1,
        #                 "mean_val_loss @ epoch", -1,
        #                 "mean_val_loss @ mini-epoch", -1,
        #                 "mean_val_loss @ time", -1,
        #                 "mean_val_loss @ trn_loss", -1,
        #                 "test loss", -1,
        #                 "model parameters", num_params,
        #                 "fold", -1,
        #                 "device", device,
        #                 "solver", os.environ["SOLVER"],
        #                 *unrolldict(benchmark_losses),
        #                 *unrolldict(cp),  # unroll the data params dictionary
        #                 *unrolldict(dp),  # unroll the data params dictionary
        #                 *unrolldict(hp),  # unroll the hyperparams dictionary
        #                 *unrolloptims(val_loss_optims[0], val_score_optims[0], trn_loss_optims[0],
        #                             trn_score_optims[0], final_optims[0]),
        #                 prefix="\n=======================================================EXPERIMENT========================================================\n",
        #                 suffix="\n=========================================================================================================================\n")
        print(f"Model {hp.model_name} failed with error:\n{e}")
        traceback.print_exc()


def run_benchmarks(cp, dp, data_folded, testdata, score_fns, dense_columns, plot_all_scores=True):
    
    benchmark_losses = {}

    if dp.eval_benchmarks:
        mean_id_scores, _, _, _ = crossvalidate(
                    fit_and_evaluate_fn=models_fitted.evaluate_identity_function, 
                    data_folded=data_folded, data_test=testdata, 
                    score_fns=score_fns,
                    whichfold=dp.whichfold, 
                    filepath_out_expt="results/benchmarks/expt.csv",
                    filepath_out_fold="results/benchmarks/fold.csv",
                    out_rowinfo_dict={
                        "model_name": "identity", 
                        "dataset": dp.y_dataset, 
                        "data_subset": dp.data_subset,
                        "data_validation_samples": dp.data_validation_samples,
                        "kfolds": dp.kfolds, 
                        "config_configid": cp.config_configid, 
                        "dataset_configid": dp.data_configid
                    }
                )
                # models_fitted.fit_and_evaluate_linear_regression(data_train, data_valid, data_test, fold_num, score_fn, data_dim, verbosity=0)
        lambda_linreg = lambda data_train, data_valid, data_test, fold_num, score_fns, verbosity: models_fitted.fit_and_evaluate_lr_or_mp(
                    data_train, data_valid, data_test, fold_num, score_fns, dense_columns, verbosity=verbosity)
        mean_lin_scores, _, _, _ = crossvalidate(
                    fit_and_evaluate_fn=lambda_linreg, 
                    data_folded=data_folded, data_test=testdata, 
                    score_fns=score_fns,
                    whichfold=dp.whichfold, 
                    filepath_out_expt="results/benchmarks/expt.csv",
                    filepath_out_fold="results/benchmarks/fold.csv",
                    out_rowinfo_dict={
                        "model_name": "LinearRegression-MP", 
                        "dataset": dp.y_dataset, 
                        "data_subset": dp.data_subset,
                        "data_validation_samples": dp.data_validation_samples,
                        "kfolds": dp.kfolds, 
                        "config_configid": cp.config_configid, 
                        "dataset_configid": dp.data_configid
                    },
                    required_scores=["train_loss", "train_mse"]
                )
        print(mean_id_scores)
        print(mean_lin_scores)
        if plot_all_scores:
            for score_name in score_fns:
                # plotstream.plot_loss(f"{score_name} on dataset {dataname}", f"{model_config} fold {fold}", 0, trn_scores[score_name], val_scores[score_name], add_point=False, y_log=True)
                benchmark_losses[score_name] = {
                        "identity": mean_id_scores[f"mean_valid_{score_name}"],
                        "Linear Regression (Moore-Penrose) - Trn": mean_lin_scores[f"mean_train_{score_name}"],
                        "Linear Regression (Moore-Penrose) - Val": mean_lin_scores[f"mean_valid_{score_name}"],
                }
        else:
            benchmark_losses = {
                        "identity": mean_id_scores["mean_valid_loss"],
                        "Linear Regression (Moore-Penrose) - Trn": mean_lin_scores["mean_train_loss"],
                        "Linear Regression (Moore-Penrose) - Val": mean_lin_scores["mean_valid_loss"],
                        # "LinReg_test": mean_lin_scores["mean_test_loss"],
                    }
        
    return benchmark_losses



def process_config_params(cp):
    for key, value in cp.items():
        print(f"{key}: {value}")


def process_data_params(dp):
    # datasets
    base_filepath = f"data/{dp.x_dataset}/"

    filenames = {
        chunked_dataset.DK_BINARY: f"_binary",
        chunked_dataset.DK_IDS: f"ids-sparse",
        chunked_dataset.DK_X: f"x0", 
        chunked_dataset.DK_XSPARSE: f"x0-sparse",
        chunked_dataset.DK_Y: f"{dp.y_dataset}/{dp.y_dataset}_y",
        chunked_dataset.DK_YSPARSE: f"{dp.y_dataset}/{dp.y_dataset}_y-sparse",
    }

    data_folded, dp.total_train_samples, dp.data_fraction, dense_columns, sparse_columns = chunked_dataset.load_folded_datasets(
        base_filepath, 
        filenames,
        dp.minibatch_examples,
        data_train_samples=dp.data_subset,
        data_validation_samples=dp.data_validation_samples,
        kfolds=dp.kfolds,
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