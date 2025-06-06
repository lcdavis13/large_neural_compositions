import os
import numpy as np
from dotsy import ency
import torch


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

    def track_best(self, dict, model=None, model_path="optimum_model.pth"):
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
            if model is not None:
                # make sure path exists
                folder = os.path.dirname(model_path)
                if folder and not os.path.exists(folder):
                    os.makedirs(folder)

                torch.save(model.state_dict(), model_path)  # Save model parameters
            
        return best
    
    def __str__(self):
        # Start by printing the metric_name first
        result = f"{self.metric_name}: {self.dict.get(self.metric_name)}"
        
        # Add all other keys, prefixing them with '@'
        for key, value in self.dict.items():
            if key != self.metric_name:
                result += f", @ {key}: {value}"
        
        return result


def summarize_statistic(fold_losses):
    model_score = np.mean(fold_losses)
    #model_score = np.max([np.mean(fold_losses), np.median(fold_losses)])  # most pessimistic summary
    return model_score


def summarize(optimum_list):
    if not optimum_list:
        raise ValueError("The list of Optimum objects cannot be empty")
    
    # Assume all Optimum objects have the same metric and metric_type
    metric = optimum_list[0].metric_name
    if metric is None:
        metric = "mini_epoch"
    metric_type = optimum_list[0].metric_type
    
    # Collect keys from the first Optimum dict for summarization
    summary_dict = {}
    
    # We assume all Optimum objects have similar dict keys
    keys = optimum_list[0].dict.keys()
    
    # Summarize for each key
    for key in keys:
        fold_losses = [opt.dict.get(key, None) for opt in optimum_list if opt.dict.get(key) is not None]
        
        if fold_losses:
            summary_dict[key] = summarize_statistic(fold_losses)
    
    # Create a new Optimum object with the summarized results
    return Optimum(metric, metric_type, summary_dict)


def _unroll_items(opt):
    optimized_key = opt.metric_name or "mini_epoch"
    optimized_value = opt.dict.get(optimized_key)

    pairs = []

    # Add the optimized key-value pair first
    if optimized_key in opt.dict:
        pairs.append((optimized_key, optimized_value))

    # Then, add all other key-value pairs with prefix
    for key, value in opt.dict.items():
        if key != optimized_key:
            prefixed_key = f"{optimized_key} @ {key}"
            pairs.append((prefixed_key, value))

    return pairs

def unrolloptims(*optimums):
    unrolled_items = []
    for opt in optimums:
        for key, value in _unroll_items(opt):
            unrolled_items.extend([key, value])
    return unrolled_items

def unrolloptims_dict(*optimums):
    unrolled_dict = {}
    for opt in optimums:
        for key, value in _unroll_items(opt):
            unrolled_dict[key] = value
    return unrolled_dict
