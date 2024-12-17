import numpy as np
from dotsy import ency


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


def unrolloptims(*optimums):
    unrolled_items = []
    
    for opt in optimums:
        # First, add the optimized key-value pair
        optimized_key = opt.metric_name
        optimized_value = opt.dict.get(optimized_key)
        
        # Append the optimized key-value pair first
        if optimized_key in opt.dict:
            unrolled_items.append(optimized_key)
            unrolled_items.append(optimized_value)
        
        # Then, append all other key-value pairs, prefixed by the optimized key
        for key, value in opt.dict.items():
            if key != optimized_key:
                prefixed_key = f"{optimized_key} @ {key}"
                unrolled_items.append(prefixed_key)
                unrolled_items.append(value)
    
    return unrolled_items
