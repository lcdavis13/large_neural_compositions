from abc import ABC, abstractmethod
import math




def get_epoch_manager_constructors():
    # Epoch managers
    # TODO: make all of these args accessible to command line
    epoch_mngr_constructors = {
        "Fixed": lambda args: FixedManager(max_epochs=args.adjusted_epochs),
        "AdaptiveValPlateau": lambda args: AdaptiveValPlateauManager(memory=0.75, rate_threshold_factor=0.05, min_epochs=args.min_epochs, max_epochs=args.adjusted_epochs, patience=args.patience),
    }
    return epoch_mngr_constructors





#
# # holt winters
# # single exponential smoothing
# from statsmodels.tsa.holtwinters import SimpleExpSmoothing
# # double and triple exponential smoothing
# from statsmodels.tsa.holtwinters import ExponentialSmoothing

class EpochManager(ABC):
    """Parent class for epoch managers"""
    
    @abstractmethod
    def should_stop(self, epoch, metrics):
        pass
    
    @abstractmethod
    def get_metric(self):
        pass

    @abstractmethod
    def get_threshold(self):
        pass

    @abstractmethod
    def get_supplemental(self):
        pass

    @abstractmethod
    def set_baseline(self, metrics):
        pass


class FixedManager(EpochManager):
    """Manager that runs for a fixed number of epochs"""
    
    def __init__(self, max_epochs):
        self.max_epochs = max_epochs
        self.epoch = 0
    
    def should_stop(self, metrics):
        self.epoch += 1
        return self.epoch >= self.max_epochs
    
    def get_metric(self):
        return self.epoch
    
    def get_threshold(self):
        return self.max_epochs
    
    def get_supplemental(self):
        return {}
    
    def set_baseline(self, metrics):
        return


def is_finite_number(number):
    return number and math.isfinite(number) and not math.isnan(number)


class MovingAverageEpochManager(EpochManager):
    def __init__(self, memory, threshold, mode='rel', criterion='earlystop', min_epochs=0, max_epochs=None):
        self.epoch = 0
        self.min_epochs = min_epochs
        self.max_epochs = max_epochs
        self.memory = memory
        self.threshold = threshold
        self.mode = mode
        self.criterion = criterion
        self.ema1 = None
        self.ema2 = None
        self.initial_score = None
        self.inf_score = 1e10 ## not infinity to ensure I can end experiment if we reach infinity
        self.best_score = self.inf_score
        self.best_raw_score = self.inf_score
    
    def update_dema(self, value):
        if self.ema1 is None:
            self.ema1 = value
        else:
            self.ema1 = (1 - self.memory) * value + self.memory * self.ema1
        
        if self.ema2 is None:
            self.ema2 = self.ema1
        else:
            self.ema2 = (1 - self.memory) * self.ema1 + self.memory * self.ema2
        
        dema = 2 * self.ema1 - self.ema2
        return dema
    
    def should_stop(self, metrics):
        self.epoch += 1
        
        # choose score metric based on criterion
        current_score = metrics["val_loss"] if (self.criterion == 'earlystop') else metrics["trn_loss"]
        
        # tracked stats
        dema_score = self.update_dema(current_score)
        self.best_score = min(self.best_score, dema_score)
        if self.initial_score is None:
            self.initial_score = current_score
            
        # abort in case of infinity
        if (not is_finite_number(current_score)) or current_score > self.inf_score or current_score < -self.inf_score:
            return True
        
        # min/max epochs
        if self.epoch < self.min_epochs:
            return False
        if self.max_epochs is not None and self.epoch >= self.max_epochs:
            return True
        
        # check stopping criterion
        if self.criterion == 'divergence' or self.criterion == 'earlystop':
            reference = self.best_score
        elif self.criterion == 'explosion':
            reference = self.initial_score
        else:
            return True  # default to stopping in case of unknown criterion
        
        if self.mode == 'rel':
            return dema_score > reference + reference * self.threshold
        elif self.mode == 'rel_start':
            return dema_score > reference + self.initial_score * self.threshold
        else:
            return dema_score > reference + self.threshold
    
    def get_metric(self):
        if self.ema1 is None:
            return None
        return 2.0 * self.ema1 - self.ema2
    
    def get_threshold(self):
        return self.threshold
    
    def get_supplemental(self):
        return {}
    


class EarlyStopManager(MovingAverageEpochManager):
    """Manager that stops training when the validation loss starts increasing"""
    
    def __init__(self, memory, threshold, mode='rel', min_epochs=0, max_epochs=None):
        super().__init__(memory, threshold, mode, criterion='earlystop', min_epochs=min_epochs, max_epochs=max_epochs)


class TrainDivergenceManager(MovingAverageEpochManager):
    """Manager that stops training when the training loss starts increasing"""
    
    def __init__(self, memory, threshold, mode='rel', min_epochs=0, max_epochs=None):
        super().__init__(memory, threshold, mode, criterion='divergence', min_epochs=min_epochs, max_epochs=max_epochs)


class TrainExplosionManager(MovingAverageEpochManager):
    """Manager that stops training when the training loss becomes worse than its starting value"""
    
    def __init__(self, memory, threshold, mode='rel', min_epochs=0, max_epochs=None):
        super().__init__(memory, threshold, mode, criterion='explosion', min_epochs=min_epochs, max_epochs=max_epochs)


class TrainConvergenceManager(EpochManager):
    """Manager that stops training when the rate of training loss decrease becomes low"""
    
    def __init__(self, memory, threshold, mode='rel', min_epochs=0, max_epochs=None):
        self.epoch = 0
        self.min_epochs = min_epochs
        self.max_epochs = max_epochs
        self.memory = memory
        self.threshold = threshold
        self.mode = mode
        self.ema1 = None
        self.ema2 = None
        self.inf_score = 1e10 ## not infinity to ensure I can end experiment if we reach infinity
        self.best_score = self.inf_score
        self.prev_score = None
        self.initial_score = None
    
    def update_dema(self, value):
        if self.ema1 is None:
            self.ema1 = value
        else:
            self.ema1 = (1 - self.memory) * value + self.memory * self.ema1
        
        if self.ema2 is None:
            self.ema2 = self.ema1
        else:
            self.ema2 = (1 - self.memory) * self.ema1 + self.memory * self.ema2
        
        dema = 2 * self.ema1 - self.ema2
        return dema
    
    def should_stop(self, metrics):
        self.epoch += 1
        
        current_score = metrics["trn_loss"]
        
        if self.initial_score is None:
            self.initial_score = current_score
        
        if self.prev_score is None:
            self.prev_score = current_score
            return False  # Can't compute slope until 2nd epoch. Inconsequential bug: if max_epochs is 1, this will cause it to run 2 epochs instead.
        
        # abort in case of infinity
        if (not is_finite_number(current_score)) or current_score > self.inf_score or current_score < -self.inf_score:
            return True
        
        # tracked stats
        dema_score = self.update_dema(current_score - self.prev_score)
        self.prev_score = current_score
        self.best_score = min(self.best_score, dema_score)
        
        # min/max epochs
        if self.epoch < self.min_epochs:
            return False
        if self.max_epochs is not None and self.epoch >= self.max_epochs:
            return True
        
        if self.mode == 'rel':
            result = dema_score > self.best_score * (1.0 + self.threshold)
        elif self.mode == 'abs':
            result = dema_score > self.best_score + self.threshold
        elif self.mode == 'rel_start':
            result = dema_score > self.best_score + self.initial_score * self.threshold
        elif self.mode == "const":
            result = dema_score > self.threshold
        
        return result
    
    def get_metric(self):
        if self.ema1 is None:
            return None
        return 2.0 * self.ema1 - self.ema2
    
    def get_threshold(self):
        return self.threshold
    
    def get_supplemental(self):
        return {}


class AdaptiveValPlateauManager(EpochManager):
    """Manager that stops training when the rate of (smoothed) validation loss decrease becomes low relative to its highest value"""
    
    def __init__(self, memory, rate_threshold_factor, min_epochs=0, max_epochs=None, patience=0):
        self.epoch = 0
        self.min_epochs = min_epochs
        self.max_epochs = max_epochs
        self.memory = memory
        self.best_rate = 0.0
        self.threshold = 0.0
        self.rate_threshold_factor = rate_threshold_factor
        self.patience = patience
        self.failed_times = 0
        self.inf_score = 1e10 ## not infinity to ensure I can end experiment if we reach infinity
        self.rate = None
        self.ema = None
        self.prev_ema = None
        
    def set_baseline(self, metrics):
        self.ema = metrics["val_loss"]
        self.prev_ema = self.ema
    
    def update_ema(self, value):
        if self.ema is None:
            self.ema = value
        else:
            self.ema = (1.0 - self.memory) * value + self.memory * self.ema
        
        return self.ema
    
    def should_stop(self, metrics):
        self.epoch += 1
        
        current_score = metrics["val_loss"]
        
        # abort in case of infinity
        if (not is_finite_number(current_score)) or current_score > self.inf_score or current_score < -self.inf_score:
            print(f"Stopping due to infinite score {current_score}")
            return True
        
        # tracked stats
        self.update_ema(current_score)
        if self.prev_ema:
            self.rate = self.ema - self.prev_ema
            self.best_rate = min(self.best_rate, self.rate)
            self.threshold = self.best_rate * self.rate_threshold_factor
            cant_eval_yet = False
        else:
            cant_eval_yet = True
        self.prev_ema = self.ema
        
        # min/max epochs
        if self.max_epochs is not None and self.epoch >= self.max_epochs:
            print(f"Stopping after {self.epoch} epochs due to max_epochs")
            return True
        if cant_eval_yet or (self.min_epochs is not None and self.epoch < self.min_epochs):
            return False
        
        failed = self.rate > self.threshold

        if failed:
            self.failed_times += 1
        else:
            self.failed_times = 0
        
        result = self.failed_times >= self.patience

        if result:
            print(f"Stopping after {self.epoch} epochs, rate={self.rate}, threshold={self.threshold}, best_rate={self.best_rate}, failed_times={self.failed_times}, patience={self.patience}")

        return result
    
    def get_metric(self):
        return self.rate
    
    def get_threshold(self):
        return self.threshold
    
    def get_supplemental(self):
        return {"val_EMA": self.ema}


class AdaptiveValDEMAPlateauManager(EpochManager):
    """Manager that stops training when the rate of (smoothed) validation loss decrease becomes low relative to its highest value"""
    
    def __init__(self, memory, rate_threshold_factor, min_epochs=0, max_epochs=None, patience=0):
        self.epoch = 0
        self.min_epochs = min_epochs
        self.max_epochs = max_epochs
        self.memory = memory
        self.best_rate = 0.0
        self.threshold = 0.0
        self.rate_threshold_factor = rate_threshold_factor
        self.patience = patience
        self.failed_times = 0
        self.inf_score = 1e10 ## not infinity to ensure I can end experiment if we reach infinity
        self.rate = None
        self.ema1 = None
        self.ema2 = None
        self.dema = None
        self.prev_dema = None
        
    def set_baseline(self, metrics):
        self.ema1 = metrics["val_loss"]
        self.ema2 = metrics["val_loss"]
        self.dema = 2.0 * self.ema1 - self.ema2
        self.prev_dema = self.dema
    
    def update_dema(self, value):
        if self.ema1 is None:
            self.ema1 = value
        else:
            self.ema1 = (1.0 - self.memory) * value + self.memory * self.ema1
        
        if self.ema2 is None:
            self.ema2 = self.ema1
        else:
            self.ema2 = (1.0 - self.memory) * self.ema1 + self.memory * self.ema2
        
        self.dema = 2.0 * self.ema1 - self.ema2
        return self.dema
    
    def should_stop(self, metrics):
        self.epoch += 1
        
        current_score = metrics["val_loss"]
        
        # abort in case of infinity
        if (not is_finite_number(current_score)) or current_score > self.inf_score or current_score < -self.inf_score:
            print(f"Stopping due to infinite score {current_score}")
            return True
        
        # tracked stats
        self.update_dema(current_score)
        if self.prev_dema:
            self.rate = self.dema - self.prev_dema
            self.best_rate = min(self.best_rate, self.rate)
            self.threshold = self.best_rate * self.rate_threshold_factor
            cant_eval_yet = False
        else:
            cant_eval_yet = True
        self.prev_dema = self.dema
        
        # min/max epochs
        if self.max_epochs is not None and self.epoch >= self.max_epochs:
            print(f"Stopping after {self.epoch} epochs due to max_epochs")
            return True
        if cant_eval_yet or (self.min_epochs is not None and self.epoch < self.min_epochs):
            return False
        
        failed = self.rate > self.threshold

        if failed:
            self.failed_times += 1
        else:
            self.failed_times = 0
        
        result = self.failed_times >= self.patience

        if result:
            print(f"Stopping after {self.epoch} epochs, rate={self.rate}, threshold={self.threshold}, best_rate={self.best_rate}, failed_times={self.failed_times}, patience={self.patience}")

        return result
    
    def get_metric(self):
        return self.rate
    
    def get_threshold(self):
        return self.threshold
    
    def get_supplemental(self):
        return {"val_DEMA": self.dema}


if __name__ == "__main__":
    class Metrics:
        def __init__(self, trn_loss, val_loss):
            self.trn_loss = trn_loss
            self.val_loss = val_loss
    
    
    manager = EarlyStopManager(memory=0.7, threshold=0.05, mode='rel', min_epochs=3, max_epochs=15)
    
    # mock training loop
    train_scores = [0.5, 0.4, 0.45, 0.43, 0.3, 0.2, 0.1, 0.05, 0.04, 0.03, 0.02, 0.01, 0.005, 0.002, 0.001]
    val_scores = [0.6, 0.5, 0.55, 0.53, 0.4, 0.3, 0.2, 0.3, 0.34, 0.33, 0.32, 0.31, 0.305, 0.302, 0.301]
    while True:
        metrics = Metrics(trn_loss=train_scores[manager.epoch], val_loss=val_scores[manager.epoch])
        if manager.should_stop(metrics):
            break
    print(f"Done after {manager.epoch} epochs")

# TO DO: add max and min epochs to all of the halters except FixedHalter
# Migrate my run_epochs early stopping to use this instead