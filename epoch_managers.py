from abc import ABC, abstractmethod


class EpochManager(ABC):
    @abstractmethod
    def should_stop(self, epoch, metrics):
        pass

    @abstractmethod
    def get_metric(self):
        pass


class FixedManager(EpochManager):
    def __init__(self, max_epochs):
        self.max_epochs = max_epochs
        self.epoch = 0
    
    def should_stop(self, metrics):
        self.epoch += 1
        return self.epoch >= self.max_epochs
    
    def get_metric(self):
        return self.epoch / self.max_epochs


class MovingAverageEpochManager(EpochManager):
    def __init__(self, memory, threshold, mode='rel', criterion='earlystop', min_epochs=0, max_epochs=None):
        self.epoch = 0
        self.min_epochs = min_epochs
        self.max_epochs = max_epochs
        self.memory = memory
        self.threshold = threshold
        self.mode = mode
        self.criterion = criterion
        self.ema = None
        self.initial_score = None
        self.best_score = float('inf')
        self.best_raw_score = float('inf')
    
    def update_ema(self, value):
        if self.ema is None:
            self.ema = value
        else:
            self.ema = (1 - self.memory) * value + self.memory * self.ema
        return self.ema
    
    def should_stop(self, metrics):
        self.epoch += 1
        
        # choose score metric based on criterion
        current_score = metrics.val_loss if (self.criterion == 'earlystop') else metrics.trn_loss
        
        # tracked stats
        ema_score = self.update_ema(current_score)
        self.best_score = min(self.best_score, ema_score)
        if self.initial_score is None:
            self.initial_score = current_score
        
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
            return ema_score > reference + reference*self.threshold
        elif self.mode == 'rel_start':
            return ema_score > reference + self.initial_score*self.threshold
        else:
            return ema_score > reference + self.threshold
        
    def get_metric(self):
        return self.ema


class EarlyStopManager(MovingAverageEpochManager):
    def __init__(self, memory, threshold, mode='rel', min_epochs=0, max_epochs=None):
        super().__init__(memory, threshold, mode, criterion='earlystop', min_epochs=min_epochs, max_epochs=max_epochs)


class DivergenceManager(MovingAverageEpochManager):
    def __init__(self, memory, threshold, mode='rel', min_epochs=0, max_epochs=None):
        super().__init__(memory, threshold, mode, criterion='divergence', min_epochs=min_epochs, max_epochs=max_epochs)


class ExplosionManager(MovingAverageEpochManager):
    def __init__(self, memory, threshold, mode='rel', min_epochs=0, max_epochs=None):
        super().__init__(memory, threshold, mode, criterion='explosion', min_epochs=min_epochs, max_epochs=max_epochs)


class ConvergenceManager(EpochManager):
    def __init__(self, memory, threshold, mode='rel', min_epochs=0, max_epochs=None):
        self.epoch = 0
        self.min_epochs = min_epochs
        self.max_epochs = max_epochs
        self.memory = memory
        self.threshold = threshold
        self.mode = mode
        self.ema = None
        self.best_score = float('inf')
        self.prev_score = None
    
    def update_ema(self, value):
        if self.ema is None:
            self.ema = value
        else:
            self.ema = (1 - self.memory) * value + self.memory * self.ema
        return self.ema
    
    def should_stop(self, metrics):
        self.epoch += 1
        
        current_score = metrics.trn_loss
        
        if self.prev_score is None:
            self.prev_score = current_score
            return False # Can't compute slope until 2nd epoch. Inconsequential bug: if max_epochs is 1, this will cause it to run 2 epochs instead.
            
        # tracked stats
        ema_score = self.update_ema(current_score - self.prev_score)
        self.prev_score = current_score
        self.best_score = min(self.best_score, ema_score)
        
        # min/max epochs
        if self.epoch < self.min_epochs:
            return False
        if self.max_epochs is not None and self.epoch >= self.max_epochs:
            return True
        
        if self.mode == 'rel':
            result = ema_score > self.best_score * (1.0 + self.threshold)
        else:
            result = ema_score > self.best_score + self.threshold
        
        return result
        
    def get_metric(self):
        return self.ema


if __name__ == "__main__":
    class Metrics:
        def __init__(self, trn_loss, val_loss):
            self.trn_loss = trn_loss
            self.val_loss = val_loss
    
    # manager = FixedHalter(max_epochs=10)
    # manager = MovingAverageEpochManager(memory=0.7, threshold=0.05, mode='rel', criterion='earlystop', min_epochs=3, max_epochs=15)
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