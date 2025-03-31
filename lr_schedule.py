import torch
import math
from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau


class ConstantLR(torch.optim.lr_scheduler.LRScheduler): 
    """
    Constant learning rate scheduler.
    """
    
    def __init__(self, optimizer, last_epoch=-1):
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]


class LinearLR(torch.optim.lr_scheduler.LRScheduler):
    
    def __init__(self, optimizer, epoch_lr_delta, steps_per_epoch, last_epoch=-1):
        self.step_lr_delta = epoch_lr_delta / steps_per_epoch
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        return [group['lr'] + self.step_lr_delta for group in self.optimizer.param_groups]
    

class ExponentialLR(torch.optim.lr_scheduler.LRScheduler):
    """
    Exponentially multiplies the learning rate by a fixed factor per epoch.
    steps_per_epoch is the number of times lr_scheduler.get_lr() will be called per epoch,
    used to calculate the LR factor per step.
    Note that the epoch_lr_factor and steps_per_epoch don't strictly need to correspond to one epoch; the two numbers
    just need to be consistent. The define the factor and the unit of time at which the factor is relevant.
    """
    
    def __init__(self, optimizer, epoch_lr_factor, steps_per_epoch, last_epoch=-1):
        self.step_lr_factor = epoch_lr_factor ** (1.0 / steps_per_epoch)
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        return [group['lr'] * self.step_lr_factor for group in self.optimizer.param_groups]
    

class TransformerLR(torch.optim.lr_scheduler.LRScheduler):
    def __init__(self, optimizer, warmup_steps, peak_lr, last_epoch=-1):
        self.warmup_steps = warmup_steps
        # self.d_model = d_model
        self.peak_lr = peak_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = max(self.last_epoch, 0.0001)  # Avoid zero division
        # scale = self.d_model ** -0.5
        scale = self.peak_lr
        lr = scale * min(step ** -0.5, step * (self.warmup_steps ** -1.5))
        return [lr for _ in self.optimizer.param_groups]
    

class WarmupDecayPlateauLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps, final_lr, peak_lr=None, 
                 decay_halflife_steps=None, last_epoch=-1, start_lr=1e-8):
        """
        Args:
            optimizer (Optimizer): Wrapped optimizer.
            warmup_steps (int): Number of steps for warm-up.
            start_lr (float): Learning rate at the beginning of warm-up.
            final_lr (float): Learning rate at the end of training.
            peak_lr (float, optional): Peak learning rate before decay. If None, no decay occurs.
            decay_halflife_steps (int, optional): Number of steps for the learning rate to decay by half. If None, no decay occurs.
            last_epoch (int): The index of last epoch. Default: -1.
        """
        self.warmup_steps = warmup_steps
        self.start_lr = start_lr
        self.final_lr = final_lr
        if decay_halflife_steps is not None and peak_lr is not None:
            self.peak_lr = peak_lr
            self.decay_rate = math.log(2) / decay_halflife_steps
        else:
            self.peak_lr = final_lr
            self.decay_rate = 1.0

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch + 1  # step starts from 1

        if step <= self.warmup_steps:
            # Linear warm-up
            return [self.start_lr + (self.peak_lr - self.start_lr) * (step / self.warmup_steps)
                    for _ in self.base_lrs]
        
        # elif self.peak_lr is not None and self.decay_rate is not None:
        elif self.peak_lr != self.final_lr and self.decay_rate != 1.0:
            # Exponential decay after warm-up
            return [self.final_lr + (self.peak_lr - self.final_lr) * math.exp(-self.decay_rate * (step - self.warmup_steps))
                    for _ in self.base_lrs]
        
        else:
            # Constant final learning rate (no decay)
            return [self.final_lr for _ in self.base_lrs]
    

class RationalLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, peak_lr, warmup_steps, wd, 
                 last_epoch=-1, start_lr=1e-8):
        """
        Schedule-free method based on https://openreview.net/pdf?id=hrOlBgHsMI#page=15&zoom=100,110,466
        After a warmup, decay LR in such a manner that the same hyperparameters are optimal regardless of training duration. 
        This requires the optimizer to be AdamW and the WD must be known at construction.

        Recurrent definition from paper is lr_(i+1) <- lr_i / (1 + lr_i*wd)
        Closed form: lr_i = lr_0 / (1 + i*lr_0*wd)
        """
        self.warmup_steps = warmup_steps
        self.start_lr = start_lr
        self.peak_lr = peak_lr
        self.wd = wd
        self.current_lr = start_lr

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch + 1  # step starts from 1

        if step <= self.warmup_steps:
            # Linear warm-
            lr = self.start_lr + (self.peak_lr - self.start_lr) * (step / self.warmup_steps)
            return [lr for _ in self.base_lrs]
        
        else:
            # WD-dependent decay
            i = step - self.warmup_steps
            lr = self.peak_lr / (1 + i * self.peak_lr * self.wd) 
            return [lr for _ in self.base_lrs]
      
        

class Serializable:
    """Mixin class that provides automatic state_dict and load_state_dict, including nested Serializable objects."""
    
    def state_dict(self):
        """Recursively serializes all attributes, detecting Serializable objects automatically."""
        state = {}
        for key, value in vars(self).items():
            if isinstance(value, Serializable):
                state[key] = value.state_dict()  # Recursively save state_dict
            else:
                state[key] = value
        return state

    def load_state_dict(self, state_dict):
        """Recursively restores all attributes, detecting Serializable objects automatically."""
        for key, value in state_dict.items():
            if isinstance(getattr(self, key, None), Serializable):
                getattr(self, key).load_state_dict(value)  # Recursively load state_dict
            else:
                setattr(self, key, value)


class _BaseScheduler(Serializable):
    def __init__(self):
        raise NotImplementedError

    def compute_lr_factor(self, step):
        """
        Computes the learning rate factor for a given step.
        Should be implemented by subclasses.
        """
        raise NotImplementedError


class LinearScheduler(_BaseScheduler):
    """
    linear schedule from `start_factor` to `end_factor` over `total_steps`.
    """
    def __init__(self, start_factor, end_factor, total_steps):
        self.start_factor = start_factor
        self.end_factor = end_factor
        self.total_steps = total_steps

    def compute_lr_factor(self, step):
        if step >= self.total_steps:
            return self.end_factor
        progress = step / self.total_steps
        return self.start_factor + (self.end_factor - self.start_factor) * progress
    

class CosineScheduler(_BaseScheduler):
    """
    cosine schedule from `start_factor` to `end_factor` over `total_steps`.
    """
    def __init__(self, start_factor, end_factor, total_steps):
        self.start_factor = start_factor
        self.end_factor = end_factor
        self.total_steps = total_steps

    def compute_lr_factor(self, step):
        if step >= self.total_steps:
            return self.end_factor
        progress = step / self.total_steps
        return self.start_factor + (self.end_factor - self.start_factor) * (1 + math.cos(math.pi * progress)) / 2


class LRScheduleWrapper(Serializable):
    def __init__(self, optimizer, base_scheduler, peak_lr, max_steps, warmup_step_proportion, decay_step_proportion):
        """
        LR scheduler wrapper class that applies:
        1. Linear warmup
        2. An arbitrary scheduler provided as a _BaseScheduler
        3. Linear cooldown

        warmup_step_proportion is specified as proportion of max_steps
        decay_step_proportion is specified as proportion of non-warmup steps (and will adjust dynamically if early stopping occurs)
        """
        assert isinstance(base_scheduler, _BaseScheduler), "base_scheduler must be a subclass of _BaseScheduler"

        self.optimizer = optimizer
        self.base_scheduler = base_scheduler
        self.peak_lr = peak_lr
        self.decay_step_proportion = decay_step_proportion

        self.warmup_steps = int(max_steps * warmup_step_proportion)
        self.decay_steps = int((max_steps - self.warmup_steps) * decay_step_proportion)
        self.train_steps = max_steps - self.warmup_steps - self.decay_steps
        self.max_steps = max_steps

        self.warmup_scheduler = LinearScheduler(start_factor=0.0, end_factor=1.0, total_steps=self.warmup_steps+1) # adding 1 to avoid zero LR
        self.decay_scheduler = LinearScheduler(start_factor=1.0, end_factor=0.0, total_steps=self.decay_steps+1) # adding 1 to avoid maintaining the last LR for a single step after we want to start linear decay

        self.step_count = 0
        self.last_lr = 0.0

        self.initial_lrs = [group['lr'] for group in optimizer.param_groups]

        self.finished_training = False  # Flag to indicate when base scheduler is finished
        self.stopping = False  # Flag to indicate when early stopping has been initiated
        self.final_base_factor = 1.0  # Stores the last LR factor from base scheduler before decay
        self.total_steps = max_steps # If early stopping is initiated, this will be updated to reflect a number smaller than max_steps
    
    def get_lr_factor(self, step):
        """Computes the global LR scaling factor for a given step."""
        if step <= self.warmup_steps:
            return self.warmup_scheduler.compute_lr_factor(step + 1) * self.peak_lr # adding 1 to avoid zero LR

        elif step <= self.warmup_steps + self.train_steps:
            base_factor = self.base_scheduler.compute_lr_factor(step - self.warmup_steps)
            if base_factor <= 0.0:
                self.finished_training = True
            return base_factor

        elif step <= self.max_steps:
            self.stopping = True
            if not self.finished_training:
                self.final_base_factor = self.base_scheduler.compute_lr_factor(self.train_steps)
                self.finished_training = True
            decay_factor = self.decay_scheduler.compute_lr_factor(step + 1 - self.warmup_steps - self.train_steps) # adding 1 to avoid maintaining the last LR for a single step after we want to start linear decay
            return self.final_base_factor * decay_factor
        
        else:
            self.finished_training = True
            return 0.0

    def step(self):
        """Updates the optimizer's learning rates."""
        self.step_count += 1
        self.last_lr = self.get_lr_factor(self.step_count)

        for param_group, initial_lr in zip(self.optimizer.param_groups, self.initial_lrs):
            param_group['lr'] = initial_lr * self.last_lr

    def get_last_lr(self):
        return self.last_lr
    
    def begin_early_stopping(self):
        """
        Initiate cooldown process earlier than planned. 
        If restoring a backup model before cooldown, restore a corresponding backup of this class before calling this method.
        """

        if self.stopping:
            return # Already stopping
        self.stopping = True

        # recalculate step counts
        self.train_steps = self.step_count - self.warmup_steps
        self.decay_steps = int(self.train_steps * self.decay_step_proportion)
        self.total_steps = self.step_count + self.decay_steps

        self.decay_scheduler = LinearScheduler(start_factor=1.0, end_factor=0.0, total_steps=self.decay_steps+1) # adding 1 to avoid maintaining the last LR for a single step after we want to start linear decay



# Example usage
def main():
    epochs = 5
    steps_per_epoch = 10
    
    model = torch.nn.Linear(10, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # choose a scheduler
    scheduler = OneCycleLR(optimizer, max_lr=0.1, epochs=epochs, steps_per_epoch=steps_per_epoch)
    # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=1, threshold_mode='rel', threshold=0.01)
    # scheduler = ExponentialLR(optimizer, epoch_lr_factor=10.0, steps_per_epoch=steps_per_epoch)
    
    lr_scheduler = LRScheduler(scheduler, initial_lr=0.01)

    # Mock validation loss for ReduceLROnPlateau
    validation_losses = [0.5, 0.4, 0.45, 0.43, 0.3]
    
    # if not isinstance(scheduler, ReduceLROnPlateau):
    print(f"Start, LR: {lr_scheduler.get_last_lr()}")
    for epoch in range(epochs):
        for step in range(steps_per_epoch):
            optimizer.step()
            lr_scheduler.batch_step()
            # if epoch % epochs > 0 or not isinstance(scheduler, ReduceLROnPlateau):
            print(f"Step: {step}, LR: {lr_scheduler.get_last_lr()}")
        lr_scheduler.epoch_step(metrics=validation_losses[epoch])
        print(f"Epoch: {epoch}, LR: {lr_scheduler.get_last_lr()}")
    

if __name__ == "__main__":
    main()
