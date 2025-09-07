import torch
from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau

class ConstantLR(torch.optim.lr_scheduler.LRScheduler): 
    """
    Constant learning rate scheduler.
    """
    
    def __init__(self, optimizer, last_epoch=-1):
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]


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
    

class DirectToZero(torch.optim.lr_scheduler.LRScheduler):
    def __init__(self, optimizer, peak_lr, update_steps, warmup_proportion=0.1):
        
        self.optimizer = optimizer
        self.peak_lr = peak_lr
        self.update_steps = update_steps
        self.warmup_proportion = warmup_proportion
        self.warmup_steps = int(update_steps * warmup_proportion)

        self.stepnum = -1
        
        super().__init__(optimizer)

    def step(self):
        self.stepnum += 1

        if self.stepnum < self.warmup_steps:
            lr = self.peak_lr * ((self.stepnum / self.warmup_steps) if self.warmup_steps > 0 else 1.0)
        else:
            lr = self.peak_lr * (1 - (self.stepnum - self.warmup_steps) / (self.update_steps - self.warmup_steps))
        lr = max(lr, 0.0)
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def get_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]
    


class LRScheduler:
    def __init__(self, scheduler, initial_lr=0.01, update_interval=None):
        self.scheduler = scheduler

        # If the scheduler is ReduceLROnPlateau, we need to report the LR manually before first step has occurred.
        self.has_stepped = False
        self.initial_lrs = initial_lr
        
        # # For OneCycleLR and ExponentialLR (I suspect it is any scheduler that uses get_lr() to perform the update step), we need to undo the first LR step that for some reason occurs before the first batch.
        # if not isinstance(scheduler, ReduceLROnPlateau):
        #     scheduler.optimizer.param_groups[0]['lr'] = initial_lr  # TODO: Figure out how to fix this without messing up models that have multiple LR values in different param_groups.
        
        if update_interval:
            self.update_interval = update_interval
        else:
            if isinstance(scheduler, ReduceLROnPlateau):
                self.update_interval = 'epoch'
            else:
                self.update_interval = 'batch'
    
    def batch_step(self, metrics=None):
        if self.update_interval == 'batch':
            self.step(metrics)
            
    def epoch_step(self, metrics=None):
        if self.update_interval == 'epoch':
            self.step(metrics)
    
    def step(self, metrics=None):
        if isinstance(self.scheduler, ReduceLROnPlateau):
            self.scheduler.step(metrics)
        else:
            self.scheduler.step()
        self.has_stepped = True
    
    def get_lr(self):
        return self.scheduler.get_lr()
    
    def get_last_lr(self):
        if not self.has_stepped:# and isinstance(self.scheduler, ReduceLROnPlateau):
            return self.initial_lrs
        return self.scheduler.optimizer.param_groups[0]['lr']
    

class WDScheduler:
    def __init__(self, scheduler, update_interval=None):
        self.scheduler = scheduler

        if update_interval:
            self.update_interval = update_interval
        else:
            self.update_interval = 'batch'

    def batch_step(self):
        if self.update_interval == 'batch':
            self.scheduler.step()

    def epoch_step(self):
        if self.update_interval == 'epoch':
            self.scheduler.step()

    def step(self):
        self.scheduler.step()

    def get_wd(self):
        return self.scheduler.get_wd()
    
    def get_last_wd(self):
        return self.scheduler.optimizer.param_groups[0]['weight_decay']

class WDConstant:
    def __init__(self, optimizer, wd):
        self.optimizer = optimizer
        self.wd = wd

    def step(self):
        for param_group in self.optimizer.param_groups:
            param_group['weight_decay'] = self.wd


class WDZeroWarmup:
    def __init__(self, optimizer, wd, update_steps, warmup_proportion=0.1):
        self.optimizer = optimizer
        self.wd = wd
        self.update_steps = update_steps
        self.warmup_proportion = warmup_proportion

    def step(self):
        # zero during warmup, wd after warmup
        if self.update_steps < self.warmup_proportion * self.update_steps:
            for param_group in self.optimizer.param_groups:
                param_group['weight_decay'] = 0.0
        else:
            for param_group in self.optimizer.param_groups:
                param_group['weight_decay'] = self.wd


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
