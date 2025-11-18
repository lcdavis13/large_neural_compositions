import torch.nn as nn
from contextlib import contextmanager


    # with ode_safe(model):        # freezes BN, resets LockedDropout masks once
    #     if use_deq:
    #         x_star, info = deq_solve(g_map_or_func, x0)   # iterations, no t
    #     else:
    #         traj = heun_odeint(func, x0, t)               # sequential calls with t


@contextmanager
def ode_safe(model):
    # Save states
    bn_layers, bn_training = [], []
    ld_layers = []
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            bn_layers.append(m); bn_training.append(m.training); m.eval()  # freeze stats
        if isinstance(m, LockedDropout):
            ld_layers.append(m); m.reset_state()  # new mask per solve
    try:
        yield
    finally:
        # Restore BN training flags
        for m, was_train in zip(bn_layers, bn_training):
            m.train(was_train)


class LockedDropout(nn.Module):
    """
    LockedDropout applies the same dropout mask across the temporal dimension, allowing stable ODE solves for either IVP (torchdiffeq/my Heun solver) or DEQ (torchdeq).

    Usage:
    with ode_safe(model):
        y = model(x)
    """
    def __init__(self, p=0.1):
        super().__init__()
        self.p = float(p)
        self._mask = None
        self.training = True

    def reset_state(self):
        self._mask = None  # call once per ODE solve / DEQ solve

    def forward(self, x):
        if not self.training or self.p <= 0.0:
            return x
        # one mask per sample, shared across features (broadcast)
        if (self._mask is None) or (self._mask.size(0) != x.size(0)):
            keep = 1.0 - self.p
            shape = (x.size(0), *([1] * (x.dim() - 1)))
            self._mask = x.new_empty(shape).bernoulli_(keep).div_(keep)
        return x * self._mask
