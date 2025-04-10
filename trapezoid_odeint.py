import torch

def odeint(func, y0, t, non_negative=True):
    """
    Solves the ODE using Heun's Method (improved trapezoid rule). This method is differentiable using PyTorch's autograd.

    Args:
        func: The function defining the ODE, dy/dt = func(t, y).
        y0: Initial value (tensor) at t[0].
        t: 1D tensor of timesteps at which to evaluate the solution.
        non_negative: If True, clips negative values to zero.

    Returns:
        Tensor of shape (time_steps, batch_size, state_dim) with the solution.
    """
    y = y0.clone()  # (batch_size, state_dim)
    ys = [y.clone()]  # store initial state

    for i in range(1, len(t)):
        t0, t1 = t[i - 1], t[i]
        dt = t1 - t0

        f0 = func(t0, y)  # (batch_size, state_dim)
        y1_pred = y + dt * f0  # Euler prediction

        f1 = func(t1, y1_pred)  # (batch_size, state_dim)

        y = y + dt / 2.0 * (f0 + f1)  # Heun update

        if non_negative:
            y = torch.clamp(y, min=0.0)  # non-negative constraint

        ys.append(y.clone())

    return torch.stack(ys, dim=0)  # (time_steps, batch_size, state_dim)