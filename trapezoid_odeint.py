import torch

def odeint(func, y0, t):
    """
    Solves the ODE using the trapezoid rule. This method is differentiable using PyTorch's autograd.

    Args:
        func: The function defining the ODE, dy/dt = func(t, y).
        y0: Initial value (tensor) at t[0].
        t: 1D tensor of timesteps at which to evaluate the solution.

    Returns:
        A tensor of the same shape as t, containing the solution at each time step.
    """
    y = y0.unsqueeze(0)  # Add batch dimension to y0
    ys = [y0]  # Store the solution at each time step
    
    for i in range(1, len(t)):
        t0, t1 = t[i - 1], t[i]
        dt = t1 - t0
        
        # Evaluate the ODE function at the start and end of the interval
        f0 = func(t0, y.squeeze(0)).unsqueeze(0)  # Shape of (batch, *y.shape)
        y1_pred = y + dt * f0  # Forward Euler step to predict y1
        f1 = func(t1, y1_pred.squeeze(0)).unsqueeze(0)  # Evaluate function at the end of the interval
        
        # Trapezoid step update
        y = y + dt / 2 * (f0 + f1)  # Trapezoid rule step
        ys.append(y.squeeze(0))  # Remove batch dimension
    
    return torch.stack(ys, dim=0)  # Stack solutions along the time dimension