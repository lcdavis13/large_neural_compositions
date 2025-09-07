import torch


def clamp_straight_through(x, min=0.0, max=None):
    y = torch.clamp(x, min=min, max=max)
    # Forward: returns clamped value
    # Backward: gradient wrt x is identity (no clamp effect on grads)
    return x + (y - x).detach()


def odeint(func, y0, t, non_negative=True):
    """
    Heun's method with output shaped (time, 2, batch, state),
    where out[:, 0] = y(t) and out[:, 1] = dy/dt(t).

    No extra func evaluations vs the original two-per-step Heun loop.
    dy/dt at the final timestep uses the predictor derivative to avoid
    one last call; see inline comment if you prefer an exact terminal derivative.
    """
    y = y0.clone()  # (batch, state)
    ys = [y.clone()]

    # Derivative lists same length as t
    dydts = []

    # Initial derivative at (t[0], y0)
    f0 = func(t[0], y)                 # 1st call (matches original)
    dydts.append(f0.clone())

    for i in range(1, len(t)):
        t0, t1 = t[i - 1], t[i]
        dt = t1 - t0

        # Predictor using f0 at (t0, y)
        y1_pred = y + dt * f0

        # Derivative at (t1, y1_pred)
        f1_pred = func(t1, y1_pred)    # per-step 2nd call (matches original)

        # Heun corrector
        y = y + (dt / 2.0) * (f0 + f1_pred)

        if non_negative:
            y = clamp_straight_through(y, min=0.0)

        ys.append(y.clone())

        if i < len(t) - 1:
            # Prepare next step's f0 at (t1, y) — this is also dy/dt at t1
            # This shifts the usual "start of next iter" eval to here,
            # keeping the total number of calls identical.
            f0 = func(t1, y)           # reused next iteration's f0
            dydts.append(f0.clone())
        else:
            # Final timestep: avoid an extra func call.
            # Use predictor derivative at t1 (computed above).
            # If you prefer exact dy/dt at (t1, y), uncomment the next two lines
            # and delete the append below (this would add exactly 1 extra call):
            # f_final = func(t1, y)
            # dydts.append(f_final.clone())
            dydts.append(f1_pred.clone())

    ys = torch.stack(ys, dim=0)        # (time, batch, state)
    dydts = torch.stack(dydts, dim=0)  # (time, batch, state)

    # (time, 2, batch, state): 0 = y, 1 = dy/dt
    return torch.stack((ys, dydts), dim=1)
