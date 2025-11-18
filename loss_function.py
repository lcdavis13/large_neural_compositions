import torch


def get_loss_functions():
    # specify loss function
    # loss_fn = loss_bc_plus_derivativeL2
    # loss_fn = aitchison_loss
    # loss_fn = lyapunov_aitchison_loss

    data_loss_fn = loss_l1
    total_loss_fn = loss_l1

    score_fns = {
        "loss": total_loss_fn,
        "dataloss": data_loss_fn, 
        "aitchison": aitchison_loss,
        # "lyapunov": lyapunov_penalty_aitchison,
        "BCD_L1": loss_l1,
        # "derivative_L2": derivative_L2,
        "simplex_distance": distribution_error,
    }

    return total_loss_fn, score_fns


def _final_state(x):
    """
    Extract the final timestep from a possibly-sequence input.

    Accepts:
        - list/tuple of tensors: returns last element
        - single tensor: returns as-is
    """
    if isinstance(x, (list, tuple)):
        if len(x) == 0:
            raise ValueError("Received an empty sequence for x.")
        return x[-1]
    return x


def _final_derivative_from_sequence(x, dt: float = 1.0):
    """
    Compute derivative at final timestep using last two states:
        dydt_T ≈ (x_T - x_{T-1}) / dt

    If we can't compute it (no time dimension), returns zeros.
    """
    if isinstance(x, (list, tuple)):
        if len(x) < 2:
            # Not enough timesteps, treat derivative as zero
            last = _final_state(x)
            return torch.zeros_like(last)
        x_T = x[-1]
        x_Tm1 = x[-2]
        return (x_T - x_Tm1) / dt
    else:
        # No notion of timesteps; treat derivative as zero
        return torch.zeros_like(x)


def _clr(z, eps):
    z = torch.log(z.clamp_min(eps))
    return z - z.mean(dim=1, keepdim=True)


def aitchison_loss(y_pred, y_true, *, hp=None):
    y_pred = _final_state(y_pred)
    y_true = y_true

    eps = hp.aitchison_eps
    clr_pred = _clr(y_pred, eps)
    clr_true = _clr(y_true, eps)

    return (clr_pred - clr_true).norm(dim=1).mean()

def lyapunov_penalty_aitchison(y_pred, y_true, *, hp=None):
    """
    Computes a Lyapunov monotonicity penalty over a *sequence* y_pred.

    Penalizes positive differences ReLU(V_{k+1} - V_k), weighted by a
    ramp that begins at hp.lyapunov_ramp_start_k and ends at 1.0
    on the last timestep.

    Args:
        y_pred: list/tuple of tensors with shape [T, B, D] conceptually.
                If not a sequence, penalty = 0.
        y_true: tensor [B, D]
        hp: hyperparameters containing:
            - aitchison_eps
            - lyapunov_ramp_start_k
            - lyapunov_weight (optional, default=1.0)

    Returns:
        scalar Lyapunov penalty
    """
    # No time dimension → no Lyapunov structure → penalty is zero
    if not isinstance(y_pred, (list, tuple)) or len(y_pred) < 2:
        return torch.zeros((), device=y_true.device, dtype=y_true.dtype)

    eps = hp.aitchison_eps
    ramp_start = int(hp.lyapunov_ramp_start_k * (len(y_pred) - 1))
    lyapunov_lambda = getattr(hp, "lyapunov_lambda", 1.0)

    # If weighting is zero, no need to compute anything else
    if lyapunov_lambda == 0.0:
        return torch.zeros((), device=y_true.device, dtype=y_true.dtype)

    # ----- Compute V_k = Aitchison distance at each timestep -----
    Vs = []
    clr_true = _clr(y_true, eps)

    for y_k in y_pred:
        clr_pred_k = _clr(y_k, eps)
        V_k = (clr_pred_k - clr_true).norm(dim=1)  # [batch]
        Vs.append(V_k)

    V = torch.stack(Vs, dim=0)  # [T, batch]
    T = V.shape[0]

    # ----- Finite differences: V_{k+1} - V_k -----
    V_diff = V[1:] - V[:-1]            # [T-1, batch]
    increases = torch.relu(V_diff)     # [T-1, batch]

    # ----- Ramp weights over k = 1..T-1  (the "end" of each transition) -----
    k_end = torch.arange(1, T, device=V.device)  # [1, 2, ..., T-1]
    start = float(ramp_start)
    end = float(T - 1)

    if end <= start:
        w = torch.zeros_like(k_end, dtype=V.dtype)
    else:
        w = (k_end.float() - start) / max(1.0, (end - start))
        w = w.clamp(min=0.0, max=1.0)

    w = w.unsqueeze(-1)  # [T-1, 1] so it broadcasts with [T-1, batch]

    # ----- Weighted penalty -----
    lyap_penalty = (w * increases).mean()
    return lyap_penalty



def lyapunov_aitchison_loss(y_pred, y_true, *, hp=None):
    """
    Combined:
        final-time Aitchison loss
      + Lyapunov monotonicity penalty.
    """
    data_loss = aitchison_loss(y_pred, y_true, hp=hp)
    lyap_lambda = hp.lyapunov_lambda
    lyap_penalty = lyapunov_penalty_aitchison(y_pred, y_true, hp=hp)
    return data_loss + lyap_lambda*lyap_penalty




def loss_bc_plus_derivativeL2(y_pred, y_true, *, hp=None):
    # Both terms use only the final timestep of y_pred (and y_true)
    loss = loss_bc(y_pred, y_true)
    if hp is not None and getattr(hp, "supervise_derivative", False):
        factor = 10.0 ** hp.derivative_loss_log10_scale
        loss = loss + factor * derivative_L2(y_pred, y_true, hp=hp)
    return loss


def derivative_L2(y_pred, y_true, *, hp=None):
    """
    L2 penalty on the derivative at the final timestep.

    y_pred: sequence of timesteps or single tensor.
    y_true: unused, kept for API consistency.
    """
    dydt_final = _final_derivative_from_sequence(y_pred)
    return torch.mean(dydt_final ** 2)


def loss_bc_dki(y_pred, y_true, *, hp=None):
    # Use only final timestep
    y_pred = _final_state(y_pred)
    # y_true = _final_state(y_true)
    return torch.sum(torch.abs(y_pred - y_true)) / torch.sum(
        torch.abs(y_pred + y_true)
    )  # DKI repo implementation (incorrect)

def loss_l1(y_pred, y_true, *, hp=None):
    # Use only final timestep
    y_pred = _final_state(y_pred)
    # y_true = _final_state(y_true)
    return torch.mean(torch.abs(y_pred - y_true))

def loss_bc(y_pred, y_true, *, hp=None):  # Bray-Curtis Dissimilarity
    # Use only final timestep
    y_pred = _final_state(y_pred)
    # y_true = _final_state(y_true)
    return torch.mean(
        torch.sum(torch.abs(y_pred - y_true), dim=-1)
        / torch.sum(torch.abs(y_pred) + torch.abs(y_true), dim=-1)
    )


def loss_masked_aitchison(x, y, *, hp=None):
    """
    Aitchison distance with zeros masked out. Differentiable & vectorized.
    Uses only final timestep if given time sequences.

    Args:
        x, y: (B, D) tensors of simplex vectors, same zero pattern within each row.

    Returns:
        scalar: mean distance over batch.
    """
    x = _final_state(x)
    y = _final_state(y)

    assert x.shape == y.shape
    mask = (y > 0)  # shape: (B, D)

    # Avoid log(0) by masking: log(x) where x > 0, else 0
    log_x = torch.where(mask, torch.log(x.clamp_min(eps)), torch.zeros_like(x))
    log_y = torch.where(mask, torch.log(y.clamp_min(eps)), torch.zeros_like(y))

    # Compute masked mean
    count = mask.sum(dim=-1, keepdim=True).clamp(min=1)  # Avoid division by 0
    mean_log_x = log_x.sum(dim=-1, keepdim=True) / count
    mean_log_y = log_y.sum(dim=-1, keepdim=True) / count

    # CLR transform with masked mean
    clr_x = torch.where(mask, log_x - mean_log_x, torch.zeros_like(x))
    clr_y = torch.where(mask, log_y - mean_log_y, torch.zeros_like(y))

    # Euclidean distance
    diff = clr_x - clr_y
    dist = torch.norm(diff, dim=-1)  # (B,)

    return dist.mean()


def loss_logbc(y_pred, y_true, *, hp=None):
    """
    Bray-Curtis on log-transformed data, using only final timestep.
    Emphasizes loss of rare species.
    """
    y_pred = _final_state(y_pred)
    # y_true = _final_state(y_true)
    return loss_bc(torch.log(y_pred + 1), torch.log(y_true + 1))


def loss_loglogbc(y_pred, y_true, *, hp=None):
    """
    Bray-Curtis on log(log(x+1)+1) transformed data, using only final timestep.
    Emphasizes loss of rare species even more.
    """
    y_pred = _final_state(y_pred)
    # y_true = _final_state(y_true)
    return loss_logbc(torch.log(y_pred + 1), torch.log(y_true + 1))


def loss_bc_old(y_pred, y_true, *, hp=None):
    # Use only final timestep
    y_pred = _final_state(y_pred)
    # y_true = _final_state(y_true)
    return torch.sum(torch.abs(y_pred - y_true)) / torch.sum(
        torch.abs(y_pred) + torch.abs(y_true)
    )


def loss_bc_scaled(y_pred, y_true, *, hp=None, epsilon=1e-10):
    y_pred = _final_state(y_pred)
    # y_true = _final_state(y_true)
    numerator = torch.sum(
        torch.abs(y_pred - y_true) / (torch.abs(y_true) + epsilon), dim=-1
    )
    denominator = torch.sum(
        torch.abs(y_pred)
        + torch.abs(y_true) / (torch.abs(y_true) + epsilon),
        dim=-1,
    )
    return torch.mean(numerator / denominator)


def loss_bc_root(y_pred, y_true, *, hp=None):
    return torch.sqrt(loss_bc(y_pred, y_true))


def loss_bc_logscaled(y_pred, y_true, *, hp=None, epsilon=1e-10):
    y_pred = _final_state(y_pred)
    # y_true = _final_state(y_true)
    numerator = torch.sum(
        torch.abs(y_pred - y_true)
        / torch.log(torch.abs(y_true) + 1 + epsilon)
    )
    denominator = torch.sum(
        torch.abs(y_pred)
        + torch.abs(y_true)
        / torch.log(torch.abs(y_true) + 1 + epsilon)
    )
    return numerator / denominator


def distribution_error(x, y, *, hp=None):
    """
    Penalty for invalid distributions, measured as the Euclidean distance
    from each vector to the probability simplex.

    Uses only the final timestep if x is a sequence.
    y is unused but included to match signature of other score functions.
    """

    x = _final_state(x)

    def project_onto_simplex(v, axis=-1):
        v_sorted, _ = torch.sort(v, descending=True, dim=axis)
        cssv = torch.cumsum(v_sorted, dim=axis) - 1
        ind = torch.arange(1, v.size(axis) + 1, device=v.device).view(
            [1] * axis + [-1]
        )
        cond = v_sorted - cssv / ind > 0

        rho = cond.cumsum(dim=axis)
        rho[cond == 0] = 0
        rho_max, _ = rho.max(dim=axis, keepdim=True)

        # Clamp to avoid invalid indices
        safe_idx = torch.clamp(rho_max - 1, min=0)
        theta = cssv.gather(axis, safe_idx) / rho_max.clamp(min=1).type(v.dtype)
        return torch.clamp(v - theta, min=0.0)

    projected = project_onto_simplex(x)
    distances = torch.norm(x - projected, dim=1)
    return torch.mean(distances)
