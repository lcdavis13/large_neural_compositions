import torch


def get_loss_functions():
    # specify data loss function and total loss function, and any other losses we want to monitor/plot

    # data_loss_fn = loss_l1
    # total_loss_fn = loss_l1

    data_loss_fn = aitchison_loss_earlystop
    total_loss_fn = aitchison_plus_stability_early

    # data_loss_fn = aitchison_loss
    # total_loss_fn = aitchison_plus_fr_stationarity_and_deceleration
    # total_loss_fn = total_loss
    # total_loss_fn = aitchison_loss

    score_fns = {
        "loss": total_loss_fn,
        "dataloss": data_loss_fn, 
        "aitchison_late": aitchison_loss,
        # "fisher-rao": fisher_rao_loss,
        # "BCD_L1": loss_l1,
        # "stationarity-fisher-rao": stationarity_penalty_fisher_rao,
        # "deceleration-fisher-rao": deceleration_penalty_fisher_rao,
        # 'kinetic_energy': kinetic_energy_penalty,
        # 'jacobian_frobenius': jacobian_surrogate_penalty,
        # "BCD_L1_earlystop": loss_l1_earlystop,
        # "EarlyStopDifference_L1": earlystop_difference_L1,
        "aitchison": aitchison_loss_earlystop,
        "aitchison_late_diff": earlystop_difference_aitch,
        "contraction": terminal_contraction_penalty_aitchison_earlystop,
        "stationarity": stationarity_penalty_aitchison,
        # "derivative_L2": derivative_L2,
        # "simplex_distance": distribution_error,
    }

    return total_loss_fn, score_fns


def _get_state(x, t_fraction=-1.0):
    """
    Extract the final timestep from a possibly-sequence input.

    Accepts:
        - list/tuple of tensors: returns last element (or fractional index)
        - tensor with leading time dim (dim >= 3): index along dim 0
        - other tensor: returns as-is
    """

    # Case 1: Python sequence of tensors
    if isinstance(x, (list, tuple)):
        if len(x) == 0:
            raise ValueError("Received an empty sequence for x.")

        if 0 < t_fraction < 1:
            i = int(t_fraction * len(x))
            if i == 0 or i == len(x) - 1:
                print(
                    "WARNING: _get_state with fractional t resulted in edge index. "
                    "Do you have more than two time steps reported? Index: ", i
                )
            # else:
            #     print(f"DEBUG: _get_state using fractional t resulted in index {i} of {len(x)}.")
        else:
            i = -1

        return x[i]

    # Case 2: Tensor (possibly time series)
    if isinstance(x, torch.Tensor):
        # Heuristic: only treat as time series if we have at least [T, B, D] or similar
        if x.dim() >= 3:
            T = x.size(0)
            if 0 < t_fraction < 1:
                i = int(t_fraction * T)
                if i == 0 or i == T - 1:
                    print(
                        "WARNING: _get_state with fractional t on tensor resulted in edge index. "
                        f"T={T}, index={i}"
                    )
                # else:
                #     print(f"DEBUG: _get_state(tensor) using fractional t resulted in index {i} of {T}.")
            else:
                i = -1
            return x[i]

        # For 1D/2D tensors, treat as single state
        return x

    # Fallback: non-sequence, non-tensor
    # print("DEBUG: _get_state received non-sequence, non-tensor input, returning as-is.")
    return x


def _support_mask(x, support_mask=None):
    """
    Returns a boolean mask with the same shape as `x`, marking nonzero (or
    externally supported) components.

    Args:
        x: [..., D]
        support_mask: None or broadcastable to [..., D]

    Returns:
        mask: [..., D] boolean
    """
    D = x.shape[-1]

    if support_mask is None:
        # Mask from x itself, same shape as x
        support_mask = (x > 0)
    else:
        # Make sure support_mask has a compositional last dim D
        assert support_mask.shape[-1] == D, "support_mask last dim must match x's last dim"

        # Broadcast support_mask to x.shape if needed:
        # e.g. support_mask [B, D] → x [T, B, D]
        while support_mask.dim() < x.dim():
            support_mask = support_mask.unsqueeze(0)
        # Now we can expand if necessary
        support_mask = support_mask.expand_as(x)

    return support_mask


def _clr(x, support_mask=None):
    """
    x: [..., D] compositional data on the simplex (zeros allowed).

    Returns:
        clr_full: [..., D] CLR-transformed values embedded in R^D
                  (zero outside the support).
        mask:     [..., D] boolean mask of supported components.
    """
    support_mask = _support_mask(x, support_mask)
    D = x.shape[-1]

    # Log only on the support; leave zeros elsewhere
    logx = torch.zeros_like(x)
    logx[support_mask] = x[support_mask].log()

    # Number of supported components per "sample" (over last dim)
    counts = support_mask.sum(dim=-1, keepdim=True).clamp_min(1)  # [..., 1]

    # Mean log only over support: sum(logx * mask) / |S|
    mean_logx = (logx * support_mask).sum(dim=-1, keepdim=True) / counts  # [..., 1]

    # clr on support
    clr = logx - mean_logx  # [..., D]

    # Zero outside support
    clr_full = torch.zeros_like(x)
    clr_full[support_mask] = clr[support_mask]

    return clr_full, support_mask


def aitchison_loss(y_pred, y_true, *, ode_fn=None, t=None, hp=None):
    y_pred = _get_state(y_pred)
    y_true = y_true

    # eps = hp.get("aitchison_eps", 1e-8) if hp is not None else 1e-8
    clr_true, mask = _clr(y_true)
    clr_pred, _ = _clr(y_pred, support_mask=mask)

    return ((clr_pred - clr_true)*mask).norm(dim=1).mean()

def aitchison_loss_earlystop(y_pred, y_true, *, ode_fn=None, t=None, hp=None):
    y_pred = _get_state(y_pred, t_fraction=hp.earlystop_fraction)
    y_true = y_true

    # eps = hp.get("aitchison_eps", 1e-8) if hp is not None else 1e-8
    clr_true, mask = _clr(y_true)
    clr_pred, _ = _clr(y_pred, support_mask=mask)

    return ((clr_pred - clr_true)*mask).norm(dim=1).mean()

def earlystop_difference_aitch(y_pred, y_true, *, ode_fn=None, t=None, hp=None):
    # instead of comparing y_pred to y_true, compare y_pred at t=0.67 to y_pred at t=1.0
    # print("DEBUG: y_pred shape:", y_pred.shape)
    # print("DEBUG: y_pred type:", type(y_pred))
    y_pred_early = _get_state(y_pred, t_fraction=hp.earlystop_fraction)
    y_pred_final = _get_state(y_pred, t_fraction=-1)
    return aitchison_loss(y_pred_early, y_pred_final)

def aitchison_plus_stability(y_traj, y_true, *, ode_fn=None, t=None, hp=None):
    """
    Aitchison data loss plus terminal contraction penalty.
    """
    data_loss = aitchison_loss(y_traj, y_true, hp=hp)

    if ode_fn is not None and hp.stabilityloss_weight > 0.0 and hp.contractionloss_weight > 0.0: 
        contraction_loss = hp.stabilityloss_weight * hp.contractionloss_weight* terminal_contraction_penalty_aitchison(y_traj, y_true, ode_fn=ode_fn, t=t, hp=hp)
    else:
        contraction_loss = 0.0

    if ode_fn is not None and hp.stabilityloss_weight > 0.0 and hp.stationarityloss_weight > 0.0:
        stationarity_loss = hp.stabilityloss_weight * hp.stationarityloss_weight * stationarity_penalty_aitchison(y_traj, y_true, ode_fn=ode_fn, t=t, hp=hp)
    else:
        stationarity_loss = 0.0

    return data_loss + contraction_loss + stationarity_loss

def aitchison_plus_stability_early(y_traj, y_true, *, ode_fn=None, t=None, hp=None):
    """
    Aitchison data loss plus terminal contraction penalty.
    """
    data_loss = aitchison_loss_earlystop(y_traj, y_true, hp=hp)
    if ode_fn is not None and hp.stabilityloss_weight > 0.0 and hp.contractionloss_weight > 0.0: 
        contraction_loss = hp.stabilityloss_weight *hp.contractionloss_weight* terminal_contraction_penalty_aitchison_earlystop(y_traj, y_true, ode_fn=ode_fn, t=t, hp=hp)
    else:
        contraction_loss = 0.0

    if ode_fn is not None and hp.stabilityloss_weight > 0.0 and hp.stationarityloss_weight > 0.0:
        stationarity_loss = hp.stabilityloss_weight *hp.stationarityloss_weight * stationarity_penalty_aitchison(y_traj, y_true, ode_fn=ode_fn, t=t, hp=hp)
    else:
        stationarity_loss = 0.0

    return data_loss + contraction_loss + stationarity_loss



def terminal_contraction_penalty(y_traj, y_attract, *, ode_fn=None, t=None, hp=None):
    """
    y_traj:   [T, B, N]  trajectory samples
    y_attract:[B, N]     attractor point per sample
    ode_fn:   callable(t, y) -> dy/dt with y [B, N], t [1]
    t:        [T]        time samples (unused by the vector field itself)
    hp:       dot-accessible hyperparameter container.
    """

    # If no ODE function, contraction loss is irrelevant
    if ode_fn is None:
        return torch.zeros((), dtype=y_traj.dtype, device=y_traj.device)

    # Extract state according to earlystop_fraction
    y_final = _get_state(y_traj, t_fraction=-1.0)   # [B, N]

    assert y_final.shape == y_attract.shape, "y_final and y_attract must both be [B, N]"
    B, N = y_attract.shape

    # --- Construct direction: from attractor to final state ---
    delta = y_final - y_attract     # [B, N]

    # --- Restrict to face support defined by attractor ---
    mask = (y_attract > hp.contractionloss_support_eps).to(y_traj.dtype)  # [B, N]
    delta = delta * mask

    # --- Enforce sum(delta_i) = 0 over the support (tangent space) ---
    support_counts = mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
    mean_support   = (delta * mask).sum(dim=-1, keepdim=True) / support_counts
    delta = delta - mean_support * mask

    # --- Normalize and validate ---
    dir_norm = delta.norm(dim=-1, keepdim=True)
    valid = (dir_norm.squeeze(-1) > hp.contractionloss_min_dir_norm)

    if not valid.any():
        return torch.zeros((), dtype=y_traj.dtype, device=y_traj.device)

    delta_normed = delta / dir_norm.clamp_min(hp.contractionloss_eps)

    # Select valid indices
    valid_idx = valid.nonzero(as_tuple=False).squeeze(-1)

    # Time argument for ode_fn (you said time-invariant, so any scalar is fine)
    t_eval = y_traj.new_tensor([0.0])

    penalties = []

    # KEY: re-enable grad locally, even if caller used torch.no_grad().
    with torch.enable_grad():
        for b in valid_idx:
            # Single batch element
            y0 = y_attract[b:b+1].clone().detach()
            y0.requires_grad_(True)                     # [1, N]
            v  = delta_normed[b:b+1]                    # [1, N], treated as constant

            # Evaluate vector field at attractor
            f = ode_fn(t_eval, y0)                      # [1, N]

            # Sanity: if this fails, ode_fn itself is doing no_grad/detach.
            if not f.requires_grad:
                raise RuntimeError(
                    "ode_fn output does not require grad. "
                    "If you're calling this during validation under torch.no_grad(), "
                    "ensure that ode_fn itself is not wrapped in torch.no_grad() "
                    "or decorated with @torch.no_grad()."
                )

            f_b = f[0]

            # Scalar inner product f · v
            scalar = (f_b * v[0]).sum()

            # Compute gradient wrt y0: grad(scalar) = J^T v
            grad_y = torch.autograd.grad(
                scalar, y0, create_graph=True, retain_graph=True
            )[0][0]   # [N]

            # Rayleigh quotient v^T J v (v already normalized)
            rq = (grad_y * v[0]).sum()

            # Penalty: want rq <= -margin
            penalties.append(torch.relu(rq + hp.contractionloss_margin))

    penalties = torch.stack(penalties)   # [B_valid]

    return penalties.mean()


def terminal_contraction_penalty_earlystop(y_traj, y_attract, *, ode_fn=None, t=None, hp=None):
    """
    y_traj:   [T, B, N]  trajectory samples
    y_attract:[B, N]     attractor point per sample
    ode_fn:   callable(t, y) -> dy/dt with y [B, N], t [1]
    t:        [T]        time samples (unused by the vector field itself)
    hp:       dot-accessible hyperparameter container.
    """

    # If no ODE function, contraction loss is irrelevant
    if ode_fn is None:
        return torch.zeros((), dtype=y_traj.dtype, device=y_traj.device)

    # Extract state according to earlystop_fraction
    y_final = _get_state(y_traj, t_fraction=hp.earlystop_fraction)   # [B, N]

    assert y_final.shape == y_attract.shape, "y_final and y_attract must both be [B, N]"
    B, N = y_attract.shape

    # --- Construct direction: from attractor to final state ---
    delta = y_final - y_attract     # [B, N]

    # --- Restrict to face support defined by attractor ---
    mask = (y_attract > hp.contractionloss_support_eps).to(y_traj.dtype)  # [B, N]
    delta = delta * mask

    # --- Enforce sum(delta_i) = 0 over the support (tangent space) ---
    support_counts = mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
    mean_support   = (delta * mask).sum(dim=-1, keepdim=True) / support_counts
    delta = delta - mean_support * mask

    # --- Normalize and validate ---
    dir_norm = delta.norm(dim=-1, keepdim=True)
    valid = (dir_norm.squeeze(-1) > hp.contractionloss_min_dir_norm)

    if not valid.any():
        return torch.zeros((), dtype=y_traj.dtype, device=y_traj.device)

    delta_normed = delta / dir_norm.clamp_min(hp.contractionloss_eps)

    # Select valid indices
    valid_idx = valid.nonzero(as_tuple=False).squeeze(-1)

    # Time argument for ode_fn (you said time-invariant, so any scalar is fine)
    t_eval = y_traj.new_tensor([0.0])

    penalties = []

    # KEY: re-enable grad locally, even if caller used torch.no_grad().
    with torch.enable_grad():
        for b in valid_idx:
            # Single batch element
            y0 = y_attract[b:b+1].clone().detach()
            y0.requires_grad_(True)                     # [1, N]
            v  = delta_normed[b:b+1]                    # [1, N], treated as constant

            # Evaluate vector field at attractor
            f = ode_fn(t_eval, y0)                      # [1, N]

            # Sanity: if this fails, ode_fn itself is doing no_grad/detach.
            if not f.requires_grad:
                raise RuntimeError(
                    "ode_fn output does not require grad. "
                    "If you're calling this during validation under torch.no_grad(), "
                    "ensure that ode_fn itself is not wrapped in torch.no_grad() "
                    "or decorated with @torch.no_grad()."
                )

            f_b = f[0]

            # Scalar inner product f · v
            scalar = (f_b * v[0]).sum()

            # Compute gradient wrt y0: grad(scalar) = J^T v
            grad_y = torch.autograd.grad(
                scalar, y0, create_graph=True, retain_graph=True
            )[0][0]   # [N]

            # Rayleigh quotient v^T J v (v already normalized)
            rq = (grad_y * v[0]).sum()

            # Penalty: want rq <= -margin
            penalties.append(torch.relu(rq + hp.contractionloss_margin))

    penalties = torch.stack(penalties)   # [B_valid]

    return penalties.mean()


def terminal_contraction_penalty_aitchison(y_traj, y_attract, *, ode_fn=None, t=None, hp=None):
    """
    Aitchison-geometry contraction penalty evaluated at y_attract.
    Direction is defined by (y_final - y_attract) projected to the simplex-face tangent space.

    y_traj:    [T, B, N]
    y_attract: [B, N]
    ode_fn:    callable(t, y) -> dy/dt with y [B, N], t [1]
    t:         [T] (unused by the field; kept for API)
    hp:        dot-accessible hyperparameters (must exist; no internal defaults)
    """

    if ode_fn is None:
        return torch.zeros((), dtype=y_traj.dtype, device=y_traj.device)

    # Extract state used to define direction
    y_final = _get_state(y_traj, t_fraction=-1.0)  # [B, N]
    assert y_final.shape == y_attract.shape

    # --- Direction in y-space (tangent to the face) ---
    v = y_final - y_attract  # [B, N]

    # Support mask defines the face (use your existing convention)
    support_mask = (y_attract > hp.contractionloss_support_eps)  # [B, N] boolean
    v = v * support_mask.to(v.dtype)

    # Enforce tangent constraint sum_{i in S} v_i = 0
    counts = support_mask.sum(dim=-1, keepdim=True).clamp_min(1)
    v_mean = (v * support_mask).sum(dim=-1, keepdim=True) / counts
    v = v - v_mean * support_mask.to(v.dtype)

    # Validate direction magnitude
    v_norm = v.norm(dim=-1, keepdim=True)
    valid = (v_norm.squeeze(-1) > hp.contractionloss_min_dir_norm)
    if not valid.any():
        return torch.zeros((), dtype=y_traj.dtype, device=y_traj.device)

    # We do NOT need v to be Euclidean-unit; Aitchison norm uses M.
    # However, keeping scale reasonable helps numerics.
    v = v / v_norm.clamp_min(hp.contractionloss_eps)

    # Time arg for ode_fn (time-invariant field; any scalar is fine)
    t_eval = y_traj.new_tensor([0.0])

    penalties = []
    valid_idx = valid.nonzero(as_tuple=False).squeeze(-1)

    # Helper: apply Aitchison metric M(y*) to a tangent vector v without forming M
    # Using: Dclr(y) v = (v/y) - mean(v/y) on support
    # Then M v = Dclr(y)^T Dclr(y) v = ( (v/y) - mean ) / y   on support
    def _apply_aitchison_metric(y, v, support_mask):
        # y, v: [1, N]; support_mask: [1, N] boolean
        y_safe = y.clamp_min(hp.contractionloss_y_eps)  # avoid division by zero on support

        # u = v / y on support
        u = torch.zeros_like(v)
        u[support_mask] = v[support_mask] / y_safe[support_mask]

        # subtract mean over support: u <- u - mean(u)
        counts = support_mask.sum(dim=-1, keepdim=True).clamp_min(1)
        u_mean = (u * support_mask).sum(dim=-1, keepdim=True) / counts
        u = u - u_mean * support_mask.to(u.dtype)

        # Mv = u / y on support (since projector is symmetric)
        Mv = torch.zeros_like(v)
        Mv[support_mask] = u[support_mask] / y_safe[support_mask]
        return Mv

    with torch.enable_grad():
        for b in valid_idx:
            y0 = y_attract[b:b+1].clone().detach()
            y0.requires_grad_(True)  # [1, N]

            vb = v[b:b+1].detach()   # [1, N] treat direction as constant for the penalty

            sm = support_mask[b:b+1]  # [1, N] boolean

            # Apply Aitchison metric at the attractor: w = M(y*) v
            w = _apply_aitchison_metric(y0, vb, sm)  # [1, N]

            # Denominator: v^T M v  (Aitchison squared norm up to constant scaling)
            denom = (vb * w).sum()  # scalar
            if denom.abs() < hp.contractionloss_eps:
                # Degenerate direction under the metric; skip
                continue

            # Field at attractor
            f = ode_fn(t_eval, y0)  # [1, N]
            if not f.requires_grad:
                raise RuntimeError(
                    "ode_fn output does not require grad. Ensure ode_fn is not under torch.no_grad() "
                    "and is not detaching outputs."
                )

            # scalar = f · w  => grad_y scalar = J^T w
            scalar = (f[0] * w[0]).sum()

            grad_y = torch.autograd.grad(
                scalar, y0, create_graph=True, retain_graph=True
            )[0][0]  # [N] = J^T w (evaluated at y0)

            # Numerator: v^T (J^T w) = v^T J^T M v = v^T M J v (same scalar)
            numer = (grad_y * vb[0]).sum()

            # Aitchison directional contraction rate
            rho = numer / denom

            penalties.append(torch.relu(rho + hp.contractionloss_margin))

    if not penalties:
        return torch.zeros((), dtype=y_traj.dtype, device=y_traj.device)

    penalties = torch.stack(penalties)
    return penalties.mean()


def terminal_contraction_penalty_aitchison_earlystop(y_traj, y_attract, *, ode_fn=None, t=None, hp=None):
    """
    Aitchison-geometry contraction penalty evaluated at y_attract.
    Direction is defined by (y_final - y_attract) projected to the simplex-face tangent space.

    y_traj:    [T, B, N]
    y_attract: [B, N]
    ode_fn:    callable(t, y) -> dy/dt with y [B, N], t [1]
    t:         [T] (unused by the field; kept for API)
    hp:        dot-accessible hyperparameters (must exist; no internal defaults)
    """

    if ode_fn is None:
        return torch.zeros((), dtype=y_traj.dtype, device=y_traj.device)

    # Extract state used to define direction
    y_final = _get_state(y_traj, t_fraction=hp.earlystop_fraction)  # [B, N]
    assert y_final.shape == y_attract.shape

    # --- Direction in y-space (tangent to the face) ---
    v = y_final - y_attract  # [B, N]

    # Support mask defines the face (use your existing convention)
    support_mask = (y_attract > hp.contractionloss_support_eps)  # [B, N] boolean
    v = v * support_mask.to(v.dtype)

    # Enforce tangent constraint sum_{i in S} v_i = 0
    counts = support_mask.sum(dim=-1, keepdim=True).clamp_min(1)
    v_mean = (v * support_mask).sum(dim=-1, keepdim=True) / counts
    v = v - v_mean * support_mask.to(v.dtype)

    # Validate direction magnitude
    v_norm = v.norm(dim=-1, keepdim=True)
    valid = (v_norm.squeeze(-1) > hp.contractionloss_min_dir_norm)
    if not valid.any():
        return torch.zeros((), dtype=y_traj.dtype, device=y_traj.device)

    # We do NOT need v to be Euclidean-unit; Aitchison norm uses M.
    # However, keeping scale reasonable helps numerics.
    v = v / v_norm.clamp_min(hp.contractionloss_eps)

    # Time arg for ode_fn (time-invariant field; any scalar is fine)
    t_eval = y_traj.new_tensor([0.0])

    penalties = []
    valid_idx = valid.nonzero(as_tuple=False).squeeze(-1)

    # Helper: apply Aitchison metric M(y*) to a tangent vector v without forming M
    # Using: Dclr(y) v = (v/y) - mean(v/y) on support
    # Then M v = Dclr(y)^T Dclr(y) v = ( (v/y) - mean ) / y   on support
    def _apply_aitchison_metric(y, v, support_mask):
        # y, v: [1, N]; support_mask: [1, N] boolean
        y_safe = y.clamp_min(hp.contractionloss_y_eps)  # avoid division by zero on support

        # u = v / y on support
        u = torch.zeros_like(v)
        u[support_mask] = v[support_mask] / y_safe[support_mask]

        # subtract mean over support: u <- u - mean(u)
        counts = support_mask.sum(dim=-1, keepdim=True).clamp_min(1)
        u_mean = (u * support_mask).sum(dim=-1, keepdim=True) / counts
        u = u - u_mean * support_mask.to(u.dtype)

        # Mv = u / y on support (since projector is symmetric)
        Mv = torch.zeros_like(v)
        Mv[support_mask] = u[support_mask] / y_safe[support_mask]
        return Mv

    with torch.enable_grad():
        for b in valid_idx:
            y0 = y_attract[b:b+1].clone().detach()
            y0.requires_grad_(True)  # [1, N]

            vb = v[b:b+1].detach()   # [1, N] treat direction as constant for the penalty

            sm = support_mask[b:b+1]  # [1, N] boolean

            # Apply Aitchison metric at the attractor: w = M(y*) v
            w = _apply_aitchison_metric(y0, vb, sm)  # [1, N]

            # Denominator: v^T M v  (Aitchison squared norm up to constant scaling)
            denom = (vb * w).sum()  # scalar
            if denom.abs() < hp.contractionloss_eps:
                # Degenerate direction under the metric; skip
                continue

            # Field at attractor
            f = ode_fn(t_eval, y0)  # [1, N]
            if not f.requires_grad:
                raise RuntimeError(
                    "ode_fn output does not require grad. Ensure ode_fn is not under torch.no_grad() "
                    "and is not detaching outputs."
                )

            # scalar = f · w  => grad_y scalar = J^T w
            scalar = (f[0] * w[0]).sum()

            grad_y = torch.autograd.grad(
                scalar, y0, create_graph=True, retain_graph=True
            )[0][0]  # [N] = J^T w (evaluated at y0)

            # Numerator: v^T (J^T w) = v^T J^T M v = v^T M J v (same scalar)
            numer = (grad_y * vb[0]).sum()

            # Aitchison directional contraction rate
            rho = numer / denom

            penalties.append(torch.relu(rho + hp.contractionloss_margin))

    if not penalties:
        return torch.zeros((), dtype=y_traj.dtype, device=y_traj.device)

    penalties = torch.stack(penalties)
    return penalties.mean()


def stationarity_penalty_aitchison(y_traj, y_attract, *, ode_fn=None, t=None, hp=None):
    """
    Aitchison-geometry stationarity penalty evaluated at y_attract.

    Penalizes || f(y_attract) ||_A^2, where the Aitchison norm is induced by the
    clr differential / metric on the simplex face defined by the attractor support.

    y_traj:    [T, B, N]  (unused, present for signature compatibility)
    y_attract: [B, N]
    ode_fn:    callable(t, y) -> dy/dt with y [B, N], t [1]
    t:         [T] (unused by the field; kept for API compatibility)
    hp:        dot-accessible hyperparameter container (must exist; no internal defaults)
    """

    if ode_fn is None:
        return torch.zeros((), dtype=y_traj.dtype, device=y_traj.device)

    # Support mask defines the face
    support_mask = (y_attract > hp.stationarityloss_support_eps)  # [B, N] boolean

    # Time argument for ode_fn (time-invariant field; any scalar is fine)
    t_eval = y_traj.new_tensor([0.0])

    # Apply Aitchison metric M(y*) to a tangent vector v without forming M explicitly.
    # Using: Dclr(y) v = (v/y) - mean_support(v/y) on support
    # Then:  M v = Dclr(y)^T Dclr(y) v = ((v/y) - mean) / y on support
    def _apply_aitchison_metric(y, v, sm):
        y_safe = y.clamp_min(hp.stationarityloss_y_eps)

        u = torch.zeros_like(v)
        u[sm] = v[sm] / y_safe[sm]

        counts = sm.sum(dim=-1, keepdim=True).clamp_min(1)
        u_mean = (u * sm).sum(dim=-1, keepdim=True) / counts
        u = u - u_mean * sm.to(u.dtype)

        Mv = torch.zeros_like(v)
        Mv[sm] = u[sm] / y_safe[sm]
        return Mv

    # We want this term present in validation too, so re-enable grads locally if needed.
    with torch.enable_grad():
        y0 = y_attract.clone().detach()
        y0.requires_grad_(True)  # ensure ode_fn output can carry grad if needed downstream

        f0 = ode_fn(t_eval, y0)  # [B, N]
        if not f0.requires_grad:
            raise RuntimeError(
                "ode_fn output does not require grad. Ensure ode_fn is not under torch.no_grad() "
                "and is not detaching outputs."
            )

        # Restrict to the face support
        f0 = f0 * support_mask.to(f0.dtype)

        # Ensure tangent constraint sum_{i in S} f_i = 0 (safe even if already true)
        counts = support_mask.sum(dim=-1, keepdim=True).clamp_min(1)
        mean_f0 = (f0 * support_mask).sum(dim=-1, keepdim=True) / counts
        f0 = f0 - mean_f0 * support_mask.to(f0.dtype)

        # Aitchison norm squared: f^T M f
        Mf0 = _apply_aitchison_metric(y0, f0, support_mask)
        per_sample = (f0 * Mf0).sum(dim=-1)  # [B]

        # Numerical safety: avoid negative from tiny numerical artifacts
        per_sample = torch.clamp(per_sample, min=0.0)

        return per_sample.mean()



def fisher_rao_loss(y_pred, y_true, *, ode_fn=None, t=None, hp=None):
    """
    Fisher–Rao loss on the simplex, supporting arbitrary leading batch dims.

    y_pred, y_true: [..., D], with last dim = compositional components.
    """
    y_pred = _get_state(y_pred)  # map to simplex
    y_true = y_true

    eps = getattr(hp, "fisher_rao_eps", 1e-8)

    # Mask determined by y_true (same convention as Aitchison)
    mask = _support_mask(y_true)          # [..., D], bool

    # Apply mask and clamp to avoid log/sqrt issues
    y_pred_safe = y_pred.clamp_min(eps) * mask
    y_true_safe = y_true.clamp_min(eps) * mask

    # Renormalize over the support so we are on a proper simplex subspace
    pred_sum = y_pred_safe.sum(dim=-1, keepdim=True).clamp_min(eps)
    true_sum = y_true_safe.sum(dim=-1, keepdim=True).clamp_min(eps)

    p = y_pred_safe / pred_sum           # [..., D]
    q = y_true_safe / true_sum           # [..., D]

    # Inner product in square-root coordinates over the LAST dimension
    inner = (p.sqrt() * q.sqrt()).sum(dim=-1)  # [...], batch-only shape

    # Numerical safety
    inner = inner.clamp(0.0, 1.0 - eps)

    # Fisher–Rao distance per sample
    dist = 2.0 * torch.acos(inner)       # [...], same batch shape

    # Average over all batch dimensions
    return dist.mean()


def loss_l1(y_pred, y_true, *, ode_fn=None, t=None, hp=None):
    # Use only final timestep
    y_pred = _get_state(y_pred)
    # y_true = _final_state(y_true)
    return torch.mean(torch.abs(y_pred - y_true))

def loss_l1_earlystop(y_pred, y_true, *, ode_fn=None, t=None, hp=None):
    # Use only final timestep
    y_pred = _get_state(y_pred, t_fraction=hp.earlystop_fraction)
    # y_true = _final_state(y_true)
    return torch.mean(torch.abs(y_pred - y_true))

def earlystop_difference_L1(y_pred, y_true, *, ode_fn=None, t=None, hp=None):
    # instead of comparing y_pred to y_true, compare y_pred at t=0.67 to y_pred at t=1.0
    # print("DEBUG: y_pred shape:", y_pred.shape)
    # print("DEBUG: y_pred type:", type(y_pred))
    y_pred_early = _get_state(y_pred, t_fraction=hp.earlystop_fraction)
    y_pred_final = _get_state(y_pred, t_fraction=-1)
    return torch.mean(torch.abs(y_pred_early - y_pred_final))


def distribution_error(x, y, *, ode_fn=None, t=None, hp=None):
    """
    Penalty for invalid distributions, measured as the Euclidean distance
    from each vector to the probability simplex.

    Uses only the final timestep if x is a sequence.
    y is unused but included to match signature of other score functions.
    """

    x = _get_state(x)

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




# OLD CODE BELOW THIS

def old_kinetic_energy_penalty(y_pred, y_true, *, ode_fn=None, t=None, hp=None):
    """
    TO DO: Now that I have access to ode_fn, this old finite-difference version is outdated.

    Finite-difference kinetic energy penalty ala 'How to Train Your Neural ODEs'.

    Approximates the vector field f via:
        f_k ≈ (y_{k+1} - y_k) / Δt_k

    and penalizes a time-weighted average:
        KE ≈ mean_batch sum_k w_k ||f_k||^2
    where w_k = Δt_k / sum_j Δt_j (so the scale is invariant to time rescaling).

    Args:
        y_pred: tensor [T, B, D]
        y_true: tensor [B, D] (unused, kept for interface symmetry)
        t:      1D tensor / array-like [T] of times; if None, assume uniform.
        hp: hyperparameters containing:
            - kinetic_energy_lambda  (float > 0 to enable)

    Returns:
        scalar kinetic energy penalty (lambda is NOT applied here).
    """
    Y = y_pred
    if Y.ndim < 3 or Y.shape[0] < 2:
        return torch.zeros((), device=y_true.device, dtype=y_true.dtype)

    if hp is None or hp.kinetic_energy_lambda <= 0.0:
        return torch.zeros((), device=y_true.device, dtype=y_true.dtype)

    T = Y.shape[0]

    # ----- Time grid -----
    if t is None:
        t = torch.linspace(0.0, 1.0, steps=T,
                           device=y_true.device, dtype=y_true.dtype)
    else:
        t = torch.tensor(t, device=y_true.device, dtype=y_true.dtype)

    dt = t[1:] - t[:-1]                      # [T-1]
    dt = dt.clamp_min(1e-8)

    # ----- Finite-difference approximation of f -----
    dY = Y[1:] - Y[:-1]                      # [T-1, B, D]
    dt_reshaped = dt.view(-1, 1, 1)          # [T-1, 1, 1]
    f_approx = dY / dt_reshaped              # [T-1, B, D]

    # ||f||^2 per interval and sample
    f_sq = f_approx.pow(2).sum(dim=-1)       # [T-1, B]

    # Time-weighted average over intervals, then mean over batch
    w = dt / dt.sum()                        # [T-1]
    w_expanded = w.view(-1, 1)               # [T-1, 1]

    weighted_f_sq = w_expanded * f_sq        # [T-1, B]
    ke_per_sample = weighted_f_sq.sum(dim=0) # [B]
    penalty = ke_per_sample.mean()           # scalar

    return penalty


def old_jacobian_surrogate_penalty(y_pred, y_true, *, ode_fn=None, t=None, hp=None):
    """
    TO DO: Now that I have access to ode_fn, this old finite-difference version is outdated.

    Finite-difference surrogate for the Jacobian Frobenius penalty.

    Approximates curvature / stiffness by second differences:

        v_k      ≈ (y_{k+1} - y_k) / Δt_k
        a_k      ≈ (v_{k+1} - v_k) / Δτ_k,  Δτ_k ≈ 0.5 * (Δt_k + Δt_{k+1})

    and penalizes a time-weighted average of ||a_k||^2.

    This is NOT an exact Jacobian-norm penalty, but a practical proxy that
    only uses the trajectory y_pred, in the spirit of controlling stiffness.

    Args:
        y_pred: tensor [T, B, D]
        y_true: tensor [B, D] (unused)
        t:      1D tensor / array-like [T] of times; if None, assume uniform.
        hp: hyperparameters containing:
            - jacobian_lambda  (float > 0 to enable)

    Returns:
        scalar curvature/stiffness penalty (lambda is NOT applied here).
    """
    Y = y_pred
    if Y.ndim < 3 or Y.shape[0] < 3:
        # Need at least 3 time points for second differences
        return torch.zeros((), device=y_true.device, dtype=y_true.dtype)

    if hp is None or hp.jacobian_lambda <= 0.0:
        return torch.zeros((), device=y_true.device, dtype=y_true.dtype)

    T = Y.shape[0]

    # ----- Time grid -----
    if t is None:
        t = torch.linspace(0.0, 1.0, steps=T,
                           device=y_true.device, dtype=y_true.dtype)
    else:
        t = torch.tensor(t, device=y_true.device, dtype=y_true.dtype)

    dt = t[1:] - t[:-1]                      # [T-1]
    dt = dt.clamp_min(1e-8)

    # ----- First differences: velocity -----
    dY = Y[1:] - Y[:-1]                      # [T-1, B, D]
    dt_reshaped = dt.view(-1, 1, 1)          # [T-1, 1, 1]
    v = dY / dt_reshaped                     # [T-1, B, D]

    # ----- Second differences: acceleration -----
    dv = v[1:] - v[:-1]                      # [T-2, B, D]

    # Effective time step for accelerations: average of neighboring dt
    dt_mid = 0.5 * (dt[1:] + dt[:-1])        # [T-2]
    dt_mid = dt_mid.clamp_min(1e-8)
    dt_mid_reshaped = dt_mid.view(-1, 1, 1)  # [T-2, 1, 1]

    a = dv / dt_mid_reshaped                 # [T-2, B, D]

    # ||a||^2 per mid-interval and sample
    a_sq = a.pow(2).sum(dim=-1)              # [T-2, B]

    # Time-weighted average over mid-intervals, then mean over batch
    w = dt_mid / dt_mid.sum()                # [T-2]
    w_expanded = w.view(-1, 1)               # [T-2, 1]

    weighted_a_sq = w_expanded * a_sq        # [T-2, B]
    contractionloss_per_sample = weighted_a_sq.sum(dim=0)# [B]
    penalty = contractionloss_per_sample.mean()          # scalar

    return penalty



def loss_bc_dki(y_pred, y_true, *, ode_fn=None, t=None, hp=None):
    # Use only final timestep
    y_pred = _get_state(y_pred)
    # y_true = _final_state(y_true)
    return torch.sum(torch.abs(y_pred - y_true)) / torch.sum(
        torch.abs(y_pred + y_true)
    )  # DKI repo implementation (incorrect)

def loss_bc_old(y_pred, y_true, *, ode_fn=None, t=None, hp=None):
    # Use only final timestep
    y_pred = _get_state(y_pred)
    # y_true = _final_state(y_true)
    return torch.sum(torch.abs(y_pred - y_true)) / torch.sum(
        torch.abs(y_pred) + torch.abs(y_true)
    )
