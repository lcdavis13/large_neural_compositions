import torch
import chunked_dataset
import torch.nn.functional as F


def evaluate_stream(dataset, prediction_fn, score_fn_dict, hp, device=None):
    model_eval = getattr(prediction_fn, 'eval', None)
    if model_eval:
        model_eval()

    total_scores = {name: 0.0 for name in score_fn_dict}
    total_samples = 0

    with torch.no_grad():
        for chunk in dataset.stream_by_chunk(device=device):
            x = chunk[chunked_dataset.DK_X]
            y = chunk[chunked_dataset.DK_Y]
            y_pred = prediction_fn(x)
            batch_size = y.size(0)

            for name, fn in score_fn_dict.items():
                batch_score = fn(y_pred, y, hp=hp) * batch_size
                total_scores[name] += batch_score

            total_samples += batch_size

    avg_scores = {
        name: total / total_samples if total_samples > 0 else float("nan")
        for name, total in total_scores.items()
    }
    return avg_scores



def fit_and_evaluate_linear_regression(
    data_train, data_valid, data_test,
    fold_num, score_fns, data_dim, hp,
    verbosity=0
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_score_fns = dict(score_fns)
    all_score_fns['mse'] = lambda y_pred, y, hp: F.mse_loss(y_pred, y, reduction='mean')

    x_list, y_list = [], []
    num_samples = 0

    for chunk in data_train.stream_by_chunk(device=device):
        x = chunk[chunked_dataset.DK_X]
        y = chunk[chunked_dataset.DK_Y]
        x_list.append(x)
        y_list.append(y)
        num_samples += x.shape[0]

    # With intercept, effective parameter count is D+1 (per output dim),
    # but your original data_dim logic is external; we keep the same guard.
    if num_samples < data_dim:
        return None  # Underdetermined system: skip

    X = torch.cat(x_list, dim=0)
    Y = torch.cat(y_list, dim=0)

    X_aug = add_intercept(X)  # (N, D+1)

    xtx = X_aug.T @ X_aug
    xty = X_aug.T @ Y
    try:
        weights = torch.linalg.solve(xtx, xty)  # (D+1, K)
    except RuntimeError:
        weights = torch.linalg.pinv(xtx) @ xty

    def predict(x):
        x_aug = add_intercept(x)
        return x_aug @ weights

    trn_scores = evaluate_stream(data_train, prediction_fn=predict, score_fn_dict=all_score_fns, hp=hp, device=device)
    val_scores = evaluate_stream(data_valid, prediction_fn=predict, score_fn_dict=all_score_fns, hp=hp, device=device)
    test_scores = evaluate_stream(data_test, prediction_fn=predict, score_fn_dict=all_score_fns, hp=hp, device=device)

    fold_stats_dict = {f"train_{k}": v.item() for k, v in trn_scores.items()}
    fold_stats_dict.update({f"valid_{k}": v.item() for k, v in val_scores.items()})
    fold_stats_dict.update({f"test_{k}": v.item() for k, v in test_scores.items()})

    if verbosity > 0:
        print(f"\nLINEAR REGRESSION (with intercept) metrics for fold {fold_num}:")
        for k, v in val_scores.items():
            print(f"  {k}: {v:.6f}")
        print()

    return fold_stats_dict, {}


def fit_and_evaluate_moore_penrose(
    data_train, data_valid, data_test,
    fold_num, score_fns, data_dim, hp,
    verbosity=0
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_score_fns = dict(score_fns)
    all_score_fns['mse'] = lambda y_pred, y, hp: F.mse_loss(y_pred, y, reduction='mean')

    x_list, y_list = [], []
    num_samples = 0

    for chunk in data_train.stream_by_chunk(device=device):
        x = chunk[chunked_dataset.DK_X]
        y = chunk[chunked_dataset.DK_Y]
        x_list.append(x)
        y_list.append(y)
        num_samples += x.shape[0]

    if num_samples >= data_dim:
        return None  # Not underdetermined: skip

    X = torch.cat(x_list, dim=0)
    Y = torch.cat(y_list, dim=0)

    X_aug = add_intercept(X)  # (N, D+1)

    weights = torch.linalg.pinv(X_aug) @ Y  # (D+1, K)

    def predict(x):
        x_aug = add_intercept(x)
        return x_aug @ weights

    trn_scores = evaluate_stream(data_train, prediction_fn=predict, score_fn_dict=all_score_fns, hp=hp, device=device)
    val_scores = evaluate_stream(data_valid, prediction_fn=predict, score_fn_dict=all_score_fns, hp=hp, device=device)
    test_scores = evaluate_stream(data_test, prediction_fn=predict, score_fn_dict=all_score_fns, hp=hp, device=device)

    fold_stats_dict = {f"train_{k}": v.item() for k, v in trn_scores.items()}
    fold_stats_dict.update({f"valid_{k}": v.item() for k, v in val_scores.items()})
    fold_stats_dict.update({f"test_{k}": v.item() for k, v in test_scores.items()})

    if verbosity > 0:
        print(f"\nMOORE-PENROSE REGRESSION (with intercept) metrics for fold {fold_num}:")
        for k, v in val_scores.items():
            print(f"  {k}: {v:.6f}")
        print()

    return fold_stats_dict, {}


def fit_and_evaluate_lr_or_mp(data_train, data_valid, data_test, fold_num, score_fns, data_dim, hp, verbosity=0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_samples = 0

    for chunk in data_train.stream_by_chunk(device=device):
        x = chunk[chunked_dataset.DK_X]
        num_samples += x.shape[0]

    if num_samples >= data_dim:
        return fit_and_evaluate_linear_regression(data_train, data_valid, data_test, fold_num, score_fns, data_dim, hp, verbosity)
    else:
        return fit_and_evaluate_moore_penrose(data_train, data_valid, data_test, fold_num, score_fns, data_dim, hp, verbosity)




def _support_mask(x, support_mask=None):
    D = x.shape[-1]
    if support_mask is None:
        support_mask = (x > 0)
    else:
        assert support_mask.shape[-1] == D
        while support_mask.dim() < x.dim():
            support_mask = support_mask.unsqueeze(0)
        support_mask = support_mask.expand_as(x)
    return support_mask


def _clr(x, support_mask=None, eps=1e-12):
    """
    Masked CLR: logs only on support, zeros elsewhere.
    eps is used to clamp supported entries away from 0 for numerical safety.
    """
    support_mask = _support_mask(x, support_mask)
    x = torch.where(support_mask, torch.clamp(x, min=eps), x)

    logx = torch.zeros_like(x)
    logx[support_mask] = x[support_mask].log()

    counts = support_mask.sum(dim=-1, keepdim=True).clamp_min(1)
    mean_logx = (logx * support_mask).sum(dim=-1, keepdim=True) / counts
    clr = logx - mean_logx

    clr_full = torch.zeros_like(x)
    clr_full[support_mask] = clr[support_mask]
    return clr_full, support_mask


# ----------------------------
# Regression helpers
# ----------------------------

def add_intercept(X: torch.Tensor) -> torch.Tensor:
    ones = torch.ones((X.shape[0], 1), device=X.device, dtype=X.dtype)
    return torch.cat([X, ones], dim=1)


def masked_softmax_to_simplex(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Enforce support and closure:
      - logits outside mask are -inf
      - softmax distributes mass over mask only
    """
    # If you ever have mask with zero trues, fall back to uniform
    counts = mask.sum(dim=-1, keepdim=True)
    safe = counts > 0

    neg_inf = torch.finfo(logits.dtype).min
    masked_logits = torch.where(mask, logits, torch.full_like(logits, neg_inf))
    y = torch.softmax(masked_logits, dim=-1)
    y = torch.where(mask, y, torch.zeros_like(y))

    if (~safe).any():
        D = logits.shape[-1]
        y_fallback = torch.full_like(y, 1.0 / D)
        y = torch.where(safe, y, y_fallback)

    return y



# ----------------------------
# Fit + evaluate: CLR(X)->LR->masked softmax
# ----------------------------

def fit_and_evaluate_clr_lr_masked_softmax(
    data_train, data_valid, data_test,
    fold_num, score_fns, data_dim, hp,
    verbosity=0,
    eps_x=1e-12,     # clamp for logs on support
    ridge=0.0        # optional
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_score_fns = dict(score_fns)
    all_score_fns['mse'] = lambda y_pred, y, hp: F.mse_loss(y_pred, y, reduction='mean')

    x_list, y_list = [], []
    num_samples = 0

    for chunk in data_train.stream_by_chunk(device=device):
        x = chunk[chunked_dataset.DK_X]
        y = chunk[chunked_dataset.DK_Y]
        x_list.append(x)
        y_list.append(y)
        num_samples += x.shape[0]

    if num_samples < data_dim:
        return None

    X = torch.cat(x_list, dim=0)
    Y = torch.cat(y_list, dim=0)

    # Transform X to masked CLR space using its own support
    X_clr, _ = _clr(X, support_mask=None, eps=eps_x)

    X_aug = add_intercept(X_clr)

    xtx = X_aug.T @ X_aug
    if ridge and ridge > 0.0:
        xtx = xtx + ridge * torch.eye(xtx.shape[0], device=device, dtype=xtx.dtype)
    xty = X_aug.T @ Y  # NOTE: target is still in simplex coordinates here (as logits proxy)

    try:
        W = torch.linalg.solve(xtx, xty)      # (D+1, D)
    except RuntimeError:
        W = torch.linalg.pinv(xtx) @ xty

    def predict(x):
        # Input transform
        x_clr, mask = _clr(x, support_mask=None, eps=eps_x)  # mask inferred from x>0
        x_aug = add_intercept(x_clr)

        # Linear map produces logits-like scores per component
        logits = x_aug @ W

        # Project to simplex on the inferred support
        return masked_softmax_to_simplex(logits, mask)

    trn_scores = evaluate_stream(data_train, predict, all_score_fns, hp, device=device)
    val_scores = evaluate_stream(data_valid, predict, all_score_fns, hp, device=device)
    test_scores = evaluate_stream(data_test, predict, all_score_fns, hp, device=device)

    fold_stats = {f"train_{k}": v.item() for k, v in trn_scores.items()}
    fold_stats.update({f"valid_{k}": v.item() for k, v in val_scores.items()})
    fold_stats.update({f"test_{k}": v.item() for k, v in test_scores.items()})

    if verbosity > 0:
        print(f"\nCLR(X) -> LR(logits) -> masked softmax fold {fold_num}:")
        for k, v in val_scores.items():
            print(f"  {k}: {v:.6f}")
        print()

    return fold_stats, {"W": W, "eps_x": eps_x, "ridge": ridge}


def fit_and_evaluate_clr_mp_masked_softmax(
    data_train, data_valid, data_test,
    fold_num, score_fns, data_dim, hp,
    verbosity=0,
    eps_x=1e-12
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_score_fns = dict(score_fns)
    all_score_fns['mse'] = lambda y_pred, y, hp: F.mse_loss(y_pred, y, reduction='mean')

    x_list, y_list = [], []
    num_samples = 0

    for chunk in data_train.stream_by_chunk(device=device):
        x = chunk[chunked_dataset.DK_X]
        y = chunk[chunked_dataset.DK_Y]
        x_list.append(x)
        y_list.append(y)
        num_samples += x.shape[0]

    if num_samples >= data_dim:
        return None

    X = torch.cat(x_list, dim=0)
    Y = torch.cat(y_list, dim=0)

    X_clr, _ = _clr(X, support_mask=None, eps=eps_x)
    X_aug = add_intercept(X_clr)

    W = torch.linalg.pinv(X_aug) @ Y

    def predict(x):
        x_clr, mask = _clr(x, support_mask=None, eps=eps_x)
        logits = add_intercept(x_clr) @ W
        return masked_softmax_to_simplex(logits, mask)

    trn_scores = evaluate_stream(data_train, predict, all_score_fns, hp, device=device)
    val_scores = evaluate_stream(data_valid, predict, all_score_fns, hp, device=device)
    test_scores = evaluate_stream(data_test, predict, all_score_fns, hp, device=device)

    fold_stats = {f"train_{k}": v.item() for k, v in trn_scores.items()}
    fold_stats.update({f"valid_{k}": v.item() for k, v in val_scores.items()})
    fold_stats.update({f"test_{k}": v.item() for k, v in test_scores.items()})

    if verbosity > 0:
        print(f"\nCLR(X) -> MP(logits) -> masked softmax fold {fold_num}:")
        for k, v in val_scores.items():
            print(f"  {k}: {v:.6f}")
        print()

    return fold_stats, {"W": W, "eps_x": eps_x}


def fit_and_evaluate_clr_lr_or_mp_masked_softmax(
    data_train, data_valid, data_test,
    fold_num, score_fns, data_dim, hp,
    verbosity=0,
    eps_x=1e-12,
    ridge=0.0
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_samples = 0
    for chunk in data_train.stream_by_chunk(device=device):
        num_samples += chunk[chunked_dataset.DK_X].shape[0]

    if num_samples >= data_dim:
        return fit_and_evaluate_clr_lr_masked_softmax(
            data_train, data_valid, data_test,
            fold_num, score_fns, data_dim, hp,
            verbosity=verbosity, eps_x=eps_x, ridge=ridge
        )
    else:
        return fit_and_evaluate_clr_mp_masked_softmax(
            data_train, data_valid, data_test,
            fold_num, score_fns, data_dim, hp,
            verbosity=verbosity, eps_x=eps_x
        )





def evaluate_identity_function(data_train, data_valid, data_test, fold_num, score_fns, hp, verbosity=0):
    import torch.nn.functional as F
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_score_fns = dict(score_fns)
    all_score_fns['mse'] = lambda y_pred, y, hp: F.mse_loss(y_pred, y, reduction='mean')

    def predict_identity(x):
        return x

    trn_scores = evaluate_stream(data_train, prediction_fn=predict_identity, score_fn_dict=all_score_fns, hp=hp, device=device)
    val_scores = evaluate_stream(data_valid, prediction_fn=predict_identity, score_fn_dict=all_score_fns, hp=hp, device=device)
    test_scores = evaluate_stream(data_test, prediction_fn=predict_identity, score_fn_dict=all_score_fns, hp=hp, device=device)

    fold_stats_dict = {
        f"train_{k}": v.item() for k, v in trn_scores.items()
    }
    fold_stats_dict.update({
        f"valid_{k}": v.item() for k, v in val_scores.items()
    })
    fold_stats_dict.update({
        f"test_{k}": v.item() for k, v in test_scores.items()
    })

    if verbosity > 0:
        print(f"\nIDENTITY MODEL metrics for fold {fold_num}:")
        for k, v in val_scores.items():
            print(f"  {k}: {v:.6f}")
        print()

    return fold_stats_dict, {}

