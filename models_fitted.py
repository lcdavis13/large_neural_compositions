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





def fit_and_evaluate_linear_regression(data_train, data_valid, data_test, fold_num, score_fns, data_dim, hp, verbosity=0):
    import torch
    import torch.nn.functional as F
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
        return None  # Undertermined system: skip

    X = torch.cat(x_list, dim=0)
    Y = torch.cat(y_list, dim=0)

    xtx = X.T @ X
    xty = X.T @ Y
    try:
        weights = torch.linalg.solve(xtx, xty)
    except RuntimeError:
        weights = torch.linalg.pinv(xtx) @ xty

    def predict(x):
        return x @ weights

    trn_scores = evaluate_stream(data_train, prediction_fn=predict, score_fn_dict=all_score_fns, hp=hp, device=device)
    val_scores = evaluate_stream(data_valid, prediction_fn=predict, score_fn_dict=all_score_fns, hp=hp, device=device)
    test_scores = evaluate_stream(data_test, prediction_fn=predict, score_fn_dict=all_score_fns, hp=hp, device=device)

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
        print(f"\nLINEAR REGRESSION metrics for fold {fold_num}:")
        for k, v in val_scores.items():
            print(f"  {k}: {v:.6f}")
        print()

    return fold_stats_dict, {}


def fit_and_evaluate_moore_penrose(data_train, data_valid, data_test, fold_num, score_fns, data_dim, verbosity=0):
    import torch
    import torch.nn.functional as F
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_score_fns = dict(score_fns)
    all_score_fns['mse'] = lambda y_pred, y: F.mse_loss(y_pred, y, reduction='mean')

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

    weights = torch.linalg.pinv(X) @ Y

    def predict(x):
        return x @ weights

    trn_scores = evaluate_stream(data_train, prediction_fn=predict, score_fn_dict=all_score_fns, device=device)
    val_scores = evaluate_stream(data_valid, prediction_fn=predict, score_fn_dict=all_score_fns, device=device)
    test_scores = evaluate_stream(data_test, prediction_fn=predict, score_fn_dict=all_score_fns, device=device)

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
        print(f"\nMOORE-PENROSE REGRESSION metrics for fold {fold_num}:")
        for k, v in val_scores.items():
            print(f"  {k}: {v:.6f}")
        print()

    return fold_stats_dict, {}


def fit_and_evaluate_lr_or_mp(data_train, data_valid, data_test, fold_num, score_fns, data_dim, hp, verbosity=0):
    # First, count number of samples
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_samples = 0

    for chunk in data_train.stream_by_chunk(device=device):
        x = chunk[chunked_dataset.DK_X]
        num_samples += x.shape[0]

    if num_samples >= data_dim:
        return fit_and_evaluate_linear_regression(data_train, data_valid, data_test, fold_num, score_fns, data_dim, hp, verbosity)
    else:
        return fit_and_evaluate_moore_penrose(data_train, data_valid, data_test, fold_num, score_fns, data_dim, hp, verbosity)

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

