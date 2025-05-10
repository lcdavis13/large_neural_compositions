import torch
import chunked_dataset
import torch.nn.functional as F


def evaluate_stream(dataset, prediction_fn, score_fn_dict, device=None):
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
                batch_score = fn(y_pred, y) * batch_size
                total_scores[name] += batch_score

            total_samples += batch_size

    avg_scores = {
        name: total / total_samples if total_samples > 0 else float("nan")
        for name, total in total_scores.items()
    }
    return avg_scores





def fit_and_evaluate_linear_regression(data_train, data_valid, data_test, fold_num, score_fns, data_dim, verbosity=0):
    import torch.nn.functional as F  # For MSE
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Copy and extend score function dictionary with MSE
    all_score_fns = dict(score_fns)  # avoid mutating original
    all_score_fns['mse'] = lambda y_pred, y: F.mse_loss(y_pred, y, reduction='mean')

    # Accumulate
    xtx = torch.zeros((data_dim, data_dim), device=device)
    xty = torch.zeros((data_dim, data_dim), device=device)

    for chunk in data_train.stream_by_chunk(device=device):
        x = chunk[chunked_dataset.DK_X]
        y = chunk[chunked_dataset.DK_Y]
        xtx += x.T @ x
        xty += x.T @ y

    try:
        weights = torch.linalg.solve(xtx, xty)
    except RuntimeError:
        weights = torch.linalg.pinv(xtx) @ xty

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
        print(f"\nCLOSED-FORM LINEAR REGRESSION metrics for fold {fold_num}:")
        for k, v in val_scores.items():
            print(f"  {k}: {v:.6f}")
        print()

    return fold_stats_dict, {}



def evaluate_identity_function(data_train, data_valid, data_test, fold_num, score_fns, verbosity=0):
    import torch.nn.functional as F
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_score_fns = dict(score_fns)
    all_score_fns['mse'] = lambda y_pred, y: F.mse_loss(y_pred, y, reduction='mean')

    def predict_identity(x):
        return x

    trn_scores = evaluate_stream(data_train, prediction_fn=predict_identity, score_fn_dict=all_score_fns, device=device)
    val_scores = evaluate_stream(data_valid, prediction_fn=predict_identity, score_fn_dict=all_score_fns, device=device)
    test_scores = evaluate_stream(data_test, prediction_fn=predict_identity, score_fn_dict=all_score_fns, device=device)

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

