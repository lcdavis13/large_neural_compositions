

import torch


def get_loss_functions():
    # specify loss function
    loss_fn = loss_bc
    # avg_richness = x.count_nonzero()/x.size(0)
    # loss_fn = lambda y_pred, y_true: loss_bc_unbounded(y_pred, y_true, avg_richness)
    score_fn = loss_bc
    # loss_fn = lambda y_pred,y_true: loss_bc(y_pred, y_true) + distribution_error(y_pred)
    
    distr_error_fn = distribution_error

    score_fns = {"score": score_fn, "simplex_distance": distr_error_fn}

    return loss_fn, score_fns




def loss_bc_dki(y_pred, y_true):
    return torch.sum(torch.abs(y_pred - y_true)) / torch.sum(
        torch.abs(y_pred + y_true))  # DKI repo implementation (incorrect)


def loss_bc(y_pred, y_true):  # Bray-Curtis Dissimilarity
    return torch.mean(torch.sum(torch.abs(y_pred - y_true), dim=-1) / torch.sum(torch.abs(y_pred) + torch.abs(y_true), dim=-1))


def loss_logbc(y_pred, y_true):  # Bray-Curtis Dissimilarity on log-transformed data to emphasize loss of rare species
    return loss_bc(torch.log(y_pred + 1), torch.log(y_true + 1))


def loss_loglogbc(y_pred, y_true):  # Bray-Curtis Dissimilarity on log-log-transformed data to emphasize loss of rare species even more
    return loss_logbc(torch.log(y_pred + 1), torch.log(y_true + 1))


def loss_bc_old(y_pred, y_true):  # Bray-Curtis Dissimilarity
    return torch.sum(torch.abs(y_pred - y_true)) / torch.sum(torch.abs(y_pred) + torch.abs(y_true))


def loss_bc_scaled(y_pred, y_true, epsilon=1e-10):
    numerator = torch.sum(torch.abs(y_pred - y_true) / (torch.abs(y_true) + epsilon), dim=-1)
    denominator = torch.sum(torch.abs(y_pred) + torch.abs(y_true) / (torch.abs(y_true) + epsilon), dim=-1)
    return torch.mean(numerator / denominator)


def loss_bc_root(y_pred, y_true):
    return torch.sqrt(loss_bc(y_pred, y_true))


def loss_bc_logscaled(y_pred, y_true, epsilon=1e-10):
    numerator = torch.sum(torch.abs(y_pred - y_true) / torch.log(torch.abs(y_true) + 1 + epsilon))
    denominator = torch.sum(torch.abs(y_pred) + torch.abs(y_true) / torch.log(torch.abs(y_true) + 1 + epsilon))
    return numerator / denominator


def loss_bc_unbounded(y_pred, y_true, avg_richness, epsilon=1e-10):
    # performs the normalization per element, such that if y_pred has an extra elemen, it adds an entire 1 to the loss. This avoids the "free lunch" of adding on extra elements with small value.
    batch_loss = torch.sum(torch.div(torch.abs(y_pred - y_true), torch.abs(y_pred) + torch.abs(y_true) + epsilon))
    batch_loss = batch_loss / avg_richness
    return batch_loss / y_pred.shape[0]


# def distribution_error(x, y=None):  # penalties for invalid distributions. y is unused but included to match signautre of other score functions.
#     a = 1.0
#     b = 1.0
#     feature_penalty = torch.sum(torch.clamp(torch.abs(x - 0.5) - 0.5, min=0.0))  # each feature penalized for distance from range [0,1]. Currently not normalized.
#     sum_penalty = torch.sum(torch.abs(torch.sum(x, dim=-1) - 1.0))  # sum penalized for distance from 1.0
#     # normalize by the product of all dimensions except the final one?
#     return a * feature_penalty + b * sum_penalty


def distribution_error(x, y=None):  # penalties for invalid distributions. y is unused but included to match signautre of other score functions.
    """
    For a batch of vectors, compute the Euclidean distance from each to the probability simplex.
    
    Args:
        batch (Tensor): shape (B, N), where B is batch size and N is vector length.
    
    Returns:
        distances (Tensor): shape (B,), Euclidean distance of each vector to the simplex.
        Note that this is an L2 distance, not L1, so not directly comparable to Bray-Curtis in magnitude. Finding the minimum L1 distance to the simplex would be more complex.
    """

    def project_onto_simplex(v, axis=-1):
        """
        Projects each vector in `v` onto the probability simplex along the specified axis.
        """
        v_sorted, _ = torch.sort(v, descending=True, dim=axis)
        cssv = torch.cumsum(v_sorted, dim=axis) - 1
        ind = torch.arange(1, v.size(axis)+1, device=v.device).view([1]*axis + [-1])
        cond = v_sorted - cssv / ind > 0

        rho = cond.cumsum(dim=axis)
        rho[cond == 0] = 0
        rho_max, _ = rho.max(dim=axis, keepdim=True)

        theta = (cssv.gather(axis, rho_max - 1)) / rho_max.type(v.dtype)
        return torch.clamp(v - theta, min=0.0)

    projected = project_onto_simplex(x)
    distances = torch.norm(x - projected, dim=1)
    return torch.mean(distances)



