import numpy as np
import torch
from sklearn.model_selection import KFold

import torch

import torch


def resample_noisy(mean_values: torch.Tensor, peak_stddev: float) -> torch.Tensor:
    """
    For each element in the input tensor, use it as the mean to sample a single point
    from a parameterized Beta distribution.

    Args:
        mean_values (torch.Tensor): The input tensor of mean values.
        peak_stddev (float): The peak standard deviation to control the variance of the Beta distribution.

    Returns:
        torch.Tensor: A tensor of the same shape as mean_values, containing sampled values from the Beta distribution.
    """
    if peak_stddev <= 0:
        return mean_values
    
    if not isinstance(mean_values, torch.Tensor):
        raise TypeError("mean_values must be a torch.Tensor")
        
        # Handle edge cases for mean values near 0 and 1
    near_zero_mask = mean_values <= 0
    near_one_mask = mean_values >= 1
    
    # Replace invalid values with temporary safe values for Beta computation
    safe_means = mean_values.clone()
    safe_means[near_zero_mask] = 0.5
    safe_means[near_one_mask] = 0.5
    
    # Compute standard deviation and variance
    stddev = torch.sin(safe_means * torch.pi) * peak_stddev
    var = stddev.square()
    
    # Compute Beta parameters
    k = safe_means * (1 - safe_means) / var - 1
    alpha = safe_means * k
    beta_param = (1 - safe_means) * k
    
    # Handle invalid Beta parameters (NaNs or negative values)
    valid_mask = (alpha > 0) & (beta_param > 0)
    alpha[~valid_mask] = 1.0
    beta_param[~valid_mask] = 1.0
    
    # Sample values from Beta distribution
    beta_dist = torch.distributions.Beta(alpha, beta_param)
    sampled_values = beta_dist.sample()
    
    # Restore original 0 and 1 for edge cases
    sampled_values[near_zero_mask] = 0.0
    sampled_values[near_one_mask] = 1.0
    
    return sampled_values



def normalize(x, transposed=False):
    # Normalizes the input to sum to 1 along the correct axis
    if transposed:
        # For transposed input, normalize along the second-to-last axis
        axis = -2 if x.ndim > 1 else 0
        return x / x.sum(axis=axis, keepdims=True)
    else:
        # Normalize along the last axis
        return x / x.sum(axis=-1, keepdims=True)


def process_data(y0):
    # produces X (assemblage) from Y (composition), normalizes the composition to sum to 1, and transposes the data
    y = y0.copy()
    x = y0.copy()
    x[x > 0] = 1
    y = normalize(y, transposed=True)
    x = normalize(x, transposed=True)
    y = y.astype(np.float32)
    x = x.astype(np.float32)
    if (np.sum(np.abs(y0 - y)) > y.shape[1]/25.0):
        print('WARNING: input columns are not distributions. Is the data transposed?')
    y = torch.from_numpy(y.T)
    x = torch.from_numpy(x.T)
    return x, y


def load_data(filepath_train, device):
    # Load data
    y = np.loadtxt(filepath_train, delimiter=',')
    x, y = process_data(y)  # Assuming process_data is defined somewhere
    
    # Move data to device if specified
    if device:
        x = x.to(device)
        y = y.to(device)
    
    return x, y


def fold_data(x, y, k=5):
    if k < 0:  # Hold out 1 sample for negative values
        k = x.size(0)
    
    # Split data into k folds
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    
    fold_data = []
    
    for train_index, valid_index in kf.split(x):
        x_train, x_valid = x[train_index], x[valid_index]
        y_train, y_valid = y[train_index], y[valid_index]
        
        fold_data.append((x_train, y_train, x_valid, y_valid))
    
    return fold_data


def check_leakage(folded_data):
    """
    Checks for data leakage by verifying if any fold contains the same data row repeated in both y_train and y_valid.
    """
    for fold_idx, (x_train, y_train, x_valid, y_valid) in enumerate(folded_data):
        # Convert y_train and y_valid to sets for fast comparison
        y_train_set = set(y_train)
        y_valid_set = set(y_valid)
        
        # Find intersection between y_train and y_valid
        leakage = y_train_set.intersection(y_valid_set)
        
        if leakage:
            print(f"LEAKAGE DETECTED in fold {fold_idx}: {leakage}")
            return False
    
    # If no leakage detected in any fold
    # print("No leakage detected in any fold.")
    return True


def get_batch(x, y, t, mb_size, current_index, noise_level_x=0.0, noise_level_y=0.0, interpolate_noise=False):
    """Returns a batch of data of size `mb_size` starting from `current_index`,
    with optional noise augmentation for x and y, and returns a tensor z with interpolation according to t."""
    end_index = min(current_index + mb_size, x.size(0))
    batch_indices = torch.arange(current_index, end_index, dtype=torch.long, device=x.device)
    x_batch = x[batch_indices]
    y_batch = y[batch_indices]

    if interpolate_noise:
        if noise_level_x > 0:
            x_batch = normalize(resample_noisy(x_batch, noise_level_x))
        if noise_level_y > 0:
            y_batch = normalize(resample_noisy(y_batch, noise_level_y))

    # Prepare for interpolation
    t = t.view(-1, 1, 1)  # Reshape t to (len(t), 1, 1) for broadcasting
    z = (1 - t) * x_batch.unsqueeze(0) + t * y_batch.unsqueeze(0)

    if not interpolate_noise:
        # Interpolate noise levels and apply noise to z
        noise_level = (1 - t.squeeze()) * noise_level_x + t.squeeze() * noise_level_y
        for i in range(z.size(0)):
            z[i] = normalize(resample_noisy(z[i], noise_level[i].item()))

    return z, end_index



def shuffle_data(x, y):
    # Assuming x and y are numpy arrays or PyTorch tensors
    assert len(x) == len(y)
    
    # If x and y are PyTorch tensors
    if isinstance(x, torch.Tensor):
        permutation = torch.randperm(len(x))
        x = x[permutation]
        y = y[permutation]
    else:  # If x and y are numpy arrays
        permutation = np.random.permutation(len(x))
        x = x[permutation]
        y = y[permutation]
    
    return x, y


def shuffle_tensor(z, dim=-2):
    """Shuffles a tensor along the specified dimension `dim`."""
    perm = torch.randperm(z.size(dim))
    # Use 'slice(None)' to keep other dimensions unchanged
    idx = [slice(None)] * len(z.shape)
    idx[dim] = perm  # Replace the corresponding dimension with permuted indices
    return z[tuple(idx)]