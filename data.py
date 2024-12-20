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
    # Ensure input is a torch tensor
    if peak_stddev <= 0:
        return mean_values
    
    if not isinstance(mean_values, torch.Tensor):
        raise TypeError("mean_values must be a torch.Tensor")
    
    # Replace 0s and 1s with 0.5 temporarily to avoid issues in the Beta distribution
    adjusted_means = torch.where(mean_values <= 0, torch.full_like(mean_values, 0.5), mean_values)
    adjusted_means = torch.where(mean_values >= 1, torch.full_like(mean_values, 0.5), adjusted_means)
    
    # Mask values that are within the range (0, 1)
    valid_mask = (mean_values > 0) & (mean_values < 1)
    
    # Calculate the standard deviation for each valid mean value
    stddev = torch.sin(adjusted_means * torch.pi) * peak_stddev
    
    # Calculate the variance
    var = stddev ** 2
    
    # Calculate the Beta distribution parameters alpha and beta
    k = adjusted_means * (1.0 - adjusted_means) / var - 1.0
    
    alpha = adjusted_means * k
    beta_param = (1 - adjusted_means) * k
    
    if torch.any(torch.isnan(alpha)) or torch.any(torch.isnan(beta_param)):
        print(f"NaN detected in alpha or beta param")
        # find the index of the NaN value
        nan_idx = torch.isnan(alpha) | torch.isnan(beta_param)
        # print(f"NaN index: {mean_values[nan_idx]}")
        print(f"mean_values with NaN: {mean_values[nan_idx]}")
    
    # Sample from the Beta distribution for each mean value that is within (0, 1)
    sampled_values_within_range = torch.distributions.Beta(alpha, beta_param).sample()
    
    # Only update the sampled values for elements where 0 < mean < 1
    sampled_values = torch.where(valid_mask, sampled_values_within_range, mean_values)
    
    # Restore 0s and 1s to their original values
    sampled_values = torch.where(mean_values <= 0, torch.zeros_like(mean_values), sampled_values)
    sampled_values = torch.where(mean_values >= 1, torch.ones_like(mean_values), sampled_values)
    
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


import torch


def get_batch(x, y, t, mb_size, current_index, noise_level_x=0.0, noise_level_y=0.0, interpolate_noise=False):
    """Returns a batch of data of size `mb_size` starting from `current_index`,
    with optional noise augmentation for x and y, and returns a tensor z with interpolation according to t."""
    
    # Calculate end index for the batch
    end_index = current_index + mb_size
    if end_index > x.size(0):
        end_index = x.size(0)
    
    # Get indices for the batch
    batch_indices = torch.arange(current_index, end_index, dtype=torch.long)
    x_batch = x[batch_indices, :]
    y_batch = y[batch_indices, :]
    
    # Optionally apply noise and normalization to x_batch and y_batch
    if interpolate_noise:
        if noise_level_x > 0:
            x_batch = resample_noisy(x_batch, noise_level_x)
            x_batch = normalize(x_batch)
        
        if noise_level_y > 0:
            y_batch = resample_noisy(y_batch, noise_level_y)
            y_batch = normalize(y_batch)
    
    # Create tensor z with an extra outside dimension with the same length as t
    t = t.view(-1, 1, 1)  # Reshape t to (len(t), 1, 1) for broadcasting
    
    # Interpolate linearly between x_batch and y_batch according to t
    z = (1 - t) * x_batch.unsqueeze(0) + t * y_batch.unsqueeze(0)  # Shape: (len(t), mb_size, x.size(1))

    # Optionally apply noise to all elements of z
    if not interpolate_noise:
        # Interpolate noise levels according to t, and use to independently resample each timestep of z
        t = t.squeeze()
        noise_level = (1 - t) * noise_level_x + t * noise_level_y
        for i in range(z.size(0)):
            z[i] = resample_noisy(z[i], noise_level[i].item())
            z[i] = normalize(z[i])
    
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