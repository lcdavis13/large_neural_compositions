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
    if transposed:
        axis = -2 if x.ndim > 1 else -1
    else:
        axis = -1

    # check for any rows of all zero
    if (x.sum(axis=axis) == 0).any():
        print('WARNING: some input rows are all zero')
        # print indices
        print(np.where(x.sum(axis=axis) == 0))

    # Normalizes the input to sum to 1 along the correct axis
    return x / x.sum(axis=axis, keepdims=True)


def process_data(y0, transpose=False):
    # produces X (assemblage) from Y (composition), normalizes the composition to sum to 1, and transposes the data
    y = y0.copy()
    x = y0.copy()
    x[x > 0] = 1
    y = normalize(y, transposed=transpose)
    x = normalize(x, transposed=transpose)
    y = y.astype(np.float32)
    x = x.astype(np.float32)
    if (np.sum(np.abs(y0 - y)) > y.shape[1]/25.0):
        print('WARNING: input columns are not distributions. Is the data transposed?')
    y = torch.from_numpy(y.T if transpose else y)
    x = torch.from_numpy(x.T if transpose else x)
    return x, y


def load_data(filepath_train, filepath_train_pos, filepath_train_val, device, subset=-1):
    # Load data
    y = np.loadtxt(filepath_train, delimiter=',')
    ycon = np.loadtxt(filepath_train_val, delimiter=',')
    idcon = np.loadtxt(filepath_train_pos, delimiter=',')
    idcon = torch.from_numpy(idcon).long()
    print("uncondensed")
    x, y = process_data(y, transpose=False) # No longer transposing data, ensure that all data is in correct format (samples as rows) in pre-processing
    print("condensed")
    xcon, ycon = process_data(ycon, transpose=False)
    
    data_fraction = 1.0
    if subset > 0:
        if subset > len(x):
            print("\n============\nWARNING: subset of data was requested, but exceeds number of samples in dataset\n============\n")

        data_fraction = subset / len(x)
        x = x[:subset]
        y = y[:subset]
        xcon = xcon[:subset]
        ycon = ycon[:subset]
        idcon = idcon[:subset]

    # print(f"shapes:\n x={x.shape}\n y={y.shape}\n xcon={xcon.shape}\n ycon={ycon.shape}\n idcon={idcon.shape}")
    
    # Move data to device if specified
    if device:
        x = x.to(device)
        y = y.to(device)
        xcon = xcon.to(device)
        ycon = ycon.to(device)
        idcon = idcon.to(device)
    
    return x, y, xcon, ycon, idcon, data_fraction


def split_data(datasets, validation_samples):
    # output dimensions are (kfolds (singleton for this function), datasets, train vs valid, ...)

    # Ensure datasets is a list of tensors/arrays with matching lengths
    dataset_lengths = [len(dataset) for dataset in datasets]
    if not all(length == dataset_lengths[0] for length in dataset_lengths):
        raise ValueError("All datasets must have the same length.")
    
    # Split data into training and validation sets
    data = []
    for dataset in datasets:
        train_data = dataset[:-validation_samples]
        valid_data = dataset[-validation_samples:]
        data.append((train_data, valid_data))

    return [data]

    


def fold_data(datasets, k=5):
    # output dimensions are (kfolds, datasets, train vs valid, ...)

    # Ensure datasets is a list of tensors/arrays with matching lengths
    dataset_lengths = [len(dataset) for dataset in datasets]
    if not all(length == dataset_lengths[0] for length in dataset_lengths):
        raise ValueError("All datasets must have the same length.")
    
    if k < 0:  # Leave-One-Out if k is negative
        k = dataset_lengths[0]

    # Split data into k folds
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    fold_data = []

    for train_index, valid_index in kf.split(datasets[0]):
        fold = []
        for dataset in datasets:
            train_split = dataset[train_index]
            valid_split = dataset[valid_index]
            fold.append((train_split, valid_split))
        fold_data.append(fold)

    return fold_data



def check_leakage(folded_data):
    """
    Checks for data leakage by verifying if any fold contains the same data row repeated in both y_train and y_valid.
    """
    for fold_idx, datasets in enumerate(folded_data):
        # Unpack datasets
        y_train = datasets[1][0]
        y_valid = datasets[1][1]

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


def get_batch(data, t, mb_size, current_index, noise_level_x=0.0, noise_level_y=0.0, interpolate_noise=False, requires_timesteps=True):
    """Returns a batch of data of size `mb_size` starting from `current_index`,
    with optional noise augmentation for x and y, and returns a tensor z with interpolation according to t."""
    if len(data) == 3:
        x, y, ids = data
    else:
        x, y = data
        ids = None
    
    end_index = min(current_index + mb_size, x.size(0))
    batch_indices = torch.arange(current_index, end_index, dtype=torch.long, device=x.device)
    x_batch = x[batch_indices]
    y_batch = y[batch_indices]
    ids_batch = ids[batch_indices] if ids is not None else None
    
    if not requires_timesteps:
        t = torch.tensor([t[0], t[-1]]).to(x.device)

    # interpolated (not independent) noise
    if interpolate_noise:
        if noise_level_x > 0:
            x_batch = normalize(resample_noisy(x_batch, noise_level_x))
        if noise_level_y > 0:
            y_batch = normalize(resample_noisy(y_batch, noise_level_y))

    # Interpolate
    t = t.view(-1, 1, 1)  # Reshape t to (len(t), 1, 1) for broadcasting
    z = (1 - t) * x_batch.unsqueeze(0) + t * y_batch.unsqueeze(0)

    # Independent noise
    if not interpolate_noise:
        # Interpolate noise levels and apply noise to z
        noise_level = (1 - t.squeeze()) * noise_level_x + t.squeeze() * noise_level_y
        for i in range(z.size(0)):
            z[i] = normalize(resample_noisy(z[i], noise_level[i].item()))

    return z, ids_batch, end_index



def shuffle_data(datasets):
    # Ensure datasets is a list of tensors/arrays with matching lengths
    dataset_lengths = [len(dataset) for dataset in datasets]
    if not all(length == dataset_lengths[0] for length in dataset_lengths):
        raise ValueError("All datasets must have the same length.")

    # Determine type and create permutation
    if isinstance(datasets[0], torch.Tensor):
        permutation = torch.randperm(len(datasets[0]))
    else:  # Assuming numpy arrays
        permutation = np.random.permutation(len(datasets[0]))

    # Shuffle all datasets using the same permutation
    shuffled_datasets = [dataset[permutation] for dataset in datasets]

    return shuffled_datasets



def shuffle_tensor(z, dim=-2):
    """Shuffles a tensor along the specified dimension `dim`."""
    perm = torch.randperm(z.size(dim))
    # Use 'slice(None)' to keep other dimensions unchanged
    idx = [slice(None)] * len(z.shape)
    idx[dim] = perm  # Replace the corresponding dimension with permuted indices
    return z[tuple(idx)]