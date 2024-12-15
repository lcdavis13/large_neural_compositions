import numpy as np
import torch
from sklearn.model_selection import KFold

def add_noise(data, noise_level=0.01):
    """Adds Gaussian noise to the data."""
    noise = np.random.normal(0, noise_level, data.shape).astype(np.float32)
    return data + noise

def process_data(y0):
    # produces X (assemblage) from Y (composition), normalizes the composition to sum to 1, and transposes the data
    y = y0.copy()
    x = y0.copy()
    x[x > 0] = 1
    y = y / y.sum(axis=0)[np.newaxis, :]
    x = x / x.sum(axis=0)[np.newaxis, :]
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


def get_batch(x, y, mb_size, current_index, augment_x=False, augment_y=False, noise_level_x=0.01, noise_level_y=0.01):
    """Returns a batch of data of size `mb_size` starting from `current_index`, with optional noise augmentation for x and y."""
    end_index = current_index + mb_size
    if end_index > x.size(0):
        end_index = x.size(0)
    
    batch_indices = torch.arange(current_index, end_index, dtype=torch.long)
    x_batch = x[batch_indices, :]
    y_batch = y[batch_indices, :]
    
    if augment_x:
        x_batch = add_noise(x_batch, noise_level_x)
        
    if augment_y:
        y_batch = add_noise(y_batch, noise_level_y)
    
    return x_batch, y_batch, end_index


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