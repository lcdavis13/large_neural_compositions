import numpy as np
import torch
from sklearn.model_selection import KFold

def process_data(y):
    # produces X (assemblage) from Y (composition), normalizes the composition to sum to 1, and transposes the data
    x = y.copy()
    x[x > 0] = 1
    y = y / y.sum(axis=0)[np.newaxis, :]
    x = x / x.sum(axis=0)[np.newaxis, :]
    y = y.astype(np.float32)
    x = x.astype(np.float32)
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
    # Split data into k folds
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    
    fold_data = []
    
    for train_index, valid_index in kf.split(x):
        x_train, x_valid = x[train_index], x[valid_index]
        y_train, y_valid = y[train_index], y[valid_index]
        
        fold_data.append((x_train, y_train, x_valid, y_valid))
    
    return fold_data


def get_batch(x, y, mb_size, current_index):
    end_index = current_index + mb_size
    if end_index > x.size(0):
        end_index = x.size(0)
    batch_indices = torch.arange(current_index, end_index, dtype=torch.long)
    x_batch = x[batch_indices, :]
    y_batch = y[batch_indices, :]
    # print(f'x {x_batch.shape}')
    # print(f'x {x_batch[0, :]}')
    # print(f'y {y_batch.shape}')
    # print(f'y {y_batch[0, :]}')
    return x_batch, y_batch, end_index
