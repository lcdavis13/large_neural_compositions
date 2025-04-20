import os
import numpy as np
import torch
from sklearn.linear_model import LinearRegression

# Configuration
x_dataset = "256"
y_dataset = "256-random"
x_datafile_path_prefix = f"data/{x_dataset}/{x_dataset}_x0"
y_datafile_path_prefix = f"data/{x_dataset}/{y_dataset}_y"
val_samples = 1000
train_subset = 1000

# Bray-Curtis Dissimilarity loss
def loss(y_pred, y_true):
    return torch.mean(torch.sum(torch.abs(y_pred - y_true), dim=-1) /
                      torch.sum(torch.abs(y_pred) + torch.abs(y_true), dim=-1))

# Count total available samples in all chunks
def count_total_samples(datafile_path_prefix):
    total = 0
    i = 0
    while True:
        try:
            data = np.loadtxt(f"{datafile_path_prefix}_{i}.csv", delimiter=",")

            total += data.shape[0]
            i += 1
        except FileNotFoundError:
            break
    return total

# Load sequential samples across chunk files
def load_samples(start, count, datafile_path_prefix):
    samples = []
    i = 0
    loaded = 0
    offset = start
    while loaded < count:
        file_path = f"{datafile_path_prefix}_{i}.csv"

        if not os.path.exists(file_path):
            break
        data = np.loadtxt(f"{datafile_path_prefix}_{i}.csv", delimiter=",")

        file_len = data.shape[0]

        if offset >= file_len:
            offset -= file_len
            i += 1
            continue

        take = min(count - loaded, file_len - offset)
        samples.append(data[offset:offset + take])
        loaded += take
        offset = 0
        i += 1
    return np.concatenate(samples, axis=0)

# Run pipeline
def main(): 
    x_chunks = count_total_samples(x_datafile_path_prefix)
    y_chunks = count_total_samples(y_datafile_path_prefix)
    print(f"X samples available: {x_chunks}")
    print(f"Y samples available: {y_chunks}")

    num_samples_available = x_chunks  # assume x and y are aligned
    samples_to_load = val_samples + train_subset
    train_samples = min(samples_to_load, num_samples_available) - val_samples

    # Load validation data
    x_val = load_samples(0, val_samples, x_datafile_path_prefix)
    y_val = load_samples(0, val_samples, y_datafile_path_prefix)

    # Load training data
    x_train = load_samples(val_samples, train_samples, x_datafile_path_prefix)
    y_train = load_samples(val_samples, train_samples, y_datafile_path_prefix)

    # Fit linear regression
    model = LinearRegression()
    model.fit(x_train, y_train)

    # Evaluate on validation set
    y_pred_val = model.predict(x_val)
    y_pred_tensor = torch.tensor(y_pred_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

    val_loss = loss(y_pred_tensor, y_val_tensor)
    print(f"Validation Bray-Curtis Loss: {val_loss.item():.4f}")

if __name__ == "__main__":
    main()

