import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt


# Early stopping function
# def early_stopping_criterion(data, patience=10, threshold=0.001): # "vanilla" early stopping
#     """
#     Determines the early stopping epoch for a given fold.
#     Criteria: Stop if validation loss does not improve by 'threshold' for 'patience' epochs.
#     """
#     epochs = data['epoch'].values
#     val_loss = data['Avg Validation Loss'].values
#
#     best_loss = float('inf')
#     epochs_no_improve = 0
#     best_epoch = 0
#
#     for i, loss in enumerate(val_loss):
#         if loss < best_loss - threshold:
#             best_loss = loss
#             best_epoch = epochs[i]
#             epochs_no_improve = 0
#         else:
#             epochs_no_improve += 1
#
#         if epochs_no_improve >= patience:
#             return epochs[best_epoch]  # Return the stopping epoch
#     return epochs[-1]  # If no stopping condition met, return the final epoch

def early_stopping_criterion(data, patience=10, threshold=0.001):  # super-optimal early stopping

    return np.argmin(data['Avg Validation Loss'].values)

def julia_early_stopping_criterion(data, patience=10, early_stopping_window=10):
    """
    Implements early stopping as described in the Julia repo.
    Stops if >50% of training examples have losses exceeding the mean loss
    of the past 'early_stopping_window' epochs for 'patience' consecutive epochs.
    """
    epochs = data['epoch'].values
    train_losses = data['Avg Training Loss'].values
    stop_counts = []
    consecutive_stop_epochs = 0

    for i in range(len(epochs)):
        if i >= early_stopping_window:
            # Compute mean loss over the early stopping window
            window_mean = np.mean(train_losses[i-early_stopping_window:i])
            # Count training examples satisfying the condition
            count = np.mean(train_losses[i] >= window_mean)
            stop_counts.append(count)

            # Check if stopping condition is met
            if count > 0.5:
                consecutive_stop_epochs += 1
                if consecutive_stop_epochs > patience:
                    return epochs[i]  # Stop here
            else:
                consecutive_stop_epochs = 0
        else:
            stop_counts.append(0)

    return epochs[-1]  # Return the final epoch if no stopping condition is met



# Load CSV files
files = glob.glob("../results/from_koa_11-14/cNODE1_cNODE-paper-ocean_fold*_epochs.csv")

# Read and concatenate all files
data_frames = []
for file in files:
    df = pd.read_csv(file)
    df['fold'] = file  # Include file info to distinguish folds
    data_frames.append(df)

data = pd.concat(data_frames)

# Calculate mean and median curves
mean_validation_loss = data.groupby('epoch')['Avg Validation Loss'].mean()
median_validation_loss = data.groupby('epoch')['Avg Validation Loss'].median()

mean_training_loss = data.groupby('epoch')['Avg Training Loss'].mean()
median_training_loss = data.groupby('epoch')['Avg Training Loss'].median()

# Determine required epochs
final_epoch = data['epoch'].max()
min_mean_val_epoch = mean_validation_loss.idxmin()
min_median_val_epoch = median_validation_loss.idxmin()

# Extract loss values
final_epoch_mean_val_loss = mean_validation_loss.loc[final_epoch]
final_epoch_median_val_loss = median_validation_loss.loc[final_epoch]
epoch500_mean_val_loss = mean_validation_loss.loc[499]
epoch500_median_val_loss = median_validation_loss.loc[499]

# Print statistics
print("Final Epoch Statistics:")
print(f"Epoch: {final_epoch}")
print(f"Mean Validation Loss: {final_epoch_mean_val_loss}")
print(f"Median Validation Loss: {final_epoch_median_val_loss}")

print("\n500th Epoch Statistics:")
print(f"Epoch: 500")
print(f"Mean Validation Loss: {epoch500_mean_val_loss}")
print(f"Median Validation Loss: {epoch500_median_val_loss}")

print("\nEpoch with Minimum Avg Validation Loss Mean:")
print(f"Epoch: {min_mean_val_epoch}")
print(f"Mean Validation Loss: {mean_validation_loss[min_mean_val_epoch]}")
print(f"Median Validation Loss: {median_validation_loss[min_mean_val_epoch]}")

print("\nEpoch with Minimum Avg Validation Loss Median:")
print(f"Epoch: {min_median_val_epoch}")
print(f"Mean Validation Loss: {mean_validation_loss[min_median_val_epoch]}")
print(f"Median Validation Loss: {median_validation_loss[min_median_val_epoch]}")

# Plotting
plt.figure(figsize=(10, 6))

# Plot mean curves
plt.plot(mean_validation_loss, label='Mean Validation Loss', linestyle='-', alpha=0.8)
plt.plot(mean_training_loss, label='Mean Training Loss', linestyle='--', alpha=0.8)

# Plot median curves
plt.plot(median_validation_loss, label='Median Validation Loss', linestyle='-', alpha=0.6)
plt.plot(median_training_loss, label='Median Training Loss', linestyle='--', alpha=0.6)

# Mark the epochs with vertical dashed lines
plt.axvline(x=min_mean_val_epoch, color='blue', linestyle='--', label='Min Mean Validation Loss Epoch')
plt.axvline(x=min_median_val_epoch, color='green', linestyle='--', label='Min Median Validation Loss Epoch')


# Highlight region where Median Validation Loss < 0.065
highlight_condition = median_validation_loss < 0.065
plt.fill_between(
    median_validation_loss.index,
    median_validation_loss,
    where=highlight_condition,
    color='red',
    alpha=0.3,
    label='Median Validation Loss < 0.065'
)

# Show the plot
plt.title("Validation and Training Loss Across Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

# Early stopping results
early_stopping_epochs = []
for fold, fold_data in data.groupby('fold'):
    stop_epoch = early_stopping_criterion(fold_data)
    early_stopping_epochs.append((fold, int(stop_epoch)))

# Evaluate mean and median at early stopping
early_stopping_losses = []
for fold, stop_epoch in early_stopping_epochs:
    fold_data = data[data['fold'] == fold]
    early_stopping_losses.append(fold_data[fold_data['epoch'] == stop_epoch]['Avg Validation Loss'].values[0])

# Unzip the epochs from the tuples
_, early_stopping_epoch_values = zip(*early_stopping_epochs)

early_stopping_mean = np.mean(early_stopping_losses)
early_stopping_median = np.median(early_stopping_losses)
early_stopping_avg_epoch = np.mean(early_stopping_epoch_values)

# Early stopping results
julia_early_stopping_epochs = []
for fold, fold_data in data.groupby('fold'):
    stop_epoch = julia_early_stopping_criterion(fold_data)
    julia_early_stopping_epochs.append((fold, int(stop_epoch)))

# Evaluate mean and median at Julia early stopping
julia_early_stopping_losses = []
for fold, stop_epoch in julia_early_stopping_epochs:
    fold_data = data[data['fold'] == fold]
    julia_early_stopping_losses.append(fold_data[fold_data['epoch'] == stop_epoch]['Avg Validation Loss'].values[0])

# Unzip the epochs from the tuples
_, julia_early_stopping_epoch_values = zip(*julia_early_stopping_epochs)

julia_early_stopping_mean = np.mean(julia_early_stopping_losses)
julia_early_stopping_median = np.median(julia_early_stopping_losses)
julia_early_stopping_avg_epoch = np.mean(julia_early_stopping_epoch_values)


print("\nIdealized Early Stopping Statistics:")
print(f"Average Epoch: {early_stopping_avg_epoch}")
print(f"Mean Validation Loss at Early Stopping: {early_stopping_mean}")
print(f"Median Validation Loss at Early Stopping: {early_stopping_median}")

print("\nJulia Early Stopping Statistics:")
print(f"Average Epoch: {julia_early_stopping_avg_epoch}")
print(f"Mean Validation Loss at Early Stopping: {julia_early_stopping_mean}")
print(f"Median Validation Loss at Early Stopping: {julia_early_stopping_median}")
