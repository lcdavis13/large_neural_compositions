import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import numpy as np

# Define the folder containing the CSV files
folder_path = '../results/logs/'

# Get all CSV files in the folder that end with "_epochs.csv"
csv_files = []
# csv_files += glob.glob(os.path.join(folder_path, 'canODE_*ocean*_epochs.csv'))
# csv_files += glob.glob(os.path.join(folder_path, 'canODE-noValue_*ocean*_epochs.csv'))
# csv_files += glob.glob(os.path.join(folder_path, 'canODE-multihead_*ocean*_epochs.csv'))
# csv_files += glob.glob(os.path.join(folder_path, 'canODE-singlehead_*ocean*_epochs.csv'))
# csv_files += glob.glob(os.path.join(folder_path, 'canODE-transformer_*ocean*_epochs.csv'))
# csv_files += glob.glob(os.path.join(folder_path, 'baseline-cNODE0_*ocean*_epochs.csv'))
# csv_files += glob.glob(os.path.join(folder_path, 'cNODE1_*ocean*_epochs.csv'))
# csv_files += glob.glob(os.path.join(folder_path, 'cNODE2_*ocean*_epochs.csv'))
csv_files += glob.glob(os.path.join(folder_path, 'baseline*waimea_epochs.csv'))
csv_files += glob.glob(os.path.join(folder_path, 'cNODE2-custom*waimea_epochs.csv'))

# Initialize a dictionary to hold data from all files
data = {}

# Define line styles for different folds
line_styles = ['-', '--', '-.', ':']

# Loop over all CSV files
for file in csv_files:
    # Load the CSV file into a DataFrame
    df = pd.read_csv(file)
    
    # Extract the model identifier from the file name (assuming it's part of the file name)
    model_name = os.path.basename(file).split('_')[0]
    
    # Initialize an empty list for storing data for each fold
    if model_name not in data:
        data[model_name] = []
    
    # # Filter for fold=0 and epoch <= 60
    # df = df[(df['fold'] == 1) & (df['epoch'] <= 50)]
    
    # Group by fold and extract epoch, Avg Validation Loss, @ Training Loss, and Elapsed Time
    for fold, group in df.groupby('fold'):
        epochs = group['epoch'].to_numpy()
        val_loss = group['Avg Validation Loss'].to_numpy()
        train_loss = group['Avg Training Loss'].to_numpy()
        elapsed_time = group['Elapsed Time'].to_numpy()
        
        # Exclude epoch 0 for training loss and epochs
        mask = epochs != 0
        epochs_filtered = epochs[mask]
        elapsed_time_filtered = elapsed_time[mask]
        train_loss_filtered = train_loss[mask]
        
        data[model_name].append(
            (epochs, val_loss, train_loss_filtered, epochs_filtered, elapsed_time, elapsed_time_filtered, fold))

# Optional variable to enable averaging of data
average_data = False


# Function to interpolate and average data
def interpolate_and_average(data_list, x_key, y_key):
    all_x = sorted(set(itertools.chain.from_iterable([data[x_key] for data in data_list])))
    all_y_interpolated = []
    
    for data in data_list:
        y_interpolated = np.interp(all_x, data[x_key], data[y_key])
        all_y_interpolated.append(y_interpolated)
    
    avg_y = np.mean(all_y_interpolated, axis=0)
    return all_x, avg_y


# Function to break lines when x-coordinate decreases
def break_lines(x, y):
    x_broken = [x[0]]
    y_broken = [y[0]]
    for i in range(1, len(x)):
        if x[i] < x[i - 1]:
            x_broken.append(np.nan)
            y_broken.append(np.nan)
        x_broken.append(x[i])
        y_broken.append(y[i])
    return np.array(x_broken), np.array(y_broken)


# Plot Avg Validation Loss vs Epoch
plt.figure(figsize=(12, 8))

# Define a color palette for different models
colors = plt.cm.Set1(range(len(data)))

for color, (model_name, folds_data) in zip(colors, data.items()):
    if average_data:
        # Interpolate and average validation and training loss across all folds
        all_folds_data = [{'epoch': epochs, 'val_loss': val_loss, 'train_loss': train_loss_filtered,
                           'epochs_filtered': epochs_filtered}
                          for epochs, val_loss, train_loss_filtered, epochs_filtered, _, _ in folds_data]
        avg_epochs, avg_val_loss = interpolate_and_average(all_folds_data, 'epoch', 'val_loss')
        avg_epochs_filtered, avg_train_loss = interpolate_and_average(all_folds_data, 'epochs_filtered', 'train_loss')
        plt.plot(avg_epochs, avg_val_loss, label=f'{model_name} Validation Avg', color=color, linestyle='-')
        plt.plot(avg_epochs_filtered, avg_train_loss, label=f'{model_name} Training Avg', color=color, linestyle='--')
    else:
        for epochs, val_loss, train_loss_filtered, epochs_filtered, elapsed_time, elapsed_time_filtered, fold in folds_data:
            line_style = next(itertools.cycle(line_styles))
            epochs_broken, val_loss_broken = break_lines(epochs, val_loss)
            epochs_filtered_broken, train_loss_filtered_broken = break_lines(epochs_filtered, train_loss_filtered)
            plt.plot(epochs_broken, val_loss_broken, label=f'{model_name} Fold {fold} Validation', color=color,
                     linestyle=line_style)
            plt.plot(epochs_filtered_broken, train_loss_filtered_broken, label=f'{model_name} Fold {fold} Training',
                     color=color, linestyle='--')

# Set y-limit for Avg Validation Loss vs Epoch plot
if average_data:
    min_val_loss_epoch = min(avg_val_loss)  # Get the minimum validation loss
    y_max_epoch = min_val_loss_epoch * 1.2
    plt.ylim(bottom=0, top=y_max_epoch)

plt.ylim(bottom=0.0)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss vs Epoch')
plt.legend()
plt.grid(True)
plt.show()

# Plot Avg Validation Loss vs Elapsed Time
plt.figure(figsize=(12, 8))

for color, (model_name, folds_data) in zip(colors, data.items()):
    if average_data:
        # Interpolate and average validation and training loss across all folds
        all_folds_data = [{'elapsed_time': elapsed_time, 'val_loss': val_loss, 'train_loss': train_loss_filtered}
                          for _, val_loss, train_loss_filtered, _, elapsed_time, _ in folds_data]
        avg_elapsed_time, avg_val_loss = interpolate_and_average(all_folds_data, 'elapsed_time', 'val_loss')
        avg_elapsed_time, avg_train_loss = interpolate_and_average(all_folds_data, 'elapsed_time', 'train_loss')
        plt.plot(avg_elapsed_time, avg_val_loss, label=f'{model_name} Validation Avg', color=color, linestyle='-')
        plt.plot(avg_elapsed_time, avg_train_loss, label=f'{model_name} Training Avg', color=color, linestyle='--')
    else:
        for epochs, val_loss, train_loss_filtered, epochs_filtered, elapsed_time, elapsed_time_filtered, fold in folds_data:
            line_style = next(itertools.cycle(line_styles))
            elapsed_time_broken, val_loss_broken = break_lines(elapsed_time, val_loss)
            elapsed_time_filtered_broken, train_loss_filtered_broken = break_lines(elapsed_time_filtered,
                                                                                   train_loss_filtered)
            plt.plot(elapsed_time_broken, val_loss_broken, label=f'{model_name} Fold {fold} Validation', color=color,
                     linestyle=line_style)
            plt.plot(elapsed_time_filtered_broken, train_loss_filtered_broken,
                     label=f'{model_name} Fold {fold} Training', color=color, linestyle='--')

# Set y-limit for Avg Validation Loss vs Elapsed Time plot
if average_data:
    min_val_loss_time = min(avg_val_loss)  # Get the minimum validation loss
    y_max_time = min_val_loss_time * 1.2
    plt.ylim(bottom=0, top=y_max_time)

plt.xlabel('Elapsed Time')
plt.ylabel('Loss')
plt.ylim(bottom=0.0)
plt.title('Training and Validation Loss vs Elapsed Time')
plt.legend()
plt.grid(True)
plt.show()
