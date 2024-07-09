import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import numpy as np

# Define the folder containing the CSV files
folder_path = '../results/logs/'

# Get all CSV files in the folder that end with "_epochs.csv"
csv_files = glob.glob(os.path.join(folder_path, 'canODE_*ocean*_epochs.csv'))
csv_files2 = glob.glob(os.path.join(folder_path, 'canODE-noValue_*ocean*_epochs.csv'))
csv_files3 = glob.glob(os.path.join(folder_path, 'canODE-multihead_*ocean*_epochs.csv'))
csv_files4 = glob.glob(os.path.join(folder_path, 'canODE-singlehead_*ocean*_epochs.csv'))
csv_files5 = glob.glob(os.path.join(folder_path, 'canODE-transformer_*ocean*_epochs.csv'))
csv_files6 = glob.glob(os.path.join(folder_path, 'baseline-cNODE0_*ocean*_epochs.csv'))
csv_files7 = glob.glob(os.path.join(folder_path, 'cNODE1_*ocean*_epochs.csv'))
csv_files8 = glob.glob(os.path.join(folder_path, 'cNODE2_*ocean*_epochs.csv'))
csv_files = csv_files + csv_files2 + csv_files3 + csv_files4 + csv_files5 + csv_files6 + csv_files7 + csv_files8


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
    #
    # # Filter for fold=0 and epoch <= 60
    # df = df[(df['fold'] == 1) & (df['epoch'] <= 50)]
    
    # Group by fold and extract epoch, Avg Validation Loss, and Elapsed Time
    for fold, group in df.groupby('fold'):
        epochs = group['epoch']
        val_loss = group['Avg Validation Loss']
        elapsed_time = group['Elapsed Time']
        data[model_name].append((epochs, val_loss, elapsed_time, fold))

# Optional variable to enable averaging of data
average_data = True


# Function to interpolate and average data
def interpolate_and_average(data_list, x_key, y_key):
    all_x = sorted(set(itertools.chain.from_iterable([data[x_key] for data in data_list])))
    all_y_interpolated = []
    
    for data in data_list:
        y_interpolated = np.interp(all_x, data[x_key], data[y_key])
        all_y_interpolated.append(y_interpolated)
    
    avg_y = np.mean(all_y_interpolated, axis=0)
    return all_x, avg_y


# Plot Avg Validation Loss vs Epoch
plt.figure(figsize=(12, 8))

# Define a color palette for different models
colors = plt.cm.Set1(range(len(data)))

for color, (model_name, folds_data) in zip(colors, data.items()):
    if average_data:
        # Interpolate and average validation loss across all folds
        all_folds_data = [{'epoch': epochs, 'val_loss': val_loss} for epochs, val_loss, _, _ in folds_data]
        avg_epochs, avg_val_loss = interpolate_and_average(all_folds_data, 'epoch', 'val_loss')
        plt.plot(avg_epochs, avg_val_loss, label=f'{model_name} Average', color=color, linestyle='-')
    else:
        for epochs, val_loss, elapsed_time, fold in folds_data:
            line_style = next(itertools.cycle(line_styles))
            plt.plot(epochs, val_loss, label=f'{model_name} Fold {fold}', color=color, linestyle=line_style)

# After calculating avg_epochs and avg_val_loss for epochs-based plot
if average_data:
    min_val_loss_epoch = avg_val_loss[0]  # Assuming the first value corresponds to the earliest epoch
    y_max_epoch = min_val_loss_epoch * 1.2
# Add y_max_epoch to the y-limit of the Avg Validation Loss vs Epoch plot
plt.ylim(bottom=0, top=y_max_epoch)

plt.xlabel('Epoch')
plt.ylabel('Avg Validation Loss')
plt.title('Avg Validation Loss vs Epoch')
plt.legend()
plt.grid(True)
plt.show()

# Plot Avg Validation Loss vs Elapsed Time
plt.figure(figsize=(12, 8))

for color, (model_name, folds_data) in zip(colors, data.items()):
    if average_data:
        # Interpolate and average validation loss across all folds
        all_folds_data = [{'elapsed_time': elapsed_time, 'val_loss': val_loss} for _, val_loss, elapsed_time, _ in
                          folds_data]
        avg_elapsed_time, avg_val_loss = interpolate_and_average(all_folds_data, 'elapsed_time', 'val_loss')
        plt.plot(avg_elapsed_time, avg_val_loss, label=f'{model_name} Average', color=color, linestyle='-')
    else:
        for epochs, val_loss, elapsed_time, fold in folds_data:
            line_style = next(itertools.cycle(line_styles))
            plt.plot(elapsed_time, val_loss, label=f'{model_name} Fold {fold}', color=color, linestyle=line_style)

# After calculating avg_elapsed_time and avg_val_loss for elapsed time-based plot
if average_data:
    min_val_loss_time = avg_val_loss[0]  # Assuming the first value corresponds to the earliest time
    y_max_time = min_val_loss_time * 1.2
# Add y_max_time to the y-limit of the Avg Validation Loss vs Elapsed Time plot
plt.ylim(bottom=0, top=y_max_time)

plt.xlabel('Elapsed Time')
plt.ylabel('Avg Validation Loss')
plt.title('Avg Validation Loss vs Elapsed Time')
plt.legend()
plt.grid(True)
plt.show()
