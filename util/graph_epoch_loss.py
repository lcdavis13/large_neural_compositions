import os
import glob
import pandas as pd
# import matplotlib
# matplotlib.use("TkAgg")  # Configure the backend; do this before importing pyplot.
import matplotlib.pyplot as plt
import itertools

# Define the folder containing the CSV files
folder_path = '../results'

# Get all CSV files in the folder that end with "_epochs.csv"
csv_files = glob.glob(os.path.join(folder_path, '*_waimea-condensed_epochs.csv'))

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
    
    # Group by fold and extract epoch, Avg Validation Loss, and Elapsed Time
    for fold, group in df.groupby('fold'):
        epochs = group['epoch']
        val_loss = group['Avg Validation Loss']
        elapsed_time = group['Elapsed Time']
        data[model_name].append((epochs, val_loss, elapsed_time, fold))

# Plot Avg Validation Loss vs Epoch
plt.figure(figsize=(12, 8))

# Define a color palette for different models
colors = plt.cm.Set1(range(len(data)))

for color, (model_name, folds_data) in zip(colors, data.items()):
    for epochs, val_loss, elapsed_time, fold in folds_data:
        line_style = next(itertools.cycle(line_styles))
        plt.plot(epochs, val_loss, label=f'{model_name} Fold {fold}', color=color, linestyle=line_style)

plt.xlabel('Epoch')
plt.ylabel('Avg Validation Loss')
plt.title('Avg Validation Loss vs Epoch')
plt.legend()
plt.ylim(bottom=0)
plt.grid(True)
plt.show()

# Plot Avg Validation Loss vs Elapsed Time
plt.figure(figsize=(12, 8))

for color, (model_name, folds_data) in zip(colors, data.items()):
    for epochs, val_loss, elapsed_time, fold in folds_data:
        line_style = next(itertools.cycle(line_styles))
        plt.plot(elapsed_time, val_loss, label=f'{model_name} Fold {fold}', color=color, linestyle=line_style)

plt.xlabel('Elapsed Time')
plt.ylabel('Avg Validation Loss')
plt.title('Avg Validation Loss vs Elapsed Time')
plt.legend()
plt.ylim(bottom=0)
plt.grid(True)
plt.show()
