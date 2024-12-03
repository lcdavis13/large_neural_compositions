import os
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
import re
import matplotlib.cm as cm

def load_csv_files(folder_path):
    """
    Load all CSV files from the specified folder, extract 'model' from the filename,
    and include both filename and model as columns.
    """
    all_files = glob(os.path.join(folder_path, "*.csv"))
    dataframes = []
    for file in all_files:
        # Extract the model name from the filename
        filename = os.path.basename(file)
        model_match = re.match(r'(?P<model>[^_]+)_.*\.csv', filename)
        model = model_match.group('model') if model_match else 'unknown'
        
        if model.startswith("c"): # or model.startswith("baseline-constShaped")
        # if model.startswith("baseline") or model == "cNODE1":
            # Load the CSV and add columns
            df = pd.read_csv(file)
            df['source_file'] = filename
            df['model'] = model
            dataframes.append(df)
    return pd.concat(dataframes, ignore_index=True)


def filter_function(single_run_df):
    """
    Filter a single run based on whether the minimum loss is below 0.3.
    Returns True if the run passes the filter, False otherwise.
    """
    return single_run_df['Avg Validation Loss'].min() < 0.6

def early_stopping_function(single_run_df):
    """
    Determine the early stopping epoch for a single run based on the epoch
    with the minimum validation loss. Returns the epoch number.
    """
    # return single_run_df.loc[single_run_df['Avg Validation Loss'].idxmin(), 'epoch']
    return single_run_df['epoch'].max()

def process_data(df, filter_fn=None, early_stopping_fn=None):
    """
    Process the data: apply filtering and early stopping.
    Both functions operate on a single run at a time.
    """
    if filter_fn is None:
        filter_fn = filter_fn
    if early_stopping_fn is None:
        early_stopping_fn = early_stopping_fn

    processed_dfs = []
    for (model, fold, source_file), group in df.groupby(['model', 'fold', 'source_file']):
        # Apply filter function to the single run
        if filter_fn(group):
            # Apply early stopping to the single run
            early_stop_epoch = early_stopping_fn(group)
            # Adjust the losses after the early stopping epoch
            group.loc[group['epoch'] > early_stop_epoch, 'Avg Validation Loss'] = \
                group.loc[group['epoch'] == early_stop_epoch, 'Avg Validation Loss'].values[0]
            processed_dfs.append(group)

    return pd.concat(processed_dfs, ignore_index=True)

def plot_combined_training_curves(df, ylim=1.0, show_training_loss=False, show_median=True):
    """
    Plot mean and optionally median training curves for all models with matching colors.
    Optionally include Training Loss if `show_training_loss` is True.
    """
    unique_models = df['model'].unique()
    color_map = cm.get_cmap('tab20', len(unique_models))  # Use a colormap with distinct colors
    model_colors = {model: color_map(i) for i, model in enumerate(unique_models)}

    plt.figure(figsize=(20, 12))

    for model, group in df.groupby('model'):
        # Calculate statistics for Validation Loss
        val_stats = group.groupby('epoch')['Avg Validation Loss'].agg(['mean', 'median'])

        color = model_colors[model]
        plt.plot(val_stats.index, val_stats['mean'], color=color, label=f'{model} Validation Loss Mean')

        if show_median:
            plt.plot(val_stats.index, val_stats['median'], linestyle='--', color=color, label=f'{model} Validation Loss Median')

        # Optionally add Training Loss
        if show_training_loss and 'Avg Training Loss' in group.columns:
            train_stats = group.groupby('epoch')['Avg Training Loss'].agg(['mean', 'median'])
            plt.plot(train_stats.index, train_stats['mean'], color=color, linestyle='-.', label=f'{model} Training Loss Mean')

            if show_median:
                plt.plot(train_stats.index, train_stats['median'], linestyle=':', color=color, label=f'{model} Training Loss Median')

    plt.title('Training Curves (Mean and Median) by Model')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.ylim(0, ylim)
    plt.show()


# Example usage
# folder_path = "../results/from_koa_11-26_customLR/epochs_merged"
folder_path = "../results/from_koa_11-26_customLR/epochs"
raw_data = load_csv_files(folder_path)
processed_data = process_data(raw_data, filter_function, early_stopping_function)
plot_combined_training_curves(processed_data, ylim=1.0, show_training_loss=True, show_median=True)
plot_combined_training_curves(processed_data, ylim=0.2, show_training_loss=True, show_median=True)
