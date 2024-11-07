import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_and_align_data(file_path):
    """
    Load a single CSV file and align data based on epoch numbers.
    The CSV file should have columns: "fold", "epoch", "Avg Training Loss", "Avg Validation Loss".
    """
    # Read the CSV file
    data = pd.read_csv(file_path)

    # Initialize lists to hold training and validation data
    train_data = []
    val_data = []

    # Get unique folds
    folds = data['fold'].unique()

    # Track the longest file length
    max_length = 0

    # Process each fold separately
    for fold in folds:
        fold_data = data[data['fold'] == fold]
        train_loss = fold_data['Avg Training Loss'].reset_index(drop=True)
        val_loss = fold_data['Avg Validation Loss'].reset_index(drop=True)

        max_length = max(max_length, len(train_loss), len(val_loss))

        train_data.append(train_loss)
        val_data.append(val_loss)

    # Align data by padding shorter sequences with their last value
    padded_train_data = [
        data.reindex(range(max_length), method='ffill') for data in train_data
    ]
    padded_val_data = [
        data.reindex(range(max_length), method='ffill') for data in val_data
    ]

    # Combine all data into single DataFrames (columns = different folds)
    aligned_train_data = pd.DataFrame(padded_train_data).T
    aligned_val_data = pd.DataFrame(padded_val_data).T

    return aligned_train_data, aligned_val_data


def plot_means_and_medians(datasets, names, zoom_window=0.0, styles=None):
    """
    Plot mean and median curves for multiple datasets, print their final values,
    and generate two plots: normal and zoomed-in along the y-axis.

    Args:
    - datasets: List of DataFrames containing aligned data sequences.
    - names: List of names corresponding to each dataset.
    - zoom_window: Float value to determine the zoom level for the second plot.
    - styles: List of tuples to specify line styles for mean and median curves.
    """
    if styles is None:
        styles = [('-', '-.')] * len(datasets)  # Default styles

    final_values = []  # Collect final values to determine zoomed plot range

    # Create the first figure (normal plot)
    plt.figure(figsize=(12, 6))

    # Loop through each dataset and plot mean and median curves
    for data, name, style in zip(datasets, names, styles):
        mean_curve = data.mean(axis=1)
        median_curve = data.median(axis=1)

        # Plot both curves with specified styles
        plt.plot(mean_curve, style[0], label=f'{name} (Mean)')
        plt.plot(median_curve, style[1], label=f'{name} (Median)')

        # Print final values and store them
        final_mean = mean_curve.iloc[-1]
        final_median = median_curve.iloc[-1]
        final_values.extend([final_mean, final_median])

        print(f"Final Mean Value for {name}: {final_mean}")
        print(f"Final Median Value for {name}: {final_median}")

    # Configure the first plot (normal view)
    plt.xlabel('Epoch')
    plt.ylabel('Bray-Curtis Dissimilarity')
    plt.title('cNODE.jl-Ocean Loss Curves')
    plt.legend()

    if zoom_window >= 0.0:
        plt.show(block=False)

        # Create the second figure (zoomed-in plot)
        max_final_value = np.mean(final_values)  # Find the maximum final value
        y_max = zoom_window * max_final_value  # Set y-axis upper limit

        plt.figure(figsize=(12, 6))

        # Re-plot all curves for the zoomed-in view
        for data, name, style in zip(datasets, names, styles):
            mean_curve = data.mean(axis=1)
            median_curve = data.median(axis=1)

            plt.plot(mean_curve, style[0], label=f'{name} (Mean)')
            plt.plot(median_curve, style[1], label=f'{name} (Median)')

        # Configure the second plot (zoomed-in view)
        plt.ylim(0, y_max)
        plt.xlabel('Epoch')
        plt.ylabel('Bray-Curtis Dissimilarity')
        plt.title('cNODE.jl-Ocean Loss Curves (zoomed)')
        plt.legend()

    plt.show()


def main(file_path):
    # Load and align datasets
    train_data, val_data = load_and_align_data(file_path)

    # Call the plotting function with datasets and their names
    datasets = [train_data, val_data]
    names = ['Train', 'Validation']

    plot_means_and_medians(datasets, names, zoom_window=2.0)


if __name__ == "__main__":
    main('../results/logs/cNODE1_cNODE-paper-ocean_epochs.csv')