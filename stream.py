import csv
import itertools
import os
import time

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


def stream_results(filename, print_console, *args, prefix="", suffix=""):
    stream(False, filename, print_console, *args, prefix=prefix, suffix=suffix)
    
    
def stream_scores(filename, print_console, *args, prefix="", suffix=""):
    stream(True, filename, print_console, *args, prefix=prefix, suffix=suffix)


def stream(keep_old, filename, print_console, *args, prefix="", suffix=""):
    if len(args) % 2 != 0:
        raise ValueError("Arguments should be in pairs of names and values.")
    
    names = args[0::2]
    values = args[1::2]
    
    if print_console:
        print(prefix + (", ".join([f"{name}: {value}" for name, value in zip(names, values)])) + suffix)
    
    # Check if file exists
    if filename:
        # Initialize the set of filenames if it doesn't exist
        if not hasattr(stream, 'filenames'):
            stream.filenames = set()
        
        # Check if it's the first time the function is called for this filename during this execution
        if filename not in stream.filenames:
            stream.filenames.add(filename)
            file_started = False
        else:
            file_started = True
        
        if keep_old:
            update_or_append(file_started, filename, names, values)
        else:
            start_or_append(file_started, filename, names, values)


def start_or_append(file_started, filename, names, values):
    mode = 'a' if file_started else 'w'
    with open(filename, mode, newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # If the file is new, write the header row
        if not file_started:
            writer.writerow(names)
        
        # Write the values row
        writer.writerow(values)


def update_or_append(file_started, filename, names, values):
    # Standardize column names to lower case for case-insensitive comparison
    names_lower = [name.lower() for name in names]
    
    # If the file has not started or does not exist, we need to initialize it
    if not file_started:
        if not os.path.exists(filename):
            # Create a DataFrame with the given names and values
            df = pd.DataFrame([values], columns=names)
            # Write the DataFrame to the CSV file
            df.to_csv(filename, index=False)
        
        else:
            # Read the existing file into a DataFrame
            existing_df = pd.read_csv(filename)
            existing_columns_lower = [col.lower() for col in existing_df.columns]
            
            # Create a DataFrame with the new row of values
            new_df = pd.DataFrame([values], columns=names)
            
            # Identify new columns to add
            for name, name_lower in zip(names, names_lower):
                if name_lower not in existing_columns_lower:
                    existing_df[name] = None
            
            # Append the new row to the existing DataFrame
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            
            # Reorder the combined DataFrame columns to match the new names order (case insensitive)
            combined_columns_order = list(names + tuple([col for col in combined_df.columns if col.lower() not in names_lower]))
            combined_df = combined_df[combined_columns_order]
            
            # Write the updated DataFrame back to the CSV file
            combined_df.to_csv(filename, index=False)
    
    else:
        mode = 'a'
        with open(filename, mode, newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write the values row
            writer.writerow(values)


def plot(title, xlabel, ylabel, line_labels, x_value, y_values, add_point=False, x_log=False, y_log=False):
    plt.ion()
    """
    Plots an arbitrary number of curves against epochs.
        x_value (number): x of new point
        labels (list of str): labels of curves
        y_values (list of numbers): list of values corresponding to the labels.
        add_point (bool): Whether to add a point marker to the curves here
    """
    if not hasattr(plot, 'figs'):
        plot.figs = {}
    
    # Handle exited plots
    # (unfortunately, this code is only sometimes reached when multiple plots are opened but only some of them are closed. It seems to freeze on plt.pause() depending on which is closed first.)
    if title in plot.figs and not plot.figs[title]:
        return
    if title in plot.figs and not plt.fignum_exists(plot.figs[title][0].number):
        plot.figs[title] = None
        return
        
    
    if title not in plot.figs:
        fig, ax = plt.subplots()
        plot.figs[title] = (fig, ax)
    
    fig, ax = plot.figs[title]
    
    # Determine the current color index based on the number of existing lines and new lines
    num_existing_lines = len(ax.get_lines())
    num_new_lines = len(line_labels)
    color_index = num_existing_lines // num_new_lines
    
    # Get the color for the current set of lines
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    current_color = color_cycle[color_index % len(color_cycle)]
    
    # Define style cycle for each call
    style_cycle = itertools.cycle(['-', '--', '-.', ':'])
    
    # Map styles to labels to maintain consistency
    if not hasattr(plot, 'label_styles'):
        plot.label_styles = {}
    
    for i, (label, value) in enumerate(zip(line_labels, y_values)):
        if label not in plot.label_styles:
            plot.label_styles[label] = next(itertools.islice(style_cycle, i, None))
    
    lines = {line.get_label(): line for line in ax.get_lines()}
    
    for label, value in zip(line_labels, y_values):
        linestyle = plot.label_styles[label]
        if label in lines:
            line = lines[label]
            if value is not None:
                line.set_xdata(list(line.get_xdata()) + [x_value])
                line.set_ydata(list(line.get_ydata()) + [value])
                if add_point:
                    ax.plot([x_value], [value], '*', color=line.get_color())
        else:
            if value is not None:
                ax.plot([x_value], [value], label=label, color=current_color, linestyle=linestyle)
                if add_point:
                    ax.plot([x_value], [value], '*', color=current_color)
    
    ax.relim()
    ax.autoscale_view()
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    if x_log:
        ax.set_xscale('log')
    if y_log:
        ax.set_yscale('log')
    
    plt.draw()
    plt.pause(0.00000001)


def plot_single(title, xlabel, ylabel, line_label, x_value, y_value, add_point=False, x_log=False, y_log=False):
    """
    Helper method to call the plot function with only one value of y and its label.
    """
    plot(title, xlabel, ylabel, [line_label], x_value, [y_value], add_point=add_point, x_log=x_log, y_log=y_log)


def plot_loss(title, label, epoch, train_loss, validation_loss=None, add_point=False):
    """
    Helper function to plot validation and training loss curves.

    Parameters:
        title (str): Title of the plot.
        label (str): Base label for the curves.
        epoch (int): Current epoch.
        validation_loss (float): Validation loss value.
        train_loss (float): Training loss value.
        add_point (bool): Whether to add a point at the current epoch value.
    """
    labels = [f'{label} - Val Loss', f'{label} - Trn Loss']
    values = [validation_loss, train_loss]
    plot(title, 'Epoch', 'Loss', labels, epoch, values, add_point)
    
def keep_plots_open():
    """
    Keeps all plot windows open until they are closed by the user.
    """
    while plt.get_fignums():
        plt.waitforbuttonpress()


if __name__ == "__main__":
    plot_loss("Training Progress", "Model A", 1, 0.5, None)
    plot("other thing", "xthing", "ything", ["A thing 1", "A thing 2"], 10, [10, 5], None)
    plot_loss("Training Progress", "Model A", 2, 0.4, 2.5, True)
    plot("other thing", "xthing", "ything", ["A thing 1", "A thing 2"], 15, [8, 6], None)
    plt.pause(5.0)
    plot_loss("Training Progress", "Model A", 3, 1.4, 0.5)
    plot("other thing", "xthing", "ything", ["A thing 1", "A thing 2"], 20, [11, 9], None)
    plot_loss("Training Progress", "Model B", 1, 1.5, None)
    plot("other thing", "xthing", "ything", ["B thing 1", "B thing 2"], 10, [11, 6], None)
    plot_loss("Training Progress", "Model B", 2, 1.4, 3.5, True)
    plot("other thing", "xthing", "ything", ["B thing 1", "B thing 2"], 15, [9, 7], None)
    plot_loss("Training Progress", "Model B", 3, 2.4, 1.5)
    plot("other thing", "xthing", "ything", ["B thing 1", "B thing 2"], 20, [12, 10], None)
    # plt.pause(1.0)
    print("done")
    keep_plots_open()
    