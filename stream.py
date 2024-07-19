import csv
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")


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


def plot(title, label, epoch, validation_loss, train_loss, add_point=False):
    # Initialize the set of plots if it doesn't exist
    if not hasattr(plot, 'figs'):
        plot.figs = {}
    
    # Check if it's the first time the function is called for this plot name during this execution
    if title not in plot.figs.keys():
        fig, ax = plt.subplots()
        plot.figs[title] = (fig, ax)
    else:
        fig, ax = plot.figs[title]
    
    # Find existing lines with the given label or create new ones
    lines = {line.get_label(): line for line in ax.get_lines()}
    val_label = f'{label} - Val Loss'
    trn_label = f'{label} - Trn Loss'
    
    if val_label in lines:
        val_line = lines[val_label]
        if validation_loss:
            val_line.set_xdata(list(val_line.get_xdata()) + [epoch])
            val_line.set_ydata(list(val_line.get_ydata()) + [validation_loss])
            if add_point:
                ax.plot([epoch], [validation_loss], '*', color=val_line.get_color())
    else:
        # Create new lines if they don't exist
        if validation_loss:
            ax.plot([epoch], [validation_loss], label=val_label)
            if add_point:
                ax.plot([epoch], [validation_loss], '*', color=ax.lines[-1].get_color())
    
    if trn_label in lines:
        train_line = lines[trn_label]
        if train_loss:
            train_line.set_xdata(list(train_line.get_xdata()) + [epoch])
            train_line.set_ydata(list(train_line.get_ydata()) + [train_loss])
            if add_point:
                ax.plot([epoch], [train_loss], '*', color=train_line.get_color())
    else:
        if train_loss:
            ax.plot([epoch], [train_loss], label=trn_label)
            if add_point:
                ax.plot([epoch], [train_loss], '*', color=ax.lines[-1].get_color())
    
    # Redraw the plot with the updated data
    ax.relim()
    ax.autoscale_view()
    ax.set_title(title)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    
    plt.draw()
    plt.pause(0.0001)


if __name__ == "__main__":
    plot("Training Progress", "Model A", 1, 0.5, None)
    plt.pause(0.5)
    plot("other thing", "Model A", 10, 5, None)
    plt.pause(1.0)
    plot("Training Progress", "Model A", 2, 0.4, 2.5, True)
    plt.pause(0.5)
    plot("other thing", "Model A", 15, 5, 6)
    plt.pause(1.0)
    plot("Training Progress", "Model A", 3, 1.4, 0.5)
    plt.pause(0.5)
    plot("other thing", "Model A", 20, 4, 7)
    plt.pause(1.0)
    