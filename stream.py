import csv
import os
import pandas as pd


def stream_results(filename, print_console, save_immediately, save_at_all, *args, prefix="", suffix=""):
    if save_at_all:
        stream(False, filename, print_console, *args, prefix=prefix, suffix=suffix, save_immediately=save_immediately)
    
    
def stream_scores(filename, print_console, save_immediately, save_at_all, *args, prefix="", suffix=""):
    if save_at_all:
        stream(True, filename, print_console, *args, prefix=prefix, suffix=suffix, save_immediately=save_immediately)


def stream(keep_old, filename, print_console, *args, prefix="", suffix="", save_immediately=True):
    if len(args) % 2 != 0:
        raise ValueError("Arguments should be in pairs of names and values.")
    
    names = args[0::2]
    values = args[1::2]
    
    if print_console:
        print(prefix + (", ".join([f"{name}: {value}" for name, value in zip(names, values)])) + suffix)
        
    if hasattr(stream, 'queued_values') and stream.queued_values and stream.filename != filename:
        if stream.keep_old:
            update_or_append(stream.file_started, stream.filename, stream.names, stream.queued_values)
        else:
            start_or_append(stream.file_started, stream.filename, stream.names, stream.queued_values)
        stream.queued_values = []
    
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
        if save_immediately:
            if keep_old:
                update_or_append(file_started, filename, names, [values])
            else:
                start_or_append(file_started, filename, names, [values])
        else:
            # push the planned call to a buffer
            if (not hasattr(stream, 'queued_values')) or not stream.queued_values:
                stream.filename = filename
                stream.keep_old = keep_old
                stream.file_started = file_started
                stream.names = names
                stream.queued_values = [values]
            else:
                stream.queued_values.append(values)

def start_or_append(file_started, filename, names, values_batch):
    """
    Writes a batch of values to a CSV file. If the file is not started, it writes headers first.

    Args:
        file_started (bool): Whether the file has been started.
        filename (str): The name of the file.
        names (list): The header names for the CSV file.
        values_batch (list of lists): Batch of rows to write.
    """
    mode = 'a' if file_started else 'w'
    with open(filename, mode, newline='') as csvfile:
        writer = csv.writer(csvfile)

        # If the file is new, write the header row
        if not file_started:
            writer.writerow(names)

        # Write each row in the batch
        for values in values_batch:
            writer.writerow(values)

def update_or_append(file_started, filename, names, values_batch):
    """
    Updates or appends a batch of values to a CSV file. Ensures column alignment and handles new columns.

    Args:
        file_started (bool): Whether the file has been started.
        filename (str): The name of the file.
        names (list): The header names for the CSV file.
        values_batch (list of lists): Batch of rows to update/append.
    """
    # Standardize column names to lower case for case-insensitive comparison
    names_lower = [name.lower() for name in names]

    # If the file has not started or does not exist, initialize it
    if not file_started:
        if not os.path.exists(filename):
            # Create a DataFrame with the given names and values
            df = pd.DataFrame(values_batch, columns=names)
            # Write the DataFrame to the CSV file
            df.to_csv(filename, index=False)
        else:
            # Read the existing file into a DataFrame
            existing_df = pd.read_csv(filename)
            existing_columns_lower = [col.lower() for col in existing_df.columns]

            # Create a DataFrame with the new rows of values
            new_df = pd.DataFrame(values_batch, columns=names)

            # Identify new columns to add
            for name, name_lower in zip(names, names_lower):
                if name_lower not in existing_columns_lower:
                    existing_df[name] = None

            # Append the new rows to the existing DataFrame
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)

            # Reorder the combined DataFrame columns to match the new names order (case insensitive)
            combined_columns_order = list(names + tuple([col for col in combined_df.columns if col.lower() not in names_lower]))
            combined_df = combined_df[combined_columns_order]

            # Write the updated DataFrame back to the CSV file
            combined_df.to_csv(filename, index=False)
    else:
        # Append the batch of values directly
        mode = 'a'
        with open(filename, mode, newline='') as csvfile:
            writer = csv.writer(csvfile)

            # Write each row in the batch
            for values in values_batch:
                writer.writerow(values)