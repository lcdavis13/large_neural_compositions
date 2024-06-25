# TODO: This file might be expecting a transposed format and/or different expectations for header row and key column


import pandas as pd
import numpy as np

input = 'data/waimea.csv'
output = 'data/waimea_condensed_earlyStop.csv'
remove_percentage = 0.75


def remove_rows_and_columns(csv_file, output_file, remove_percentage_min=0.5, remove_percentage_max=0.9, early_stop_trials=10):
    # Load the CSV file into a DataFrame
    print("Loading CSV file...")
    df = pd.read_csv(csv_file)
    print(f"CSV file loaded. Total rows: {df.shape[0]}, Total columns: {df.shape[1]}")
    
    # Determine the number of rows to remove
    total_rows = df.shape[0]
    rows_to_remove = int(total_rows * remove_percentage_max)
    rows_to_remove_min = int(total_rows * remove_percentage_min)
    print(f"Number of rows to remove: {rows_to_remove}")
    
    # Convert the DataFrame to a numpy array for better performance with large data
    data = df.to_numpy()
    
    # List to track rows that will be removed
    rows_to_remove_indices = []
    
    # Set of columns to be kept initially (all columns)
    columns_to_keep = set(range(data.shape[1]))
    
    # Track the column-to-row ratio and the best row to remove
    ratio_history = []
    stop_flag = False
    
    # Iterate to remove the required number of rows
    for removal_count in range(rows_to_remove):
        if stop_flag:
            break
        
        print(f"Removing row {removal_count + 1}/{rows_to_remove}...")
        
        # Find the row whose removal will result in the maximum number of zero columns remaining
        best_row = None
        max_columns_after_removal = -1
        
        for i in range(data.shape[0]):
            if i in rows_to_remove_indices:
                continue
            
            # Simulate removal of the current row
            temp_columns_to_keep = set(columns_to_keep)
            temp_columns_to_keep -= set(np.where(data[i] != 0)[0])
            
            # Count remaining zero columns
            remaining_columns = len(temp_columns_to_keep)
            
            if remaining_columns > max_columns_after_removal:
                max_columns_after_removal = remaining_columns
                best_row = i
        
        # Remove the best row found
        rows_to_remove_indices.append(best_row)
        columns_to_keep -= set(np.where(data[best_row] != 0)[0])
        
        # Calculate the column-to-row ratio
        remaining_rows = total_rows - len(rows_to_remove_indices)
        current_ratio = len(columns_to_keep) / remaining_rows
        ratio_history.append(current_ratio)
        
        print(f"Row {best_row} removed. Columns remaining: {len(columns_to_keep)}, Current ratio: {current_ratio:.4f}")
        
        # Check the stop condition based on ratio history
        if len(rows_to_remove_indices) >= rows_to_remove_min:
            if len(ratio_history) > early_stop_trials:
                recent_ratios = ratio_history[-early_stop_trials:]
                if recent_ratios[-1] < recent_ratios[0]:
                    stop_flag = True
                    print("Early stopping condition met. Stopping row deletions.")
    
    # Remove the identified rows and columns
    print("Removing identified rows and columns...")
    final_data = np.delete(data, rows_to_remove_indices, axis=0)
    final_data = final_data[:, list(columns_to_keep)]
    print("Rows and columns removed.")
    
    # Convert the result back to a DataFrame
    final_df = pd.DataFrame(final_data, columns=df.columns[list(columns_to_keep)])
    print("DataFrame reconstructed.")
    
    # Save the DataFrame to the output CSV file
    final_df.to_csv(output_file, index=False)
    print(f"Modified data saved to {output_file}")


# Example usage
remove_rows_and_columns(input, output, remove_percentage)
