import os
import glob
import pandas as pd


def load_and_process_csv_files(path_pattern):
    """
    Load all CSV files from the given path pattern, process each file by removing rows with 'fold' = -1,
    grouping rows by the 'model' column to ensure rows with different 'model' values are not averaged together,
    averaging numeric columns within each group, and taking the first value for non-numeric columns.
    Additionally, compute summary statistics (median, standard deviation, min, and max) for the 'val_loss' column.

    Args:
        path_pattern (str): The file path pattern to match CSV files (e.g., '../results/hpsearch_cnode1_12-14/cNODE-paper-ocean-std_job*_experiments.csv')

    Returns:
        pd.DataFrame: A master DataFrame concatenating the processed data from all files.
    """
    
    # Find all CSV files that match the path pattern
    csv_files = glob.glob(path_pattern)
    
    if not csv_files:
        print("No CSV files found matching the pattern.")
        return None
    
    all_data = []
    
    for file_path in csv_files:
        try:
            # Load the CSV file
            df = pd.read_csv(file_path)
            
            # Drop rows where 'fold' = -1, if the 'fold' column exists
            if 'fold' in df.columns:
                df = df[df['fold'] != -1]
            
            # Group by 'model' if it exists, otherwise treat it as a single group
            if 'model' in df.columns:
                grouped = df.groupby('model')
            else:
                grouped = [(None, df)]
            
            for model_value, group_df in grouped:
                
                # Separate numeric and non-numeric columns
                numeric_cols = group_df.select_dtypes(include=['number']).columns
                non_numeric_cols = group_df.select_dtypes(exclude=['number']).columns
                
                # Average numeric columns and take the first value of non-numeric columns
                averaged_data = {col: group_df[col].mean() for col in numeric_cols}
                first_value_data = {col: group_df[col].iloc[0] if not group_df[col].empty else None for col in
                                    non_numeric_cols}
                
                # Calculate additional summary statistics for 'val_loss' if it exists
                if 'val_loss' in group_df.columns:
                    val_loss_stats = {
                        'val_loss_median': group_df['val_loss'].median(),
                        'val_loss_std': group_df['val_loss'].std(),
                        'val_loss_min': group_df['val_loss'].min(),
                        'val_loss_max': group_df['val_loss'].max()
                    }
                else:
                    val_loss_stats = {
                        'val_loss_median': None,
                        'val_loss_std': None,
                        'val_loss_min': None,
                        'val_loss_max': None
                    }
                
                # Add the model value to the row if applicable
                if 'model' in df.columns:
                    processed_row = {'model': model_value}
                else:
                    processed_row = {}
                
                # Combine the two dictionaries into one row
                processed_row.update(averaged_data)
                processed_row.update(first_value_data)
                processed_row.update(val_loss_stats)
                
                all_data.append(processed_row)
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
    
    # Create a master DataFrame from the list of processed rows
    master_df = pd.DataFrame(all_data)
    
    return master_df


def save_master_csv(master_df, output_path):
    """
    Save the master DataFrame to a CSV file.

    Args:
        master_df (pd.DataFrame): The master DataFrame to save.
        output_path (str): The output file path for the CSV file.
    """
    try:
        master_df.to_csv(output_path, index=False)
        print(f"Master CSV file saved successfully at {output_path}")
    except Exception as e:
        print(f"Failed to save master CSV file: {e}")


def main():
    folder = "../results/hpsearch_multiple_12-16/constShaped/"
    path_pattern = f"{folder}cNODE-paper-ocean-std_job*_experiments.csv"
    output_path = f"{folder}_experiments.csv"
    
    # Load, process, and concatenate all CSV files into a master DataFrame
    master_df = load_and_process_csv_files(path_pattern)
    
    if master_df is not None:
        # Save the master DataFrame to a CSV file
        save_master_csv(master_df, output_path)


if __name__ == "__main__":
    main()