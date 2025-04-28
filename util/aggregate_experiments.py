import os
import glob
import pandas as pd
import re

model_col = "model_config"
val_loss_col = "mini_epoch @ val_loss"
test_loss_col = "test loss"

def load_and_process_csv_files(path_pattern):
    """
    Load all CSV files from the given path pattern, process each file by removing rows with 'fold' = -1,
    grouping rows by the model_col column to ensure rows with different model_col values are not averaged together,
    averaging numeric columns within each group, and taking the first value for non-numeric columns.
    Additionally, compute summary statistics (median, standard deviation, min, and max) for the val_loss_col column.
    Extract the job number from the filename and include it as a column.

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
            # Extract job number from the filename using regex
            job_number_match = re.search(r'job(\d+)', os.path.basename(file_path))
            job_number = int(job_number_match.group(1)) if job_number_match else None
            
            # Load the CSV file
            df = pd.read_csv(file_path)
            
            # Drop rows where 'fold' = -1, if the 'fold' column exists
            if 'mean_val_loss' in df.columns:
                df = df[df['mean_val_loss'] >= 0.0]
            
            # Group by model_col if it exists, otherwise treat it as a single group
            if model_col in df.columns:
                grouped = df.groupby(model_col)
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
                
                # Calculate additional summary statistics for val_loss_col if it exists
                if val_loss_col in group_df.columns:
                    val_loss_stats = {
                        'val_loss_median': group_df[val_loss_col].median(),
                        'val_loss_std': group_df[val_loss_col].std(),
                        'val_loss_min': group_df[val_loss_col].min(),
                        'val_loss_max': group_df[val_loss_col].max()
                    }
                else:
                    val_loss_stats = {
                        'val_loss_median': None,
                        'val_loss_std': None,
                        'val_loss_min': None,
                        'val_loss_max': None
                    }
                
                # Calculate additional summary statistics for test_loss_col if it exists
                if test_loss_col in group_df.columns:
                    test_loss_stats = {
                        'test_loss_median': group_df[test_loss_col].median(),
                        'test_loss_std': group_df[test_loss_col].std(),
                        'test_loss_min': group_df[test_loss_col].min(),
                        'test_loss_max': group_df[test_loss_col].max()
                    }
                else:
                    test_loss_stats = {
                        'test_loss_median': None,
                        'test_loss_std': None,
                        'test_loss_min': None,
                        'test_loss_max': None
                    }
                
                # Add the model value and job number to the row if applicable
                processed_row = {'job_number': job_number}
                if model_col in df.columns:
                    processed_row[model_col] = model_value
                
                # Combine the dictionaries into one row
                processed_row.update(averaged_data)
                processed_row.update(first_value_data)
                processed_row.update(val_loss_stats)
                processed_row.update(test_loss_stats)
                
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
        # Define the desired column order, excluding the always-last columns
        column_order = [
            'dataset', 'data_subset', model_col, 'model_name', val_loss_col, test_loss_col, 
            'lr', 'reptile_lr', 'wd', 'identity_gate', 
            'cnode_bias', 'cnode1_init_zero', 
            'attend_dim_per_head', 'num_heads', 'depth', 'dropout', 'ode_timestep_file', 'hidden_dim', 'ffn_dim_multiplier', 
            'noise', 'interpolate', 'interpolate_noise',
            "update steps", "Avg Training Loss", "Elapsed Time", "VRAM (GB)", "Peak RAM (GB)",
            # 'mean_val_loss @ epoch', 'mean_val_loss @ time', 'mean_val_loss @ trn_loss',
            'val_loss_median', 'val_loss_std', 'val_loss_min', 'val_loss_max'
            'test_loss_median', 'test_loss_std', 'test_loss_min', 'test_loss_max'
        ]

        # Define the columns that should always appear last
        always_last_columns = [
            'device', 'solver', 'early_stop', 'whichfold', 'batchid', 'taskid', 'jobid', 'data_configid', 'configid', 'headless'
        ]

        # Reorder columns if they exist in the DataFrame
        existing_columns = [col for col in column_order if col in master_df.columns]
        remaining_columns = [col for col in master_df.columns if col not in existing_columns + always_last_columns]
        existing_last_columns = [col for col in always_last_columns if col in master_df.columns]

        # Create the final ordered list of columns
        ordered_columns = existing_columns + remaining_columns + existing_last_columns
        master_df = master_df[ordered_columns]

        # Sort by 'mean_val_loss' in ascending order if it exists
        if 'mean_val_loss' in master_df.columns:
            master_df = master_df.sort_values(by='mean_val_loss', ascending=True)

        # Save the DataFrame to a CSV file
        master_df.to_csv(output_path, index=False)
        print(f"Master CSV file saved successfully at {output_path}")
    except Exception as e:
        print(f"Failed to save master CSV file: {e}")



def main():
    folder = "./results/hpsearch_4-22_100k/expt/"
    path_pattern = f"{folder}*_job*_experiments.csv"
    output_path = f"{folder}_experiments.csv"
    
    # Load, process, and concatenate all CSV files into a master DataFrame
    master_df = load_and_process_csv_files(path_pattern)
    
    if master_df is not None:
        # Save the master DataFrame to a CSV file
        save_master_csv(master_df, output_path)


if __name__ == "__main__":
    main()
