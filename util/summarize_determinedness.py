import os
import pandas as pd

# Define the folder containing the CSV files
folder_path = '../data'  # Replace with the actual path to the folder

# Settings
transpose_files = True  # Set to True if you want to transpose the files
concatenate_files = True  # Set to False if you don't want to concatenate matching _train and _test files


# Function to calculate the number of rows, columns, ratio, and determine type
def analyze_csv_file(df, transpose=False):
    # Transpose the DataFrame if needed
    if transpose:
        df = df.transpose()
    
    # Get the number of rows and columns
    num_rows = df.shape[0]
    num_cols = df.shape[1]
    
    # Calculate the ratio of rows to columns
    if num_cols == 0:  # Avoid division by zero
        ratio = None
        determination_type = "N/A"
    else:
        ratio = num_rows / num_cols
        # Determine if it's Overdetermined or UNDERDETERMINED
        if ratio > 1:
            determination_type = "Overdetermined"
        else:
            determination_type = "UNDERDETERMINED"
    
    # Calculate avg_richness as the percentage of non-zero values
    total_values = num_rows * num_cols
    if total_values == 0:
        avg_richness = 0
    else:
        non_zero_values = (df != 0).sum().sum()
        avg_richness = (non_zero_values / total_values) * 100
    
    # Calculate interaction statistics
    # Sampled interactions: RxR where R is the number of non-zero features in each sample
    sampled_interactions = 0
    for _, row in df.iterrows():
        non_zero_in_row = (row != 0).sum()
        sampled_interactions += non_zero_in_row ** 2
    
    # Theoretical interactions: NxN where N is the number of features (columns)
    theoretical_interactions = num_cols ** 2
    
    # Interaction determination: calculate the ratio of sampled interactions to theoretical interactions
    if theoretical_interactions == 0:  # Avoid division by zero if no columns
        interaction_ratio = None
        interaction_determination_text = "N/A"
    else:
        interaction_ratio = sampled_interactions / theoretical_interactions
        if interaction_ratio >= 1:
            interaction_determination_text = "Interactions Overdetermined"
        else:
            interaction_determination_text = "INTERACTIONS UNDERDETERMINED"
    
    return num_rows, num_cols, ratio, determination_type, avg_richness, sampled_interactions, theoretical_interactions, interaction_ratio, interaction_determination_text


# Collect all files into a dictionary by their base name (without _train or _test)
file_dict = {}

for file_name in os.listdir(folder_path):
    if file_name.endswith('.csv'):
        base_name = file_name.replace('_train.csv', '').replace('_test.csv', '')
        if base_name not in file_dict:
            file_dict[base_name] = []
        file_dict[base_name].append(file_name)

# Iterate over the file pairs/groups and process them
for base_name, files in file_dict.items():
    dfs = []
    for file_name in files:
        file_path = os.path.join(folder_path, file_name)
        # Read the CSV file without headers or indexes
        df = pd.read_csv(file_path, header=None)
        
        # Transpose if needed before concatenation
        if transpose_files:
            df = df.transpose()
        
        dfs.append(df)
    
    if concatenate_files and len(dfs) > 1:
        # Concatenate train and test files by appending rows
        combined_df = pd.concat(dfs, ignore_index=True)
        print(f'{base_name}_train and {base_name}_test concatenated:')
        num_rows, num_cols, ratio, determination_type, avg_richness, sampled_interactions, theoretical_interactions, interaction_ratio, interaction_determination_text = analyze_csv_file(
            combined_df)
    else:
        # Process each file separately
        for file_name, df in zip(files, dfs):
            print(f'{file_name}:')
            num_rows, num_cols, ratio, determination_type, avg_richness, sampled_interactions, theoretical_interactions, interaction_ratio, interaction_determination_text = analyze_csv_file(
                df)
    
    # Print the result for the current file or concatenated file in the desired format
    print(f'shape: {num_rows} x {num_cols}')
    print(f'determination: {ratio}')
    print(f'{determination_type}')
    print(f'average richness: {avg_richness:.2f}%')
    print(f'interactions: {sampled_interactions} sampled x {theoretical_interactions} theoretical')
    print(f'interaction determination: {interaction_ratio:.2f}')
    print(f'{interaction_determination_text}')
    print('-' * 40)
