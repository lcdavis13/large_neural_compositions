import os
import pandas as pd

# Define the folder containing the CSV files
folder_path = '../data'  # Replace with the actual path to the folder

# Settings
transpose_files = True  # Set to True if you want to transpose the files
concatenate_files = True  # Set to False if you don't want to concatenate matching _train and _test files


# Function to count unique zero patterns and unambiguous rows
def analyze_zero_pattern_counts(df, transpose=False):
    # Transpose the DataFrame if needed
    if transpose:
        df = df.transpose()
    
    # Create a boolean DataFrame where True indicates a zero value
    zero_pattern_df = (df == 0)
    
    # Convert each row of boolean values into a tuple (this becomes the "zero pattern")
    zero_patterns = zero_pattern_df.apply(tuple, axis=1)
    
    # Count how often each unique zero pattern occurs
    zero_pattern_counts = zero_patterns.value_counts()
    
    # Calculate the total number of unique zero patterns
    unique_patterns_count = zero_patterns.nunique()  # Get number of unique zero patterns
    
    # Calculate the number of unambiguous rows (patterns that appear exactly once)
    unambiguous_count = (zero_pattern_counts == 1).sum()
    
    # Total number of rows
    total_rows = len(zero_patterns)
    
    return unique_patterns_count, unambiguous_count, total_rows


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
        unique_patterns_count, unambiguous_count, total_rows = analyze_zero_pattern_counts(combined_df)
    else:
        # Process each file separately
        for file_name, df in zip(files, dfs):
            print(f'{file_name}:')
            unique_patterns_count, unambiguous_count, total_rows = analyze_zero_pattern_counts(df)
    
    # Output the results in the desired format
    print(f'Unique zero patterns: {unique_patterns_count} / {total_rows}')
    print(f'Unambiguous samples: {unambiguous_count} / {total_rows}')
    print('-' * 40)
