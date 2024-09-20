import os
import pandas as pd

# Define the folder containing the CSV files
folder_path = '../data'  # Replace with the actual path to the folder

# Settings
transpose_files = True  # Set to True if you want to transpose the files
concatenate_files = True  # Set to False if you don't want to concatenate matching _train and _test files


# Function to count rows with unique vs. duplicated zero patterns
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
    
    # Count the number of rows with unique zero patterns and duplicated zero patterns
    unique_count = (zero_pattern_counts == 1).sum()  # Rows with a pattern that appears only once
    duplicated_count = (zero_pattern_counts > 1).sum()  # Rows with a pattern that appears more than once
    
    return unique_count, duplicated_count


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
        unique_count, duplicated_count = analyze_zero_pattern_counts(combined_df)
    else:
        # Process each file separately
        for file_name, df in zip(files, dfs):
            print(f'{file_name}:')
            unique_count, duplicated_count = analyze_zero_pattern_counts(df)
    
    # Print the result for the current file or concatenated file in the desired format
    print(f'Rows with unique zero patterns: {unique_count}')
    print(f'Rows with duplicated zero patterns: {duplicated_count}')
    print('-' * 40)
