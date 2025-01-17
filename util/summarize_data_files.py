import os
import pandas as pd

# Define the folder containing the CSV files
folder_path = './data'  # Replace with the actual path to the folder

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
    else:
        ratio = num_rows / num_cols
    
    # Calculate avg_richness as the percentage of non-zero values
    total_values = num_rows * num_cols
    if total_values == 0:
        avg_richness = 0
        std1_rich = 0
        std2_rich = 0
    else:
        non_zero_values = (df != 0).sum().sum()
        avg_richness = (non_zero_values / total_values) * 100
        # std dev of richness along axis 0
        std1_rich = (df != 0).sum(axis=0).std()
        # std dev of richness along axis 1
        std2_rich = (df != 0).sum(axis=1).std()
    
    return num_rows, num_cols, ratio, avg_richness, std1_rich, std2_rich


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
        num_rows, num_cols, ratio, avg_richness, std1_rich, std2_rich = analyze_csv_file(combined_df)
    else:
        # Process each file separately
        for file_name, df in zip(files, dfs):
            print(f'{file_name}:')
            num_rows, num_cols, ratio, avg_richness, std1_rich, std2_rich = analyze_csv_file(df)
    
    # Print the result for the current file or concatenated file in the desired format
    print(f'shape: {num_rows} x {num_cols}')
    print(f'determination ratio: {ratio}')
    print(f'average richness: {avg_richness:.2f}%')
    print(f'std deviation of feature rate: {std1_rich:.2f}')
    print(f'std deviation of sample richness: {std2_rich:.2f}')
    print('-' * 40)
