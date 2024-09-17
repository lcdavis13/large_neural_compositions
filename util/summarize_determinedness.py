import os
import pandas as pd
import numpy as np

# Define the folder containing the CSV files and the folder for output
input_folder = '../data'  # Replace with the actual path to the folder
output_folder = '../analysis/interaction_determinedness'

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Settings
transpose_files = True  # Set to True if you want to transpose the files
concatenate_files = True  # Set to False if you don't want to concatenate matching _train and _test files

# Function to calculate the determinedness matrix
def compute_interaction_determinedness(df, transpose=False):
    # Transpose the DataFrame if needed
    if transpose:
        df = df.transpose()

    # Get the number of rows (samples) and columns (features)
    num_rows = df.shape[0]
    num_cols = df.shape[1]

    # Initialize the NxN matrix for storing interaction determinedness
    interaction_matrix = np.zeros((num_cols, num_cols))

    # Iterate through each sample (row)
    for _, row in df.iterrows():
        # Find the indices of non-zero features in the current sample (row)
        non_zero_indices = np.where(row != 0)[0]
        non_zero_features = len(non_zero_indices)

        # Skip if there are no non-zero features
        if non_zero_features == 0:
            continue

        # Only compute the contribution for pairs of non-zero features
        for j in non_zero_indices:
            for k in non_zero_indices:
                # Contribution is 1 if both j and k are non-zero, normalized by R_m^2
                contribution = 1 / (non_zero_features ** 2)
                interaction_matrix[j, k] += contribution

    # Compute the average determinedness
    total_interactions = num_cols * num_cols
    average_determinedness = interaction_matrix.sum() / total_interactions if total_interactions > 0 else 0

    return interaction_matrix, average_determinedness


# Collect all files into a dictionary by their base name (without _train or _test)
file_dict = {}

for file_name in os.listdir(input_folder):
    if file_name.endswith('.csv'):
        base_name = file_name.replace('_train.csv', '').replace('_test.csv', '')
        if base_name not in file_dict:
            file_dict[base_name] = []
        file_dict[base_name].append(file_name)

# Iterate over the file pairs/groups and process them
for base_name, files in file_dict.items():
    dfs = []
    for file_name in files:
        file_path = os.path.join(input_folder, file_name)
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
        interaction_matrix, avg_determinedness = compute_interaction_determinedness(combined_df)
    else:
        # Process each file separately
        for file_name, df in zip(files, dfs):
            print(f'{file_name}:')
            interaction_matrix, avg_determinedness = compute_interaction_determinedness(df)

    # Output the interaction matrix to a file
    output_file_path = os.path.join(output_folder, f'{base_name}_interaction_determinedness.csv')
    pd.DataFrame(interaction_matrix).to_csv(output_file_path, header=False, index=False)

    # Print the average interaction determinedness
    print(f'Average interaction determinedness for {base_name}: {avg_determinedness}')
    print('-' * 40)
