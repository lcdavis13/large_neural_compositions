import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
    
    # Flatten the matrix and filter out the diagonal (self-interactions)
    flattened_matrix = interaction_matrix.flatten()
    non_zero_values = flattened_matrix[flattened_matrix != 0]
    
    # Compute the average determinedness and non-zero average determinedness
    total_interactions = num_cols * num_cols
    average_determinedness = interaction_matrix.sum() / total_interactions if total_interactions > 0 else 0
    average_nonzero_determinedness = np.mean(non_zero_values) if len(non_zero_values) > 0 else 0
    
    # Compute the standard deviations
    std_determinedness = np.std(flattened_matrix)
    std_nonzero_determinedness = np.std(non_zero_values)
    
    # Count zero and non-zero interactions
    zero_interactions_count = np.sum(flattened_matrix == 0)
    non_zero_interactions_count = np.sum(flattened_matrix != 0)
    
    return (interaction_matrix,
            average_determinedness,
            average_nonzero_determinedness,
            std_determinedness,
            std_nonzero_determinedness,
            flattened_matrix,
            zero_interactions_count,
            non_zero_interactions_count)


# Function to plot the distribution of determinedness values
def plot_determinedness_distribution(all_determinedness, base_name, output_folder):
    plt.figure(figsize=(10, 6))
    plt.hist(all_determinedness, bins=50, color='blue', alpha=0.7, label='All Interaction Determinedness')
    plt.title(f'Interaction Determinedness Distribution for {base_name}')
    plt.xlabel('Determinedness')
    plt.ylabel('Frequency')
    plt.legend()
    
    # Save the plot to a file
    output_file_path = os.path.join(output_folder, f'{base_name}_determinedness_distribution.png')
    plt.savefig(output_file_path)
    plt.close()


# Collect all files into a dictionary by their base name (without _train or _test)
file_dict = {}

for file_name in os.listdir(input_folder):
    if file_name.endswith('.csv'):
        base_name = file_name.replace('_train.csv', '').replace('_test.csv', '')
        if base_name not in file_dict:
            file_dict[base_name] = []
        file_dict[base_name].append(file_name)

# Initialize an empty list to store the statistics for each dataset
stats_list = []

for base_name, files in file_dict.items():
    dfs = []
    all_determinedness = []  # To collect all interaction determinedness for distribution plot
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
        interaction_matrix, avg_determinedness, avg_nonzero_determinedness, std_determinedness, std_nonzero_determinedness, flattened_matrix, zero_count, non_zero_count = compute_interaction_determinedness(
            combined_df)
    else:
        # Process each file separately
        for file_name, df in zip(files, dfs):
            print(f'{file_name}:')
            interaction_matrix, avg_determinedness, avg_nonzero_determinedness, std_determinedness, std_nonzero_determinedness, flattened_matrix, zero_count, non_zero_count = compute_interaction_determinedness(
                df)
    
    # Append the flattened matrix values for distribution plotting
    all_determinedness.extend(flattened_matrix)
    
    # Output the interaction matrix to a file
    output_file_path = os.path.join(output_folder, f'{base_name}_interaction_determinedness.csv')
    pd.DataFrame(interaction_matrix).to_csv(output_file_path, header=False, index=False)
    
    # Print the statistics including counts of zero and non-zero interactions
    print(f'Average interaction determinedness: {avg_determinedness}')
    print(f'Standard deviation: {std_determinedness}')
    print(f'Average non-zero interaction determinedness: {avg_nonzero_determinedness}')
    print(f'Non-zero Standard deviation: {std_nonzero_determinedness}')
    print(f'Number of zero interactions: {zero_count}')
    print(f'Number of non-zero interactions: {non_zero_count}')
    print('-' * 40)
    
    # Store the statistics in the list
    stats_list.append({
        'Dataset': base_name,
        'Avg Determinedness': avg_determinedness,
        'Std Determinedness': std_determinedness,
        'Avg Non-Zero Determinedness': avg_nonzero_determinedness,
        'Std Non-Zero Determinedness': std_nonzero_determinedness,
        'Zero Count': zero_count,
        'Non-Zero Count': non_zero_count
    })
    
    # Generate the plot of determinedness distribution
    plot_determinedness_distribution(all_determinedness, base_name, output_folder)

# Convert the stats list to a DataFrame
stats_df = pd.DataFrame(stats_list)

# Save the statistics to a CSV file
stats_output_path = os.path.join(output_folder, 'all_datasets_statistics.csv')
stats_df.to_csv(stats_output_path, index=False)

print(f"Statistics for all datasets saved to {stats_output_path}")

