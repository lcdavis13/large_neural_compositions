import pandas as pd
import numpy as np

# File paths
filepath = '../data/waimea_condensed.csv'
filepath_train = '../data/waimea_condensed_train.csv'
filepath_test = '../data/waimea_condensed_test.csv'

# Read data with pandas, assuming the first column is the index
data = pd.read_csv(filepath, index_col=0)

# Calculate split index
split_factor = 0.2
num_cols = data.shape[1]  # Number of columns (features)
num_split = int(split_factor * num_cols)  # Number of columns to include in the test set

# Generate random indices for the test columns
random_indices = np.random.choice(num_cols, size=num_split, replace=False)

# Select columns for validation and training
P_train = data.iloc[:, np.setdiff1d(range(num_cols), random_indices)]
P_test = data.iloc[:, random_indices]

# Save to CSV files
P_train.to_csv(filepath_train, index=False, header=False)
P_test.to_csv(filepath_test, index=False, header=False)
