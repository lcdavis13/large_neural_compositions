import os
import pandas as pd
from collections import defaultdict

# Set your input and output folders
input_folder = "results/datascale_4-31/expt"
output_folder = "results/datascale_4-31"

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Dictionary to group dataframes by batchid
batchid_groups = defaultdict(list)

# Process all CSV files in the folder
for filename in os.listdir(input_folder):
    if filename.endswith(".csv"):
        filepath = os.path.join(input_folder, filename)
        df = pd.read_csv(filepath)

        # Assuming 'batchid' is a column in the CSV
        batchid = df.loc[0, 'batchid']  # since each file has one row
        batchid_groups[batchid].append(df)

# Write grouped DataFrames to output files
for batchid, dfs in batchid_groups.items():
    combined_df = pd.concat(dfs, ignore_index=True)
    output_path = os.path.join(output_folder, f"batch_{batchid}.csv")
    combined_df.to_csv(output_path, index=False)
