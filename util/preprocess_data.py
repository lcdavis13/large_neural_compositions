# TODO: This file might be expecting a transposed format and/or different expectations for header row and key column


from collections import defaultdict

import numpy as np
import pandas as pd
import csv

# Load the CSV file
input_filename = 'data/dki.csv'
df = pd.read_csv(input_filename, header=None)

# Create waimea_out DataFrame
waimea_out = df.copy().transpose()
out_filename = input_filename.replace('.csv', '_out.csv')
waimea_out.to_csv(out_filename, quoting=csv.QUOTE_NONNUMERIC)

#waimea_out[data_columns] = waimea_out[data_columns].apply(lambda row: row / row.sum() if row.sum() != 0 else row, axis=1)

# Create waimea_in DataFrame
waimea_in = waimea_out.copy()
waimea_in = waimea_in.apply(lambda row: (row != 0).astype(int) / row[row != 0].count() if row[row != 0].count() != 0 else row, axis=1)

print(waimea_out.head())
print(waimea_in.head())

# Define the new file names based on the input file
in_filename = input_filename.replace('.csv', '_in.csv')

# Save to new CSV files
waimea_in.to_csv(in_filename)

print(f"CSV files '{in_filename}' and '{out_filename}' have been created successfully.")
