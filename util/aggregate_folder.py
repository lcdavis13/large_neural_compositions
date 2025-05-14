import os
import glob
import pandas as pd

# === CONFIGURATION ===
input_folder = "./results/datascale_5-13_1k/expt/"  # Update this to your target folder
output_file = "./results/datascale_5-13_1k/expt.csv"  # Output path for the merged CSV

def concatenate_csvs(input_folder, output_file):
    csv_files = glob.glob(os.path.join(input_folder, "*.csv"))
    
    if not csv_files:
        print("No CSV files found in the folder.")
        return

    all_dfs = []
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            df['source_file'] = os.path.basename(file)  # Optional: track source file
            all_dfs.append(df)
        except Exception as e:
            print(f"Failed to read {file}: {e}")

    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        combined_df.to_csv(output_file, index=False)
        print(f"Successfully saved combined CSV to: {output_file}")
    else:
        print("No valid CSV files to concatenate.")

if __name__ == "__main__":
    concatenate_csvs(input_folder, output_file)
