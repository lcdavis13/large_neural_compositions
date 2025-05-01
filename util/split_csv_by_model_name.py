import pandas as pd
import os

def split_csv_by_model(input_file, output_dir):
    # Read the CSV file
    df = pd.read_csv(input_file)

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Group by 'model_name' and save each group to a separate file
    for model_name, group in df.groupby('model_name'):
        # Clean the model name to create a safe filename
        safe_model_name = "".join(c if c.isalnum() or c in (' ', '_', '-') else '_' for c in model_name)
        output_file = os.path.join(output_dir, f"HPResults_{safe_model_name}.csv")
        group.to_csv(output_file, index=False)
        print(f"Saved {output_file}")

if __name__ == "__main__":
    folder = "batch/1kLow/" 
    input_file = f"{folder}HPResults.csv"  # Replace with your input CSV file path
    output_dir = f"{folder}"  # Replace with your desired output directory
    split_csv_by_model(input_file, output_dir)
