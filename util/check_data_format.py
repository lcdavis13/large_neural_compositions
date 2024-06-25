# TODO: This file might be expecting a transposed format and/or different expectations for header row and key column


import pandas as pd


def check_csv_file(file_path, check_invariant=False):
    print(f"Checking file: {file_path}")
    try:
        # Load the CSV file with header and index
        df = pd.read_csv(file_path, header=0, index_col=0)
        
        # Check if all row sums are close to 1
        if not df.sum(axis=1).apply(lambda x: round(x, 10)).eq(1).all():
            error = (df.sum(axis=1) - 1).max()
            print("Not all rows sum to 1. Max error: " + str(error))
            return False
        else:
            print("Row sum check passed.")
        
        if check_invariant:
            # Additional checks for waimea_in.csv
            for _, row in df.iterrows():
                non_zero_values = row[row != 0.0]
                if not non_zero_values.empty and not (non_zero_values == non_zero_values.iloc[0]).all():
                    print("FAILED: Not all non-zero values in the row are identical.")
                    return False
            print("Invariant 1/N check passed.")
        
        return True
    
    except Exception as e:
        print(f"Failed to process {file_path}: {str(e)}")
        return False


# Check waimea_out.csv
result_out = check_csv_file('data/Ptrain_out.csv')

# Check waimea_in.csv with the additional invariant
result_in = check_csv_file('data/Ptrain_in.csv', check_invariant=True)

print("\nResults:")
print(f"'out' file passed all checks: {result_out}")
print(f"'in' file passed all checks: {result_in}")
