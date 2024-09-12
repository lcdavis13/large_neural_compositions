import pandas as pd

import pandas as pd


def check_csv_file(file_path, check_invariant=False, transpose=False, hasHeader=False, hasIndex=False):
    print(f"Checking file: {file_path}")
    try:
        
        # Load the CSV file with header and index
        df = pd.read_csv(file_path, header=0 if hasHeader else None, index_col=0 if hasIndex else None)
        
        # Transpose the DataFrame if transpose is True
        if transpose:
            df = df.transpose()
        
        # Define a small tolerance for floating-point comparison
        tolerance = 1e-9
        
        # Check if all row sums are close to 1 within the tolerance
        row_sums = df.sum(axis=1)
        deviations = abs(row_sums - 1) >= tolerance
        
        if deviations.any():
            max_error = max(abs(row_sums - 1))
            print(f"Not all rows sum to 1. Max error: {max_error}")
            # Report each row key that violates the tolerance
            for row_key in df.index[deviations]:
                deviation_amount = row_sums[row_key] - 1
                print(f"Row '{row_key}' has a sum of {row_sums[row_key]}, deviating by {deviation_amount}.")
            return False
        else:
            print("Row sum check passed.")
        
        if check_invariant:
            # Additional checks for waimea_in.csv
            for idx, row in df.iterrows():
                non_zero_values = row[row != 0.0]
                if not non_zero_values.empty and not (non_zero_values == non_zero_values.iloc[0]).all():
                    print(f"FAILED: Not all non-zero values in the row are identical in row {idx}.")
                    return False
            print("Invariant 1/N check passed.")
        
        return True
    
    except Exception as e:
        print(f"Failed to process {file_path}: {str(e)}")
        return False


# Check waimea_out.csv
result_out = check_csv_file('../data/Ptrain.csv', transpose=False, hasHeader=False, hasIndex=False)

# Check waimea_in.csv with the additional invariant and transposing the data
# result_in = check_csv_file('../data/Ptrain_in.csv', check_invariant=True, transpose=True, hasHeader=False, hasIndex=False)

print("\nResults:")
print(f"'out' file passed all checks: {result_out}")
# print(f"'in' file passed all checks: {result_in}")
