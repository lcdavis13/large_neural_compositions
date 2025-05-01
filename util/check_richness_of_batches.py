import os
import random
import re
import numpy as np
import pandas as pd
from glob import glob

# Parameters
CHUNK_PATH_PREFIX = "data/256/256_x0-sparse"  # Full path and prefix up to chunk number
BATCH_SIZE = 250
BATCHES_PER_CHUNK = 20

def estimate_r_batch_max(
    chunk_path_prefix, batch_size=32, batches_per_chunk=100
):
    all_r_batch_max = []
    chunk_files = sorted(glob(f"{chunk_path_prefix}_*.csv"))

    if chunk_files:
        sample_df = pd.read_csv(chunk_files[0], nrows=1)
        print(f"Number of columns (R_max): {sample_df.shape[1]}")
    else:
        raise ValueError("No chunk files found matching the provided path prefix.")

    for chunk_file in chunk_files:
        print(f"Processing chunk: {os.path.basename(chunk_file)}")
        df = pd.read_csv(chunk_file)

        R_i_list = (df != 0).sum(axis=1).values

        for _ in range(batches_per_chunk):
            batch = np.random.choice(R_i_list, size=batch_size, replace=False)
            R_batch_max = batch.max()
            all_r_batch_max.append(R_batch_max)

    mean_r_batch_max = np.mean(all_r_batch_max)
    std_r_batch_max = np.std(all_r_batch_max)

    return mean_r_batch_max, std_r_batch_max


# Example usage:
if __name__ == "__main__":
    mean_estimate, std_estimate = estimate_r_batch_max(
        CHUNK_PATH_PREFIX,
        batch_size=BATCH_SIZE,
        batches_per_chunk=BATCHES_PER_CHUNK
    )
    print(f"Estimated Mean of R_batch_max: {mean_estimate:.2f}")
    print(f"Estimated Stddev of R_batch_max: {std_estimate:.2f}")

