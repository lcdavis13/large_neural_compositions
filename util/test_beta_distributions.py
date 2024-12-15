import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
import pandas as pd  # Import pandas to read CSV


# Function to get bounds from CSV
def get_bounds_from_csv(file_path):
    """
    Reads a CSV file and finds the highest and lowest non-zero values.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        tuple: (min_val, max_val) The minimum and maximum non-zero values from the CSV file.
    """
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Flatten the dataframe into a 1D array and drop NaN values
    values = df.values.flatten()
    
    # Filter non-zero and non-NaN values
    non_zero_values = values[(values != 0) & (~np.isnan(values))]
    
    if len(non_zero_values) == 0:
        raise ValueError("No non-zero values found in the CSV file.")
    
    min_val = np.min(non_zero_values)
    max_val = np.max(non_zero_values)
    
    return min_val, max_val


# File path to the CSV (update this path as needed)
csv_file_path = '../data/cNODE-paper-ocean-std_train.csv'

# Get bounds from the CSV file
try:
    min_mean, max_mean = get_bounds_from_csv(csv_file_path)
    # max_mean = 10*min_mean
    print(f"Bounds for means from CSV: min_mean = {min_mean}, max_mean = {max_mean}")
except Exception as e:
    print(f"Error: {e}")
    min_mean, max_mean = 0.025, (1.0-0.025)  # Default range if error occurs
    
# min_mean, max_mean = 0.025, (1.0-0.025)  # for fixed plots

stepnum = 10
stepsize = (max_mean - min_mean) / (stepnum)

# Parameters
mean_values = np.arange(min_mean, max_mean + 0.1*stepsize, stepsize)  # Use bounds from the CSV file
peak_stddev = 0.05

# Prepare plot
plt.figure(figsize=(12, 8))

# Generate and plot Beta distributions
x = np.linspace(0, min(max_mean*2, 1.0), 500)
for x_mean in mean_values:
    stddev = math.sin(x_mean * math.pi) * peak_stddev
    var = stddev ** 2
    k = x_mean * (1.0 - x_mean) / var - 1.0
    alpha = x_mean * k
    beta_param = (1 - x_mean) * k
    pdf = beta.pdf(x, alpha, beta_param)
    plt.plot(x, pdf, label=f'mean = {x_mean}')#, α = {alpha:.1f}, β = {beta_param:.1f}')


# Sample Beta distribution (currently unused, but demonstrates how to sample)
def sample_beta_distribution(mean_values, peak_stddev, num_samples=1000):
    samples = []
    for x_mean in mean_values:
        stddev = math.sin(x_mean * math.pi) * peak_stddev
        var = stddev ** 2
        k = x_mean * (1.0 - x_mean) / var - 1.0
        alpha = x_mean * k
        beta_param = (1 - x_mean) * k
        
        # Sample points
        samples.append(np.random.beta(alpha, beta_param, num_samples))
    return samples


# Plot settings
plt.title("Various Beta Distributions, Reparameterized by Mean", fontsize=16)
plt.xlabel("X", fontsize=14)
plt.ylabel("Probability Density", fontsize=14)
plt.grid(alpha=0.3)
plt.legend()
plt.show()
