import math
import numpy as np
import matplotlib.pyplot as plt
import torch
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
csv_file_path = '../data/waimea-std_train.csv'

# Get bounds from the CSV file
try:
    min_mean, max_mean = get_bounds_from_csv(csv_file_path)
    # max_mean = 10*min_mean
    print(f"Bounds for means from CSV: min_mean = {min_mean}, max_mean = {max_mean}")
except Exception as e:
    print(f"Error: {e}")
    min_mean, max_mean = 0.025, (1.0-0.025)  # Default range if error occurs
    
# min_mean, max_mean = 0.025, (1.0-0.025)  # for fixed plots

stepnum = 6
stepsize = (max_mean - min_mean) / (stepnum)

# Parameters
mean_values = np.arange(min_mean, max_mean + 0.1*stepsize, stepsize)  # Use bounds from the CSV file
peak_stddev = 0.075

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
    
plt.legend()
plt.show()

plt.figure(figsize=(12, 8))

import torch


def resample_noisy(mean_values: torch.Tensor, peak_stddev: float) -> torch.Tensor:
    """
    For each element in the input tensor, use it as the mean to sample a single point
    from a parameterized Beta distribution.

    Args:
        mean_values (torch.Tensor): The input tensor of mean values.
        peak_stddev (float): The peak standard deviation to control the variance of the Beta distribution.

    Returns:
        torch.Tensor: A tensor of the same shape as mean_values, containing sampled values from the Beta distribution.
    """
    # Ensure input is a torch tensor
    if not isinstance(mean_values, torch.Tensor):
        raise TypeError("mean_values must be a torch.Tensor")
    
    # Handle edge cases where mean_values are outside the range (0, 1)
    sampled_values = torch.where(mean_values <= 0, torch.zeros_like(mean_values), mean_values)
    sampled_values = torch.where(mean_values >= 1, torch.ones_like(mean_values), sampled_values)
    
    # Mask values that are within the range (0, 1)
    valid_mask = (mean_values > 0) & (mean_values < 1)
    
    # Calculate the standard deviation for each valid mean value
    stddev = torch.sin(mean_values * torch.pi) * peak_stddev
    
    # Calculate the variance
    var = stddev ** 2
    
    # Calculate the Beta distribution parameters alpha and beta
    k = mean_values * (1.0 - mean_values) / var - 1.0
    alpha = mean_values * k
    beta_param = (1 - mean_values) * k
    
    # Clamp alpha and beta to avoid numerical instability (e.g., negative or zero values)
    alpha = torch.clamp(alpha, min=1e-6)
    beta_param = torch.clamp(beta_param, min=1e-6)
    
    # Sample from the Beta distribution for each mean value that is within (0, 1)
    sampled_values_within_range = torch.distributions.Beta(alpha, beta_param).sample()
    
    # Only update the sampled values for elements where 0 < mean < 1
    sampled_values = torch.where(valid_mask, sampled_values_within_range, sampled_values)
    
    return sampled_values


# Use resample_noisy to sample and plot histograms
sampled_values = []
for i in range(200):
    sampled_values.append(resample_noisy(torch.tensor(mean_values), peak_stddev).numpy())
print(np.shape(sampled_values))
for i,m in enumerate(mean_values):
    data = [s[i] for s in sampled_values]
    mean = np.mean(data)
    print(f"expected mean = {m}, mean of sampled values = {mean}")
    print(f"max = {np.max(data)}, min = {np.min(data)}")
    plt.hist(data, bins=20, density=False, alpha=0.5, label=f'mean = {m}')
plt.legend()
plt.show()

# plot the first mean separately
data = [s[0] for s in sampled_values]
plt.hist(data, bins=20, density=False, alpha=0.5, label=f'mean = {mean_values[0]}')



# Plot settings
plt.title("Various Beta Distributions, Reparameterized by Mean", fontsize=16)
plt.xlabel("X", fontsize=14)
plt.ylabel("Probability Density", fontsize=14)
plt.grid(alpha=0.3)
plt.legend()
plt.show()
