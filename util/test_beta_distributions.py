import math

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

# Parameters
mean_values = np.arange(0.025, 1.0, 0.025)  # Means (multiples of 0.1)
peak_stddev = 0.05

# Prepare plot
plt.figure(figsize=(12, 8))

# Generate and plot Beta distributions
x = np.linspace(0, 1, 500)
for x_mean in mean_values:
    stddev = math.sin(x_mean*math.pi)*peak_stddev
    var = stddev ** 2
    k = x_mean*(1.0 - x_mean)/var - 1.0
    alpha = x_mean * k
    beta_param = (1 - x_mean) * k
    pdf = beta.pdf(x, alpha, beta_param)
    plt.plot(x, pdf, label=f'x = {x_mean:.1f}, alpha = {alpha:.1f}, beta = {beta_param:.1f}')

# currently unused, but demonstrates how to sample
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
plt.title("Beta Distributions with Shared k and Varying Means", fontsize=16)
plt.xlabel("x", fontsize=14)
plt.ylabel("Density", fontsize=14)
# plt.legend(title="Parameters", fontsize=10)
plt.grid(alpha=0.3)
plt.show()
