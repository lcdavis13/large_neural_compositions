import pandas as pd
import matplotlib.pyplot as plt

# === Configuration ===
CSV_FILE_PATH = "results/datascale_4-31/batch_5972537_100k.csv"
PLOT_TITLE = 'Test Loss vs Training Examples, Hyperparameters fitted to 100k examples'
X_LABEL = 'Training Examples (log scale)'
Y_LABEL = 'Test Loss (Bray-Curtis)'

# Load data
df = pd.read_csv(CSV_FILE_PATH)

# Plot
plt.figure(figsize=(10, 6))
for model_name, group in df.groupby('model_name'):
    group_sorted = group.sort_values('data_subset')
    plt.plot(group_sorted['data_subset'], group_sorted['test loss'], label=model_name, marker='o')

plt.xscale('log')
plt.xlabel(X_LABEL)
plt.ylabel(Y_LABEL)
plt.title(PLOT_TITLE)
plt.ylim(0, 0.3)  # Set Y-axis range
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
