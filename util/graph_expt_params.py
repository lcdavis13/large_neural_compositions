import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file into a DataFrame
file_path = "../results/cNODE-paper-ocean_experiments.csv"
df = pd.read_csv(file_path)

# Extract unique models
models = df['model'].unique()

# Create the plot
plt.figure(figsize=(10, 6))

for model in models:
    model_data = df[df['model'] == model]
    # Sort data by 'model parameters'
    model_data = model_data.sort_values(by='model parameters')
    plt.plot(model_data['model parameters'], model_data['Avg Validation Score'], linestyle=':', marker='o', label=model)

# Set x-axis to logarithmic scale
plt.xscale('log')

# Add labels and title
plt.xlabel('Model Parameters')
plt.ylabel('Avg Validation Score')
plt.title('Avg Validation Score vs. Model Parameters for Different Models')
plt.legend(title='Model')
plt.ylim(bottom=0)
plt.grid(True)
plt.show()