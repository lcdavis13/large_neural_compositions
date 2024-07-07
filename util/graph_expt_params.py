import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file into a DataFrame
file_path = "../results/cNODE-paper-ocean_experiments.csv"
df = pd.read_csv(file_path)

# Filter for the same experiment hyperparameters
df = df[(df['k-folds'] == 3) & (df['early stop patience'] == 15)]

# Filter out invalid rows from exceptions
df = df[df['model parameters'] > 0]


# Extract unique models
models = df['model'].unique()

# Create the first plot for Avg Validation Score vs Model Parameters
plt.figure(figsize=(10, 6))

for model in models:
    model_data = df[df['model'] == model]
    # Sort data by 'model parameters'
    model_data = model_data.sort_values(by='model parameters')
    plt.plot(model_data['model parameters'], model_data['Avg Validation Score'], linestyle=':', marker='o', label=model)

# Set x-axis to logarithmic scale
plt.xscale('log')

# plt.ylim(bottom=0)
plt.ylim(top=0.14)

# Add labels and title
plt.xlabel('Model Parameters (Log Scale)')
plt.ylabel('Avg Validation Score')
plt.title('Avg Validation Score vs. Model Parameters for Different Models')
plt.legend(title='Model')
plt.grid(True)
plt.show()

# Create the second plot for Avg Elapsed Time vs Model Parameters
plt.figure(figsize=(10, 6))

for model in models:
    model_data = df[df['model'] == model]
    # Sort data by 'model parameters'
    model_data = model_data.sort_values(by='model parameters')
    plt.plot(model_data['model parameters'], model_data['@ Avg Elapsed Time'], linestyle=':', marker='o', label=model)


# Set x-axis to logarithmic scale
plt.xscale('log')

# plt.xlim(left=1)

# Add labels and title
plt.xlabel('Model Parameters (Log Scale)')
plt.ylabel('Avg Elapsed Time')
plt.title('Avg Elapsed Time vs. Model Parameters for Different Models')
plt.legend(title='Model')
plt.grid(True)
plt.show()