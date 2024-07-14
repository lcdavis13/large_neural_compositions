import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file into a DataFrame
# file_path = "../results/cNODE-paper-ocean_experiments.csv"
file_path = "../results/waimea_experiments.csv"
df = pd.read_csv(file_path)

# Filter for the same experiment hyperparameters
# df = df[(df['k-folds'] == 3) & (df['early stop patience'] == 15)]
# df = df[(~df['model'].str.contains("canODE"))]
# df = df[(~df['model'].str.startswith("cNODE"))]
df = df[(df['WD_base'] == 0)]

# Filter out invalid rows from exceptions
df = df[df['model parameters'] > 0]


# Extract unique models
models = df['model'].unique()

# Create the first plot for Avg Validation Score vs Model Parameters
plt.figure(figsize=(10, 6))

for model in models:
    model_data = df[df['model'] == model]
    # Sort data by 'model parameters'
    model_data = model_data.sort_values(by='LR_base')
    plt.plot(model_data['LR_base'], model_data['Validation Score'], linestyle=':', marker='o', label=model)

# Set x-axis to logarithmic scale
plt.xscale('log')

# plt.ylim(bottom=0)
# plt.ylim(top=0.14)
# plt.ylim(top=0.3)

# Add labels and title
plt.xlabel('LR_base (Log Scale)')
plt.ylabel('Avg Validation Score')
plt.title('Avg Validation Score vs. LR_base for Different Models')
plt.legend(title='Model')
plt.grid(True)
plt.show()

# Create the second plot for Train Score vs Model Parameters
plt.figure(figsize=(10, 6))

for model in models:
    model_data = df[df['model'] == model]
    # Sort data by 'model parameters'
    model_data = model_data.sort_values(by='LR_base')
    plt.plot(model_data['LR_base'], model_data['Train Score'], linestyle=':', marker='o', label=model)


# Set x-axis to logarithmic scale
plt.xscale('log')

# plt.xlim(left=1)

# Add labels and title
plt.xlabel('LR_base (Log Scale)')
plt.ylabel('Train Score')
plt.title('Train Score vs. LR_base for Different Models')
plt.legend(title='Model')
plt.grid(True)
plt.show()