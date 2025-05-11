import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# === Configuration ===
filename = "batch_5972537_100k"
caption_name = "100k"
CSV_FILE_PATH = f"results/datascale_4-31/{filename}.csv"
BENCHMARK_CSV_PATH = f"results/datascale_4-31/benchmarks/expt.csv"

PLOT_TITLE = f'Test Loss vs Training Examples, Hyperparameters fitted to {caption_name} examples'
X_LABEL = 'Training Examples (log scale)'
Y_LABEL = 'Test Loss (Bray-Curtis)'
Y_LIM = (0, 0.06)

# === Column Mappings (source-specific to standard schema) ===
main_columns = {
    'x': 'data_subset',
    'y': 'test loss',
    'model': 'model_name'
}

benchmark_columns = {
    'x': 'data_subset',
    'y': 'mean_test_score',
    'model': 'model_name'
}

# === Load and normalize data ===
def load_and_rename(csv_path, column_map, source_label):
    df = pd.read_csv(csv_path)
    df = df.rename(columns={
        column_map['x']: 'x',
        column_map['y']: 'y',
        column_map['model']: 'model'
    })
    df['source'] = source_label
    return df[['x', 'y', 'model', 'source']]

df_main = load_and_rename(CSV_FILE_PATH, main_columns, 'Main')
df_benchmark = load_and_rename(BENCHMARK_CSV_PATH, benchmark_columns, 'Benchmark')

# Combine datasets
df_combined = pd.concat([df_main, df_benchmark], ignore_index=True)

# Full sorted set of expected x-values (assumed same in both)
expected_x = sorted(df_combined['x'].unique())

# Get full set of model names for color mapping
all_model_names = sorted(df_combined['model'].unique())
color_map = cm.get_cmap('tab10', len(all_model_names))
model_colors = dict(zip(all_model_names, [color_map(i) for i in range(len(all_model_names))]))

# === Plot ===
plt.figure(figsize=(10, 6))

for (model_name, source), group in df_combined.groupby(['model', 'source']):
    label = f"{model_name} ({source})"
    color = model_colors.get(model_name, 'gray')
    label_added = False

    # Align y-values to expected_x
    model_data = {
        row['x']: (row['y'] if row['y'] >= 0 else None)
        for _, row in group.iterrows()
    }
    y_values = [model_data.get(x, None) for x in expected_x]

    prev_x, prev_y = None, None
    for i, (x, y) in enumerate(zip(expected_x, y_values)):
        if y is not None:
            if prev_x is not None and prev_y is not None:
                plt.plot([prev_x, x], [prev_y, y],
                         color=color,
                         linestyle='solid',
                         marker='o',
                         label=label if not label_added else None)
                label_added = True
            prev_x, prev_y = x, y
        else:
            j = i + 1
            while j < len(expected_x) and y_values[j] is None:
                j += 1
            if j < len(expected_x):
                x_after = expected_x[j]
                y_after = y_values[j]
                if prev_x is not None and prev_y is not None:
                    plt.plot([prev_x, x_after], [prev_y, y_after],
                             color=color,
                             linestyle='dashed')
                prev_x, prev_y = x_after, y_after
            else:
                prev_x, prev_y = None, None

# Final plot formatting
plt.xscale('log')
plt.xlabel(X_LABEL)
plt.ylabel(Y_LABEL)
plt.title(PLOT_TITLE)
plt.ylim(*Y_LIM)
plt.legend(title='Model (Source)')
plt.grid(True)
plt.tight_layout()
plt.show()
