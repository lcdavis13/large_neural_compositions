import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# === Configuration ===
# filename = "batch_5972537_100k"
filename = "batch_5972295_1kLow"
caption_name = "Low Epochs, 1k"
CSV_FILE_PATH = f"results/datascale_4-31/{filename}.csv"
PLOT_TITLE = f'Test Loss vs Training Examples, Hyperparameters fitted to {caption_name} examples'
X_LABEL = 'Training Examples (log scale)'
Y_LABEL = 'Test Loss (Bray-Curtis)'
Y_LIM = (0, 0.3)

# Load data
df = pd.read_csv(CSV_FILE_PATH)

# Full sorted set of expected x-values
expected_x = sorted(df['data_subset'].unique())
x_to_index = {x: i for i, x in enumerate(expected_x)}

# Sort model names for consistent color mapping
model_names = sorted(df['model_name'].unique())

# Color map
model_names = df['model_name'].unique()
color_map = cm.get_cmap('tab10', len(model_names))
model_colors = dict(zip(model_names, [color_map(i) for i in range(len(model_names))]))

# Plot
plt.figure(figsize=(10, 6))


for model_name, group in df.groupby('model_name'):
    color = model_colors[model_name]
    label_added = False

    # Create full list of y values aligned with expected_x, treating < 0 as missing
    model_data = {
        row['data_subset']: (row['test loss'] if row['test loss'] >= 0 else None)
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
                         label=model_name if not label_added else None)
                label_added = True
            prev_x, prev_y = x, y
        else:
            # Found a gap — look ahead for the next available point
            j = i + 1
            while j < len(expected_x) and y_values[j] is None:
                j += 1
            if j < len(expected_x):
                x_after = expected_x[j]
                y_after = y_values[j]
                if prev_x is not None and prev_y is not None:
                    # Draw dashed line across the gap
                    plt.plot([prev_x, x_after], [prev_y, y_after],
                             color=color,
                             linestyle='dashed')
                # Move prev_x/y ahead to skip solid line over the gap
                prev_x, prev_y = x_after, y_after
            else:
                prev_x, prev_y = None, None  # No more valid points

# Final plot settings
plt.xscale('log')
plt.xlabel(X_LABEL)
plt.ylabel(Y_LABEL)
plt.title(PLOT_TITLE)
plt.ylim(*Y_LIM)
plt.legend(title='Model')
plt.grid(True)
plt.tight_layout()
plt.show()
