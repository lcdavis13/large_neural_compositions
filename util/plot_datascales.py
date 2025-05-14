from matplotlib.lines import Line2D
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from adjustText import adjust_text
import threading
from collections import OrderedDict
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# === Configuration ===
filename = "expt"
caption_name = "1k"
CSV_FILE_PATH = f"results/datascale_5-13_1k/{filename}.csv"
BENCHMARK_CSV_PATH = f"results/datascale_5-13_1k/benchmarks.csv"

PLOT_TITLE = f'Test Loss vs Training Examples, Hyperparameters fitted to {caption_name} examples'
X_LABEL = 'Training Examples (log scale)'
Y_LABEL = 'Test Loss (Bray-Curtis)'

# Y_LIM = (0, 0.3)
# MODELS_TO_EXCLUDE = [
#     "baseline-SLPMultSoftmax",
# ]

# Y_LIM = (0, 0.06)
# MODELS_TO_EXCLUDE = [
#     "identity", 
#     "baseline-SLPMultSoftmax",
# ]

Y_LIM = (0, 1.0)
MODELS_TO_EXCLUDE = [
    "baseline-SLPMultSoftmax",
    "baseline-SLPSoftmax",
    "baseline-cNODE0",
    "cNODE-hourglass",
    "canODE-attendFit",
    "canODE-FitMat",
]

# === Column Mappings (source-specific to standard schema) ===
main_columns = {
    'x': 'data_subset',
    'y': 'mean_test_score',
    'model': 'model_name'
}

benchmark_columns = {
    'x': 'data_subset',
    'y': 'mean_test_score',
    'model': 'model_name'
}

# === Ordered model label definitions ===
label_dict = OrderedDict([
    ("identity", "Identity"),
    ("LinearRegression", "Linear (Moore-Penrose)"),
    ("cNODE1", "cNODE1"),
    ("transformSoftmax", "Transformer + MSM"),
    ("canODE-FitMat", "canODE: fitness matrix"),
    ("canODE-attendFit", "canODE: attention fitness"),
    ("baseline-ConstSoftmax", "Const + MSM"),
    ("baseline-Linear", "Linear w/ SGD"),
    ("baseline-LinearSoftmax", "Linear + MSM"),
    ("baseline-SLPSoftmax", "SLP + MSM"),
    ("baseline-SLPMultSoftmax", "SLP + MSM"),
    ("baseline-cNODE0", "cNODE const fitness"),
    ("cNODE-hourglass", "cNODE nonlinear fitness"),
])

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

# filter unwanted models
df_combined = df_combined[~df_combined['model'].isin(MODELS_TO_EXCLUDE)]

# Only include models that are in the ordered label dict
df_combined = df_combined[df_combined['model'].isin(label_dict.keys())]

# Full sorted set of expected x-values (assumed same in both)
expected_x = sorted(df_combined['x'].unique())

# Custom extended color list
_color_list_standard = [
    # Tableau colors (default matplotlib "tab:" palette)
    'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
    'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan',

    # Additional XKCD colors
    'xkcd:cerulean', 'xkcd:bright orange', 'xkcd:grass green',
    'xkcd:brick red', 'xkcd:deep violet', 'xkcd:chocolate brown',
    'xkcd:magenta', 'xkcd:slate grey', 'xkcd:olive green', 'xkcd:cyan',

    # Pastels
    'xkcd:pastel blue', 'xkcd:pastel orange', 'xkcd:pastel green',
    'xkcd:pastel red', 'xkcd:pastel purple',
    'xkcd:pastel pink', 'xkcd:light grey', 'xkcd:pastel yellow',

    # Bonus vivid colors
    'xkcd:azure', 'xkcd:bright red', 'xkcd:chartreuse', 'xkcd:bright purple',
    'xkcd:teal', 'xkcd:coral', 'xkcd:royal blue', 'xkcd:neon green',
]

# Assign colors to models based on their order in label_dict
if len(label_dict) > len(_color_list_standard):
    raise ValueError("Not enough colors in _color_list_standard for the number of models in label_dict.")

model_colors = {
    model_name: _color_list_standard[i]
    for i, model_name in enumerate(label_dict.keys())
}


# === Plot ===
plt.figure(figsize=(10, 6))

# Store last points for annotation
last_points = []

for (model_name, source), group in df_combined.groupby(['model', 'source']):
    label = label_dict.get(model_name, model_name)
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

    if prev_x is not None and prev_y is not None:
        last_points.append((prev_x, prev_y, label, color))


def reset_text_positions():
    for text, pos, ha, va in original_states:
        text.set_position(pos)
        text.set_ha(ha)
        text.set_va(va)


resize_timer = None

def on_resize(event):
    global resize_timer
    if resize_timer is not None:
        resize_timer.cancel()
    
    # Set up a new timer: only run after 0.3s of no further resizes
    resize_timer = threading.Timer(0.3, run_adjust_text)
    resize_timer.start()

def run_adjust_text():
    reset_text_positions()
    
    adjust_text(
        texts, 
        expand_points=(1.2, 1.2),
        # arrowprops=dict(arrowstyle='-', color='gray'), 
        # min_arrow_len=100.0, 
        arrowprops=None, 
        expand_axes=True, 
        ensure_inside_axes=False,
        only_move={'explode': 'y', 'static': 'x+', 'text': 'y', 'pull': 'y'},
        # only_move='y', 
    )
    plt.draw()

# Final plot formatting
plt.xscale('log')
plt.xlabel(X_LABEL)
plt.ylabel(Y_LABEL)
plt.title(PLOT_TITLE)
plt.ylim(*Y_LIM)
plt.grid(True)

# custom legend
models_plotted = df_combined['model'].unique()
custom_legend = [
    Line2D([0], [0], color=model_colors[key], marker='o', linestyle='-', label=label_dict[key])
    for key in label_dict if key in models_plotted
]
plt.legend(handles=custom_legend, title='Model')


# Collect annotation objects
texts = []
for x, y, label, color in last_points:
    text = plt.text(x, y, label,
                    color=color,
                    fontsize=10,
                    va='center',
                    ha='left')
    texts.append(text)

# Store full text state: position and alignment
original_states = [
    (text, text.get_position(), text.get_ha(), text.get_va())
    for text in texts
]

# Connect event
plt.gcf().canvas.mpl_connect('resize_event', on_resize)

on_resize(None) 

# interpolation threshold line
plt.axvline(x=256, color='black', linestyle='dashed', linewidth=1)
plt.text(256, plt.ylim()[1]*0.95, 'Interpolation Threshold', rotation=90, va='top', ha='right', fontsize=9, color='black')

plt.tight_layout()
plt.show()
