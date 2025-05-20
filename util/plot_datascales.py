from matplotlib.lines import Line2D
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from adjustText import adjust_text
import threading
from collections import OrderedDict
from collections import defaultdict

# === Configuration ===
filename = "expt"
caption_name = "1k"
CSV_FILE_PATH = f"results/datascale_5-19_1k/{filename}.csv"
# caption_name = "100k"
# CSV_FILE_PATH = f"results/datascale_5-14_100k/{filename}.csv"
BENCHMARK_CSV_PATH = f"results/datascale_5-19_1k/benchmarks.csv"

PLOT_TITLE = f'Test Error vs Training Examples, Hyperparameters fitted to {caption_name} examples'
X_LABEL = 'Training Examples (log scale)'
Y_LABEL = 'Test Error (Bray-Curtis, log scale)'

# DRAW_TRAIN_SCORES = True
DRAW_TRAIN_SCORES = False

# DRAW_LEGEND = True
DRAW_LEGEND = False

# # Default
# Y_LIM = (0, 0.3)
# MODELS_TO_EXCLUDE = [
#     "baseline-SLPMultSoftmax",
# ]

# Zoomed in
# Y_LIM = (0, 0.06)
# MODELS_TO_EXCLUDE = [
#     "identity", 
#     "baseline-SLPMultSoftmax",
# ]

# # Zoomed out
# Y_LIM = (0, 1.0)
# MODELS_TO_EXCLUDE = [
#     "baseline-SLPMultSoftmax",
#     "baseline-SLPSoftmax",
#     "baseline-cNODE0",
#     "cNODE-hourglass",
#     "canODE-attendFit",
#     "canODE-FitMat",
# ]

# new logscale version
Y_LIM = (None, 1.0)
X_LIM = (10, None)
MODELS_TO_EXCLUDE = [
    "baseline-SLPMultSoftmax",
    "baseline-cNODE0",
]

# === Column Mappings (source-specific to standard schema) ===
main_columns = {
    'x': 'data_subset',
    'y': 'mean_test_score',
    'train_y': 'mean_trn_loss',
    'model': 'model_name'
}

benchmark_columns = {
    'x': 'data_subset',
    'y': 'mean_test_score',
    'train_y': 'mean_train_score',
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
        column_map['train_y']: 'train_y',
        column_map['model']: 'model'
    })
    df['source'] = source_label
    return df[['x', 'y', 'train_y', 'model', 'source']]

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

# === Plot ===
# Fixed color mapping to ensure consistent colors across filtered runs
base_colors = sns.color_palette('tab20', n_colors=len(label_dict))
model_colors = {
    model_name: base_colors[i]
    for i, model_name in enumerate(label_dict.keys())
}

# === Reshape for single lineplot ===
# Melt the 'y' and 'train_y' into one column, label them as 'Test' and 'Train'
df_melted = df_combined.melt(
    id_vars=['x', 'model', 'source'],
    value_vars=['y', 'train_y'],
    var_name='score_type',
    value_name='score'
)

# Map 'y' -> 'Test', 'train_y' -> 'Train'
df_melted['score_type'] = df_melted['score_type'].map({'y': 'Test', 'train_y': 'Train'})

# Remove train scores if not desired
if not DRAW_TRAIN_SCORES:
    df_melted = df_melted[df_melted['score_type'] == 'Test']

# === Plot ===
plt.figure(figsize=(10, 6))
sns.set(style="whitegrid")

# Unified plot with different styles for train/test
sns.lineplot(
    data=df_melted,
    x='x',
    y='score',
    hue='model',
    style='score_type',
    dashes=True,
    markers=True,
    palette=model_colors,
    legend=False  # Custom legend below
)



# Store last points for annotation (only for Test)
last_points = []
for model_name, group in df_melted[df_melted['score_type'] == 'Test'].groupby('model'):
    label = label_dict.get(model_name, model_name)
    color = model_colors.get(model_name, 'gray')

    group_sorted = group.sort_values('x')
    last_row = group_sorted.dropna(subset=['score']).iloc[-1]
    last_points.append((last_row['x'], last_row['score'], label, color))


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
plt.yscale('log')
plt.xlabel(X_LABEL)
plt.ylabel(Y_LABEL)
plt.title(PLOT_TITLE)
plt.ylim(*Y_LIM)
plt.xlim(*X_LIM)
plt.grid(True)

# custom legend
if DRAW_LEGEND:
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
                    fontsize=12,
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



def report_sample_counts(df):
    """
    Analyze and print the number of samples used per (model, x).
    Reports expected counts per model and consolidates anomalies.
    """
    # Count samples per (model, x)
    sample_counts = df.groupby(['model', 'x']).size().reset_index(name='count')

    # Determine expected sample count per model (max count)
    expected_counts = sample_counts.groupby('model')['count'].max().to_dict()

    # Group models by expected count
    models_by_expected = defaultdict(list)
    for model, expected in expected_counts.items():
        models_by_expected[expected].append(model)

    # Consolidate anomalies: group by (model, count) → list of x
    anomalies = defaultdict(lambda: defaultdict(list))  # model -> count -> [x]
    for _, row in sample_counts.iterrows():
        model, x, count = row['model'], row['x'], row['count']
        expected = expected_counts[model]
        if count != expected:
            anomalies[model][count].append(x)

    # === Print Report ===
    print("=== Sample Count Report ===")
    for expected_count in sorted(models_by_expected.keys(), reverse=True):
        model_list = sorted(models_by_expected[expected_count])
        print(f"{expected_count} samples: {', '.join(model_list)}")

    if any(anomalies.values()):
        print("\nAnomalies:")
        for model in sorted(anomalies.keys()):
            for count in sorted(anomalies[model].keys(), reverse=True):
                x_vals = sorted(anomalies[model][count])
                x_str = ",".join(map(str, x_vals))
                print(f"  {count} samples: {model} x={x_str}")
    else:
        print("\nNo anomalies found.")

# === Run the report ===
report_sample_counts(df_combined)
