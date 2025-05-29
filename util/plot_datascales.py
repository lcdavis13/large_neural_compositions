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
# CSV_FILE_PATH = f"results/datascale_5-19_1k/{filename}.csv"
# CSV_FILE_PATH = f"results/datascale_5-25_256-random-2/{filename}.csv"
CSV_FILE_PATH = f"results/datascale_256-123/{filename}.csv"\
# caption_name = "100k"
# CSV_FILE_PATH = f"results/datascale_5-14_100k/{filename}.csv"
# BENCHMARK_CSV_PATH = f"results/datascale_5-19_1k/benchmarks.csv"
BENCHMARK_CSV_PATH = f"results/datascale_256-123/benchmarks.csv"

PLOT_TITLE = f'Test Error vs Training Examples, Hyperparameters fitted to {caption_name} examples'
X_LABEL = 'Training Examples (log scale)'
Y_LABEL = 'Test Error (Bray-Curtis, log scale)'

# DRAW_TRAIN_SCORES = True
DRAW_TRAIN_SCORES = False

PLOT_HUE_COLUMN = 'model'
PLOT_STYLE_COLUMN = 'dataset'
PLOT_SIZE_COLUMN = 'score_type'

DRAW_LEGEND_HUE = False
DRAW_LEGEND_STYLE = True
DRAW_LEGEND_SIZE = False
DRAW_LEGEND = DRAW_LEGEND_HUE or DRAW_LEGEND_STYLE or DRAW_LEGEND_SIZE

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

HLINE_MODELS = {} #{"identity"} # currently not using this feature because I had trouble getting it to match the line styles of regular lines correctly


# === Column Mappings (source-specific to standard schema) ===
main_columns = {
    'dataset': 'dataname',
    'x': 'data_subset',
    'y': 'mean_test_score',
    'train_y': 'mean_trn_loss',
    'model': 'model_name'
}

benchmark_columns = {
    'dataset': 'dataset',
    'x': 'data_subset',
    'y': 'mean_test_score',
    'train_y': 'mean_train_score',
    'model': 'model_name'
}

# === Ordered model label definitions ===
label_dict = OrderedDict([
    ("identity", "Identity"),
    ("LinearRegression-MP", "Linear (Moore-Penrose)"),
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
    rename_dict = {column_map[src]: src for src in column_map}
    df = df.rename(columns=rename_dict)
    df['source'] = source_label
    return df[list(column_map.keys()) + ['source']]


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
if PLOT_HUE_COLUMN == 'model':
    # Use persistent color mapping based on predefined model order
    base_colors = sns.color_palette('tab20', n_colors=len(label_dict))
    line_colors = {
        model: base_colors[i]
        for i, model in enumerate(label_dict.keys())
    }
else:
    # Use dynamic hue-based mapping
    unique_hue_vals = sorted(df_combined[PLOT_HUE_COLUMN].unique())
    base_colors = sns.color_palette('tab20', n_colors=len(unique_hue_vals))
    line_colors = {
        val: base_colors[i]
        for i, val in enumerate(unique_hue_vals)
    }



# === Reshape for single lineplot ===
# Melt the 'y' and 'train_y' into one column, label them as 'Test' and 'Train'
df_melted = df_combined.melt(
    id_vars=['x', 'model', 'source', 'dataset'],
    value_vars=['y', 'train_y'],
    var_name='score_type',
    value_name='score',
)

# Map 'y' -> 'Test', 'train_y' -> 'Train'
df_melted['score_type'] = df_melted['score_type'].map({'y': 'Test', 'train_y': 'Train'})

# Remove train scores if not desired
if not DRAW_TRAIN_SCORES:
    df_melted = df_melted[df_melted['score_type'] == 'Test']

# === Plot ===
plt.figure(figsize=(10, 6))
sns.set(style="whitegrid")

# # Unified plot with different styles for train/test
# sns.lineplot(
#     data=df_melted,
#     x='x',
#     y='score',
#     hue=PLOT_HUE_COLUMN,
#     style=PLOT_STYLE_COLUMN,
#     size=PLOT_SIZE_COLUMN,
#     dashes=True,
#     markers=True,
#     palette=model_colors,
#     legend=DRAW_LEGEND,
# )

# Split data
df_hline = df_melted[df_melted['model'].isin(HLINE_MODELS)]
df_normal = df_melted[~df_melted['model'].isin(HLINE_MODELS)]

# Plot normal models with seaborn
line_ax = sns.lineplot(
    data=df_normal,
    x='x',
    y='score',
    hue=PLOT_HUE_COLUMN,
    style=PLOT_STYLE_COLUMN,
    size=PLOT_SIZE_COLUMN,
    dashes=True,
    markers=True,
    palette=line_colors,
    legend=DRAW_LEGEND,
)



# Store dash patterns and line widths per style/size combo
# === Build aesthetic key mapping from actual Seaborn lines ===

style_dash_patterns = {}
style_linewidths = {}

# Determine keys used to index aesthetics (excluding 'model')
PLOT_KEYS_USED_FOR_LINE_ID = [
    col for col in [PLOT_HUE_COLUMN, PLOT_STYLE_COLUMN, PLOT_SIZE_COLUMN]
    if col and col != 'model'
]

def extract_key(row):
    keys = [str(row[col]) for col in PLOT_KEYS_USED_FOR_LINE_ID]
    return tuple(keys) if len(keys) > 1 else (keys[0] if keys else None)

# Build a lookup: label -> full row from df_normal
label_to_row = {}

# For each aesthetic column, register all known values (coerced to str for label matching)
# Register every combination from df_normal
for _, row in df_normal.iterrows():
    key = extract_key(row)
    if key not in style_dash_patterns:
        style_dash_patterns[key] = None  # fill with actual pattern after matching with line
        style_linewidths[key] = None


print("\n--- Registered line style keys ---")
for k in style_dash_patterns:
    print(f"registered: {k!r}")

# Now map line styles based on label → key
# Build style mappings based on exact match of line label to df_normal
# Try matching line to its actual row by iterating through keys
for line in line_ax.lines:
    dash_pattern = getattr(line, "_dash_pattern", None)
    linewidth = line.get_linewidth()

    for key in style_dash_patterns:
        if style_dash_patterns[key] is not None:
            continue  # already matched

        # Convert key back into a dataframe mask
        if isinstance(key, tuple):
            match_mask = pd.Series(True, index=df_normal.index)
            for col, val in zip(PLOT_KEYS_USED_FOR_LINE_ID, key):
                match_mask &= df_normal[col].astype(str) == str(val)
        else:
            match_mask = df_normal[PLOT_KEYS_USED_FOR_LINE_ID[0]].astype(str) == str(key)

        # If this key is actually represented in the data
        if match_mask.any():
            style_dash_patterns[key] = dash_pattern
            style_linewidths[key] = linewidth
            break  # move on to next line




print("\n--- Mapped line styles ---")
for key, dash in style_dash_patterns.items():
    linewidth = style_linewidths.get(key, 0)
    print(f"key: {key!r}, dash: {dash}, linewidth: {linewidth}")







for model_name, model_df in df_hline.groupby('model'):
    label = label_dict.get(model_name, model_name)

    # Determine how to group hline models
    group_cols = PLOT_KEYS_USED_FOR_LINE_ID or [None]
    grouped = model_df.groupby(group_cols) if group_cols[0] else [(None, model_df)]

    for group_key, subdf in grouped:
        if subdf.empty:
            continue

        y_val = subdf['score'].iloc[0]
        x_min, x_max = subdf['x'].min(), subdf['x'].max()

        # Get color using hue column value
        color_val = subdf[PLOT_HUE_COLUMN].iloc[0] if PLOT_HUE_COLUMN else None
        color = line_colors.get(color_val, 'gray')

        # Build style/size key for line style lookup
        if isinstance(group_key, tuple):
            key_input = {col: str(val) for col, val in zip(PLOT_KEYS_USED_FOR_LINE_ID, group_key)}
        elif group_key is not None:
            key_input = {PLOT_KEYS_USED_FOR_LINE_ID[0]: str(group_key)}
        else:
            key_input = {}


        key = extract_key(key_input)

        print(f"lookup key: {key!r}")


        dash_pattern = style_dash_patterns.get(key, "solid")  # fallback dash
        if dash_pattern is None:
            dash_pattern = "solid"
        linewidth = style_linewidths.get(key, 0) # big problem that we're hitting the default

        print(f"Plotting hline for {label} with key={key}, dash={dash_pattern}, linewidth={linewidth}, at y={y_val}")

        plt.hlines(
            y=y_val,
            xmin=x_min,
            xmax=x_max,
            color=color,
            linestyle=dash_pattern,
            linewidth=linewidth,
            label=label if DRAW_LEGEND else None
        )














# Store last points for annotation (only for Test)
last_points = []
for model_name, group in df_melted[df_melted['score_type'] == 'Test'].groupby('model'):
    label = label_dict.get(model_name, model_name)
    color = line_colors.get(model_name, 'gray')

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

# # custom legend
# if DRAW_LEGEND:
#     models_plotted = df_combined['model'].unique()
#     custom_legend = [
#         Line2D([0], [0], color=model_colors[key], marker='o', linestyle='-', label=label_dict[key])
#         for key in label_dict if key in models_plotted
#     ]
#     plt.legend(handles=custom_legend, title='Model')


import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.legend import Legend

def filter_legend_sections(ax, hue=None, style=None, size=None,
                           show_hue=True, show_style=True, show_size=True):
    """
    Filters a Seaborn legend to keep or remove specific sections (hue, style, size),
    based on the original variable names passed to the plot.

    Parameters:
    - ax: matplotlib Axes from a Seaborn plot
    - hue, style, size: column names used in the Seaborn plot
    - show_hue/style/size: booleans indicating which sections to keep
    """
    legend: Legend = ax.get_legend()
    if legend is None:
        return

    handles = legend.legend_handles
    labels = [text.get_text() for text in legend.texts]

    # Reconstruct grouped sections by looking for known column titles
    sections = {}
    current = None
    for h, l in zip(handles, labels):
        if l in {hue, style, size}:
            current = l
            sections[current] = []
        elif current:
            sections[current].append((h, l))

    # Choose which sections to include
    include_sections = set()
    if show_hue and hue:
        include_sections.add(hue)
    if show_style and style:
        include_sections.add(style)
    if show_size and size:
        include_sections.add(size)

    # Build the filtered legend content
    new_handles = []
    new_labels = []

    for section, entries in sections.items():
        if section in include_sections:
            # Add the section title
            new_handles.append(plt.Line2D([], [], linestyle=''))
            new_labels.append(section)
            for h, l in entries:
                new_handles.append(h)
                new_labels.append(l)

    if new_handles:
        ax.legend(new_handles, new_labels,
                  bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=13)
    else:
        ax.legend().remove()



if DRAW_LEGEND:
    filter_legend_sections(
        plt.gca(),
        # df_melted,
        hue=PLOT_HUE_COLUMN,
        style=PLOT_STYLE_COLUMN,
        size=PLOT_SIZE_COLUMN,
        show_hue=DRAW_LEGEND_HUE,
        show_style=DRAW_LEGEND_STYLE,
        show_size=DRAW_LEGEND_SIZE
    )


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



def report_sample_counts(df, groupby_cols):
    """
    Analyze and print the number of samples per combination of groupby_cols and x.
    Also detects missing (zero-count) combinations where presence is expected.

    Parameters:
    - df: DataFrame containing the data.
    - groupby_cols: List of column names used for grouping (e.g., ['model', 'y_dataset']).
    """
    groupby_cols = list(groupby_cols)  # Ensure it's a list
    full_group_cols = groupby_cols + ['x']

    # Determine all expected x values
    expected_x = sorted(df['x'].unique())

    # Determine all unique combinations of groupby columns
    group_keys = df[groupby_cols].drop_duplicates()

    # Count existing samples
    sample_counts = df.groupby(full_group_cols).size().reset_index(name='count')

    # Build expected combinations (cartesian product of group_keys and x)
    from itertools import product

    expected_rows = pd.DataFrame([
        {**dict(zip(groupby_cols, group_vals)), 'x': x}
        for group_vals, x in product(group_keys.values.tolist(), expected_x)
    ])

    # Merge with actual counts (fill missing with zero)
    merged = expected_rows.merge(sample_counts, on=full_group_cols, how='left').fillna({'count': 0})
    merged['count'] = merged['count'].astype(int)

    # Determine expected count per group (maximum count observed)
    expected_per_group = (
        merged.groupby(groupby_cols)['count'].max().to_dict()
    )

    # Build normal and anomaly reports
    from collections import defaultdict
    models_by_expected = defaultdict(list)
    anomalies = defaultdict(lambda: defaultdict(list))  # group -> count -> [x]

    for _, row in merged.iterrows():
        group = tuple(row[col] for col in groupby_cols)
        x = row['x']
        count = row['count']
        expected = expected_per_group[group]
        if count != expected:
            anomalies[group][count].append(x)

    for group, expected in expected_per_group.items():
        models_by_expected[expected].append(group)

    # Print report
    print("=== Sample Count Report ===")
    for expected_count in sorted(models_by_expected.keys(), reverse=True):
        for group in sorted(models_by_expected[expected_count]):
            label = ', '.join(f"{col}={val}" for col, val in zip(groupby_cols, group))
            print(f"{expected_count} samples: {label}")

    if any(anomalies.values()):
        print("\nAnomalies:")
        for group in sorted(anomalies.keys()):
            group_str = ', '.join(f"{col}={val}" for col, val in zip(groupby_cols, group))
            for count in sorted(anomalies[group].keys(), reverse=True):
                x_vals = sorted(anomalies[group][count])
                x_str = ",".join(map(str, x_vals))
                print(f"  {count} samples: {group_str} x={x_str}")
    else:
        print("\nNo anomalies found.")



# Run the dynamic sample count report
grouping_cols = [col for col in [PLOT_HUE_COLUMN, PLOT_STYLE_COLUMN, PLOT_SIZE_COLUMN] if col]
report_sample_counts(df_combined, groupby_cols=grouping_cols)

