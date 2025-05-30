import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import math

# sortcolumn = 'Validation Score'
sortcolumn = 'Train Score'

# normalize = True
normalize = False

# Load the CSV file
# df = pd.read_csv("../results/cNODE-paper-ocean_experiments.csv")
df = pd.read_csv("../results/waimea_experiments.csv")

# Filter by model="canODE-transformer" and k-folds=3
filtered_df = df[(df['model parameters'] > 0) & (df['Train Score'] < 0.4)]


# Compute new columns
# filtered_df['attend_dims_div_heads'] = filtered_df['attend_dim'] / filtered_df['num_heads']
# filtered_df['heads_div_attend_dims'] = filtered_df['num_heads'] / filtered_df['attend_dim']
# filtered_df['attend_dims_times_heads'] = filtered_df['attend_dim'] * filtered_df['num_heads']
# filtered_df['attend_dim_times_depth'] = filtered_df['attend_dim'] * filtered_df['depth']
# filtered_df['ffn_dim_times_attend_dim'] = filtered_df['ffn_dim_multiplier'] * filtered_df['attend_dim']
# filtered_df['heads_times_depth'] = filtered_df['num_heads'] * filtered_df['depth']
# filtered_df['ffn_dim_times_heads'] = filtered_df['ffn_dim_multiplier'] * filtered_df['num_heads']
# filtered_df['ffn_dim_times_depth'] = filtered_df['ffn_dim_multiplier'] * filtered_df['depth']
# filtered_df['ffn_dim_div_depth'] = filtered_df['ffn_dim_multiplier'] / filtered_df['depth']
# filtered_df['depth_div_ffn_dim'] = filtered_df['depth'] / filtered_df['ffn_dim_multiplier']
# filtered_df['mb_times_accum'] = filtered_df['minibatch_examples'] * filtered_df['accumulated_minibatches']
filtered_df['log_lr'] = filtered_df['LR_base'].apply(lambda x: math.log(x))
# filtered_df['log_wd'] = filtered_df['WD_base'].apply(lambda x: math.log(x) if x > 0 else -15)

# Sort by Avg Validation Score ascending
sorted_df = filtered_df.sort_values(by=sortcolumn)
sorted_df = sorted_df.reset_index()

# List of columns of interest to display
columns_of_interest = [
    # 'Validation Score',
    # 'Train Score',
    # 'model parameters', # trend: a few values around the minimum, with small spikes corresponding to spikes in attend_dim
    # 'attend_dim', # trend: 3 minimum values with an emphasis on the 2nd value, but absolute best was at minimum. 3rd value and worse are the only ones occurring in the bad trials, except one small downward spike to the 2nd value.
    # 'num_heads', # trend: 2 minimum values equally, plus one fluke into the 3rd value, but absolute best was at minimum. However, all of these values occur a lot in the bad trials too.
    # 'depth', # not much. Perhaps a slight preference for the lower value, but best was at higher value.
    # 'ffn_dim_multiplier', # not much. Slight preference for lower 2 values. Best was at lowest value.
    # 'attend_dims_div_heads', # meh
    # 'attend_dims_times_heads', # meh
    # 'attend_dim_times_depth', # this might actually be a better predictor than attend_dim
    # 'heads_div_attend_dims',
    # 'ffn_dim_times_attend_dim' # decent but worse than attend_dim
    # 'ffn_dim_times_heads', # maybe slightly better than num_heads but still has a big mixture in the bad trials
    # 'heads_times_depth', # meh
    # 'ffn_dim_times_depth', # no
    # 'ffn_dim_div_depth', # no
    # 'depth_div_ffn_dim',
    
    # 'minibatch_examples',
    # # 'accumulated_minibatches',
    'LR_base',
    # 'WD_base',
    # 'learning rate',
    # 'weight decay',
    # 'mb_times_accum',
    # 'log_lr',
    # 'log_wd',
]

# Normalize each column to mean 0 variance 1
if normalize:
    scaler = StandardScaler()
    normalized_values = scaler.fit_transform(sorted_df[columns_of_interest])
else:
    normalized_values = sorted_df[columns_of_interest]
pd.set_option('display.max_columns', 5000)
normalized_df = pd.DataFrame(normalized_values, columns=columns_of_interest)

# Find the index to place the vertical bar
separation_index = sorted_df[sorted_df[sortcolumn] < 0.6].index
print(separation_index[-1])
print(normalized_df.head(separation_index[-1] + 1))
if len(separation_index) > 0:
    separation_index = separation_index[-1]
else:
    separation_index = None

# Plot each column as lines on a single plot where the x-axis is the sorted index
plt.figure(figsize=(10, 6))
for column in columns_of_interest:
    plt.plot(normalized_df.index, normalized_df[column], label=column)
    
# Add vertical bar to denote separation if it exists
if separation_index is not None:
    plt.axvline(x=sorted_df.index.get_loc(separation_index), color='r', linestyle='--', label=f'{sortcolumn} < 0.1')

plt.xlabel('Sorted Index')
plt.ylabel('Normalized Values')
plt.title('Normalized Columns of Interest')
plt.legend()
plt.show()
