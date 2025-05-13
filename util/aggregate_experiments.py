import os
import glob
import pandas as pd
import re
import numpy as np

# === Canonical output column names ===
MODEL_COL = "model_CPxDPxHP_config"
VAL_LOSS_COL = "mean_val_score"
TEST_LOSS_COL = "mean_test_score"
TRAIN_LOSS_COL = "mean_trn_score"
VALID_ROWS_COL = "valid_folds"
JOB_NUMBER_COL = "job_number"

# === Mapping of canonical names to possible aliases ===
COLUMN_MAP = {
    MODEL_COL: [MODEL_COL, "model_config"],
    VAL_LOSS_COL: ["mini_epoch @ val_score", VAL_LOSS_COL],  # unfortunately I have to put the old column name first because the new name existed for a different stat previously
    TEST_LOSS_COL: [TEST_LOSS_COL, "test loss"],
    TRAIN_LOSS_COL: ["mini_epoch @ trn_loss", TRAIN_LOSS_COL],  # unfortunately I have to put the old column name first because the new name existed for a different stat previously
}

# === Columns that must be > 0 and finite ===
REQUIRED_POSITIVE_COLUMNS = [VAL_LOSS_COL, TRAIN_LOSS_COL]

rootfolder = "./results/hpsearch_4-22_1k"
# rootfolder = "./results/hpsearch_4-22_100k"

folder = f"{rootfolder}/expt/"
path_pattern = f"{folder}*_job*_experiments.csv"
hp_folder = f"{rootfolder}/hp/"
HP_REGEX = r'(.+?)_job(\d+)_experiments\.csv'
output_path = f"{folder}_experiments.csv"


def resolve_column_name(df, possible_names):
    """Return the first matching column name from a list."""
    for name in possible_names:
        if name in df.columns:
            return name
    return None

def find_matching_hyperparams(expt_file_path, model_value, model_col):
    """Find matching row from hyperparams file using experiment path and model_col."""
    filename = os.path.basename(expt_file_path)

    match = re.match(HP_REGEX, filename)
    if not match:
        print(f"Filename pattern mismatch: {filename}")
        return None

    prefix = match.group(1)
    job_number = match.group(2)

    hp_file = f"{hp_folder.rstrip(os.sep)}/{prefix}_job{job_number}_hyperparams.csv"
    if not os.path.exists(hp_file):
        # print(f"Hyperparams file {hp_file} does not exist.")
        return None

    try:
        hp_df = pd.read_csv(hp_file)
        if model_col in hp_df.columns:
            match_row = hp_df[hp_df[model_col] == model_value]
            if not match_row.empty:
                return match_row.iloc[0].to_dict()
            else:
                print(f"No matching row in {hp_file} for model config: {model_value}")
        else:
            print(f"Model column '{model_col}' not found in {hp_file}")
    except Exception as e:
        print(f"Error reading or processing {hp_file}: {e}")

    return None


def load_and_process_csv_files(path_pattern):
    csv_files = glob.glob(path_pattern)

    if not csv_files:
        print("No CSV files found matching the pattern.")
        return None

    all_data = []

    for file_path in csv_files:
        try:
            job_number_match = re.search(r'job(\d+)', os.path.basename(file_path))
            job_number = int(job_number_match.group(1)) if job_number_match else None

            df = pd.read_csv(file_path)

            # Resolve actual column names from the mapping
            resolved_cols = {key: resolve_column_name(df, aliases) for key, aliases in COLUMN_MAP.items()}

            # Drop rows with nonpositive, NaN, or infinite values in required columns
            for canonical_col in REQUIRED_POSITIVE_COLUMNS:
                actual_col = resolved_cols.get(canonical_col)
                if actual_col and actual_col in df.columns:
                    df = df[
                        (df[actual_col] >= 0.0) &
                        (np.isfinite(df[actual_col])) &
                        (~df[actual_col].isna())
                    ]

            # Group by model column if available
            model_actual = resolved_cols.get(MODEL_COL)
            if model_actual and model_actual in df.columns:
                grouped = df.groupby(model_actual)
            else:
                grouped = [(None, df)]

            for model_value, group_df in grouped:
                numeric_cols = group_df.select_dtypes(include=['number']).columns
                non_numeric_cols = group_df.select_dtypes(exclude=['number']).columns

                averaged_data = {col: group_df[col].mean() for col in numeric_cols}
                first_value_data = {col: group_df[col].iloc[0] if not group_df[col].empty else None for col in non_numeric_cols}

                # Determine the valid row count using the column if it exists, otherwise compute
                if VALID_ROWS_COL in group_df.columns:
                    valid_row_count = group_df[VALID_ROWS_COL].iloc[0]
                else:
                    valid_row_count = len(group_df)

                # Rename resolved columns to canonical names
                canonical_data = {}
                for canonical, actual in resolved_cols.items():
                    if actual and actual in averaged_data:
                        canonical_data[canonical] = averaged_data.pop(actual)
                    elif actual and actual in first_value_data:
                        canonical_data[canonical] = first_value_data.pop(actual)

                processed_row = {JOB_NUMBER_COL: job_number}
                if model_actual:
                    processed_row[MODEL_COL] = model_value

                processed_row.update(averaged_data)
                processed_row.update(first_value_data)
                processed_row.update(canonical_data)
                processed_row[VALID_ROWS_COL] = valid_row_count

                # Try to find matching hyperparams and merge
                hp_data = find_matching_hyperparams(file_path, model_value, model_actual)
                if hp_data:
                    processed_row.update(hp_data)

                all_data.append(processed_row)

        except Exception as e:
            print(f"Error processing file {file_path}: {e}")

    return pd.DataFrame(all_data)


def save_master_csv(master_df, output_path):
    try:
        column_order = [
            'dataset', 'data_subset', 'model_name', MODEL_COL, JOB_NUMBER_COL, VALID_ROWS_COL, VAL_LOSS_COL, TEST_LOSS_COL, TRAIN_LOSS_COL,
            'lr', 'reptile_lr', 'wd', 'identity_gate',
            'cnode_bias', 'cnode1_init_zero',
            'attend_dim_per_head', 'num_heads', 'depth', 'dropout', 'ode_timestep_file', 'hidden_dim', 'ffn_dim_multiplier',
            'noise', 'interpolate', 'interpolate_noise',
            "update steps", "Avg Training Loss", "Elapsed Time", "VRAM (GB)", "Peak RAM (GB)",
            'val_loss_n', 'val_loss_median', 'val_loss_std', 'val_loss_min', 'val_loss_max',
            'test_loss_n', 'test_loss_median', 'test_loss_std', 'test_loss_min', 'test_loss_max'
        ]

        always_last_columns = [
            'device', 'solver', 'early_stop', 'whichfold', 'batchid', 'taskid', 'jobid', 'data_configid', 'configid', 'headless'
        ]

        existing_columns = [col for col in column_order if col in master_df.columns]
        remaining_columns = [col for col in master_df.columns if col not in existing_columns + always_last_columns]
        existing_last_columns = [col for col in always_last_columns if col in master_df.columns]

        ordered_columns = existing_columns + remaining_columns + existing_last_columns
        master_df = master_df[ordered_columns]

        if VAL_LOSS_COL in master_df.columns:
            master_df = master_df.sort_values(by=VAL_LOSS_COL, ascending=True)

        master_df.to_csv(output_path, index=False)
        print(f"Master CSV file saved successfully at {output_path}")
    except Exception as e:
        print(f"Failed to save master CSV file: {e}")


def main():
    master_df = load_and_process_csv_files(path_pattern)
    if master_df is not None:
        save_master_csv(master_df, output_path)


if __name__ == "__main__":
    main()
