import os
os.environ.setdefault("SOLVER", "trapezoid")
import torch
import hashlib
from collections import defaultdict
import sys
import torch
import experiment as expt
import hyperparams
import loss_function
import warnings

import chunked_dataset

def hash_tensor(tensor):
    """Compute a hash of a tensor's binary content (CPU, uint8) to detect duplicates."""
    if tensor.device.type != 'cpu':
        tensor = tensor.cpu()
    return hashlib.md5(tensor.numpy().tobytes()).hexdigest()

def check_leakage(data_train, data_valid, data_test, fold_num=0, score_fn=None, verbosity=0):
    import torch
    import hashlib
    import warnings
    from collections import defaultdict

    def hash_tensor(tensor):
        if tensor.device.type != 'cpu':
            tensor = tensor.cpu()
        return hashlib.md5(tensor.numpy().tobytes()).hexdigest()

    seen_hashes = defaultdict(set)
    within_set_dupes_count = defaultdict(int)
    cross_set_dupes_count = defaultdict(int)
    row_counts = defaultdict(int)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datasets = {'train': data_train, 'valid': data_valid, 'test': data_test}

    for name, dataset in datasets.items():
        for chunk in dataset.stream_by_chunk(device=device):
            x_chunk = chunk[chunked_dataset.DK_BINARY]
            for i in range(x_chunk.size(0)):
                sample = x_chunk[i]
                sample_hash = hash_tensor(sample)
                row_counts[name] += 1

                # Check for duplicates within the same set
                if sample_hash in seen_hashes[name]:
                    within_set_dupes_count[name] += 1

                # Check for duplicates across sets
                for other in seen_hashes:
                    if other != name and sample_hash in seen_hashes[other]:
                        pair = tuple(sorted((name, other)))  # consistent ordering
                        cross_set_dupes_count[pair] += 1
                        break  # avoid double-counting

                seen_hashes[name].add(sample_hash)

    # Report row counts
    for name, count in row_counts.items():
        print(f"{name.capitalize()} set: {count} samples")

    # Report duplicate summaries
    for name, count in within_set_dupes_count.items():
        warnings.warn(f"{count} duplicate samples found within the {name} set (fold {fold_num}).")

    for (name1, name2), count in cross_set_dupes_count.items():
        warnings.warn(f"{count} duplicate samples found between {name1} and {name2} sets (fold {fold_num}).")

    if verbosity > 0 and not within_set_dupes_count and not cross_set_dupes_count:
        print(f"\nNo duplicates found across splits for fold {fold_num} ✅\n")

    fold_stats_dict = {"train_score": None, "valid_score": None, "test_score": None}
    other_stats_dict = {
        "row_counts": dict(row_counts),
        "duplicates_within_sets": dict(within_set_dupes_count),
        "duplicates_across_sets": {f"{a}_{b}": count for (a, b), count in cross_set_dupes_count.items()}
    }

    return fold_stats_dict, other_stats_dict







def override_dict(target_dict, override_dict):
    for key in target_dict:
        if key in override_dict:
            target_dict[key] = override_dict[key]
    # return target_dict


def run_leakage_check(cli_args=None, hyperparam_csv=None, overrides={}):

    # device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    hpbuilder, data_param_cat, config_param_cat = hyperparams.construct_hyperparam_composer(hyperparam_csv=hyperparam_csv, cli_args=cli_args)

    loss_fn, score_fn, distr_error_fn = loss_function.get_loss_functions()

    # loop through possible combinations of dconfig hyperparams, though if we aren't in CSV mode there should only be one configuration
    for cp in hpbuilder.parse_and_generate_combinations(category=config_param_cat): 
        override_dict(cp, overrides)

        expt.process_config_params(cp)

        # loop through possible combinations of dataset hyperparams
        for dp in hpbuilder.parse_and_generate_combinations(category=data_param_cat):
            override_dict(dp, overrides)

            data_folded, testdata, dense_columns, sparse_columns = expt.process_data_params(dp)

            # benchmark_losses = expt.test_identity_model(dp, data_folded, device, loss_fn, score_fn, distr_error_fn)
            expt.fit_and_crossvalidate(
                fit_and_evaluate_fn=check_leakage, 
                data_folded=data_folded, data_test=testdata, 
                score_fn=score_fn,
                whichfold=dp.whichfold, 
                filepath_out_expt="results/leakage.csv",
                filepath_out_fold="results/leakage_fold.csv",
                out_rowinfo_dict={
                    "dataset": dp.y_dataset, 
                    "kfolds": dp.kfolds, 
                    "config_configid": cp.config_configid, 
                    "dataset_configid": dp.data_configid
                }
            )

    print("\n\nDONE")


# main
if __name__ == "__main__":
    # default values for overrides
    overrides = {
        } 
    run_leakage_check(cli_args=sys.argv[1:], hyperparam_csv="batch/1kLow/HPResults_baseline-SLPMultSoftmax.csv", overrides=overrides) # capture command line arguments, needs to be done explicitly so that when run_experiments is called from other contexts, CLI args aren't accidentally intercepted 
