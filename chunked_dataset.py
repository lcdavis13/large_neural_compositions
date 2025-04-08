import os
import random
import numpy as np
import torch
from torch.utils.data import IterableDataset
import pandas as pd

DK_BINARY = "binary"
DK_IDS = "ids-sparse"
DK_X = "x0"
DK_XSPARSE = "x0-sparse"
DK_Y = "y"
DK_YSPARSE = "y-sparse"

class ChunkedCSVDataset(IterableDataset):
    def __init__(self, 
                 data_dir, 
                 file_types,  # e.g., ['x', 'y']
                 split='train', 
                 batch_size=32,
                 data_validation_samples=0,
                 data_train_samples=0,
                 kfolds=0,
                 current_fold=0):
        super().__init__()
        self.data_dir = data_dir
        self.file_types = file_types
        self.split = split
        self.batch_size = batch_size
        self.data_validation_samples = data_validation_samples
        self.kfolds = kfolds
        self.current_fold = current_fold
        self.ref_key = next(iter(self.file_types))

        self.chunk_files = self._organize_chunks()
        self.chunk_sizes = self._get_chunk_sizes()
        self.total_size = sum(self.chunk_sizes)

        self.dtypes = self._infer_dtypes()

        self.data_samples = data_train_samples + data_validation_samples if data_train_samples > 0 else self.total_size

        if self.data_samples > self.total_size:
            # warning
            print(f"Warning: Requested data_samples={self.data_samples} exceeds total dataset size of {self.total_size}.")
            self.data_samples = self.total_size


        self._split_indices()
        self._init_epoch()

    def _infer_dtypes(self):
        dtypes = {}
        for key, files in self.chunk_files.items():
            sample_df = pd.read_csv(files[0], header=None, nrows=10)  # Read a small sample
            inferred_dtype = sample_df.dtypes[0]  # assume all columns have same dtype
            if np.issubdtype(inferred_dtype, np.integer):
                dtypes[key] = torch.int64
            elif np.issubdtype(inferred_dtype, np.floating):
                dtypes[key] = torch.float32
            else:
                raise ValueError(f"Unsupported data type {inferred_dtype} in file type {key}")
        return dtypes

    def _is_chunk_file(self, filename, ftype):
        if not (filename.startswith(ftype + '_') and filename.endswith('.csv')):
            return False
        suffix = filename[len(ftype) + 1 : -4]  # remove prefix and '.csv'
        return suffix.isdigit()

    def _organize_chunks(self):
        chunk_map = {}
        for key, ftype in self.file_types.items():
            files = [
                f for f in os.listdir(self.data_dir)
                if self._is_chunk_file(f, ftype)
            ]
            files = sorted(files, key=lambda f: int(f.split('_')[-1].split('.')[0]))
            chunk_map[key] = [os.path.join(self.data_dir, f) for f in files]
        return chunk_map

    def _get_chunk_sizes(self):
        ref_type = self.file_types[self.ref_key]
        sizes = []
        self.column_counts = {}  # New: store column counts per key

        for key, files in self.chunk_files.items():
            # Open only the first file per key to inspect
            df = pd.read_csv(files[0], header=None, nrows=1)
            self.column_counts[key] = df.shape[1]

        # Continue counting rows for the reference type
        for file in self.chunk_files[self.ref_key]:
            with open(file, 'r') as f:
                sizes.append(sum(1 for _ in f))  # No header assumed

        return sizes


    def _split_indices(self):
        # Collect indices only up to self.data_samples
        all_indices = []
        sample_count = 0

        for chunk_idx, size in enumerate(self.chunk_sizes):
            for row_idx in range(size):
                if sample_count >= self.data_samples:
                    break
                all_indices.append((chunk_idx, row_idx))
                sample_count += 1
            if sample_count >= self.data_samples:
                break

        if self.data_validation_samples > 0:
            # Hold-out method
            if self.split == 'train':
                self.indices = all_indices[self.data_validation_samples:self.data_samples]
            elif self.split == 'val':
                self.indices = all_indices[:self.data_validation_samples]
            else:
                raise ValueError("split must be 'train' or 'val'")
        else:
            # K-Fold cross-validation
            k = self.kfolds
            if k <= 1:
                raise ValueError("kfolds must be greater than 1 if data_validation_samples <= 0")

            fold_size = len(all_indices) // k
            fold_start = self.current_fold * fold_size
            fold_end = fold_start + fold_size

            val_indices = all_indices[fold_start:fold_end]
            train_indices = all_indices[:fold_start] + all_indices[fold_end:]

            if self.split == 'train':
                self.indices = train_indices
            elif self.split == 'val':
                self.indices = val_indices
            else:
                raise ValueError("split must be 'train' or 'val'")


    def _init_epoch(self):
        self.chunk_to_indices = {chunk_idx: [] for chunk_idx in range(len(self.chunk_sizes))}
        for chunk_idx, row_idx in self.indices:
            self.chunk_to_indices[chunk_idx].append(row_idx)

        self.chunk_order = [c for c in self.chunk_to_indices if self.chunk_to_indices[c]]
        if self.split == 'train': # TODO: There was some bug that caused a freeze when shuffling validation. We don't want to shuffle validation, but it's still concerning... EDIT: well the bug just resurfaced. Might have been a coincidence that it didn't occur right after making this change. But it's weird because it was the only time it ever got past 2.5 miniepochs, and it made it all the way to 70 something
            # print(f"Shuffling {len(self.chunk_order)} chunks for training...")
            random.shuffle(self.chunk_order)

            for c in self.chunk_to_indices:
                random.shuffle(self.chunk_to_indices[c])

        self.current_chunk_idx = 0
        self.current_row_pointer = 0

    def _load_chunk(self, chunk_idx):
        chunk_data = {}
        for key, files in self.chunk_files.items():
            df = pd.read_csv(files[chunk_idx], header=None)
            chunk_data[key] = df
        return chunk_data


    def __iter__(self):
        self._init_epoch()
        return self

    def __next__(self):
        while self.current_chunk_idx < len(self.chunk_order):
            chunk_id = self.chunk_order[self.current_chunk_idx]
            sample_indices = self.chunk_to_indices[chunk_id]

            # Move to next chunk if current is exhausted
            if self.current_row_pointer >= len(sample_indices):
                self.current_chunk_idx += 1
                self.current_row_pointer = 0
                continue

            # Load chunk if not already loaded
            if not hasattr(self, 'current_chunk_data') or self.loaded_chunk_id != chunk_id:
                self.current_chunk_data = self._load_chunk(chunk_id)
                self.loaded_chunk_id = chunk_id

            # Prepare batch data dictionary (list of rows for each key)
            batch_rows = {key: [] for key in self.current_chunk_data}

            ref_key = next(iter(batch_rows))
            while len(batch_rows[ref_key]) < self.batch_size and self.current_row_pointer < len(sample_indices):
                row_idx = sample_indices[self.current_row_pointer]
                for key, df in self.current_chunk_data.items():
                    batch_rows[key].append(df.iloc[row_idx])
                self.current_row_pointer += 1

            # In rare case of empty batch (e.g., last chunk), skip
            if not any(batch_rows.values()):
                continue

            # Convert lists of rows to batched tensors
            batch_tensors = {
                key: torch.tensor(
                    np.array([row.values for row in rows]), 
                    dtype=self.dtypes[key]
                )
                for key, rows in batch_rows.items()
            }




            return batch_tensors

        raise StopIteration



class TestCSVDataset(IterableDataset):
    def __init__(self, data_dir, file_types, batch_size=64):
        super().__init__()
        self.data_dir = data_dir
        self.file_types = file_types
        self.batch_size = batch_size

        self.dataframes = {
            key: pd.read_csv(os.path.join(data_dir, f"{ftype}_test.csv"), header=None)
            for key, ftype in file_types.items()
        }

        lengths = [len(df) for df in self.dataframes.values()]
        if not all(length == lengths[0] for length in lengths):
            raise ValueError("Test CSV files have mismatched number of rows!")
        self.total_samples = lengths[0]

        self.dtypes = {}
        for key, df in self.dataframes.items():
            inferred_dtype = df.dtypes[0]
            if np.issubdtype(inferred_dtype, np.integer):
                self.dtypes[key] = torch.int64
            elif np.issubdtype(inferred_dtype, np.floating):
                self.dtypes[key] = torch.float32
            else:
                raise ValueError(f"Unsupported data type {inferred_dtype} in test file type {key}")



    def __iter__(self):
        self.current_idx = 0
        return self

    def __next__(self):
        if self.current_idx >= self.total_samples:
            raise StopIteration

        end_idx = min(self.current_idx + self.batch_size, self.total_samples)

        batch = {
            key: torch.tensor(
                df.iloc[self.current_idx:end_idx].values,
                dtype=self.dtypes[key]
            )
            for key, df in self.dataframes.items()
        }

        self.current_idx = end_idx
        return batch


def load_folded_datasets(
    data_dir,
    file_types,
    batch_size,
    data_train_samples=0,
    data_validation_samples=0,
    kfolds=0,
    non_sparse_key=DK_X,
    sparse_key=DK_XSPARSE,
):
    folds = []

    def print_dataset_summary(train_set, val_set, dense_columns, sparse_columns):
        total_available_samples = train_set.total_size
        num_train_samples = len(train_set.indices)
        num_val_samples = len(val_set.indices)

        print("\nDataset Summary:")
        print(f"Total available samples: {total_available_samples}")
        print(f"Training samples: {num_train_samples}")
        print(f"Validation samples: {num_val_samples}")

        non_val_samples = total_available_samples - num_val_samples
        if non_val_samples > 0:
            proportion = num_train_samples / non_val_samples
            print(f"Proportion of non-validation data used: {proportion:.4f}")
        else:
            print("Proportion of non-validation data used: N/A (no non-val samples)")

        # Feature counts
        print(f"{non_sparse_key} columns: {dense_columns}")
        print(f"{sparse_key} columns: {sparse_columns}")

    # Hold-out validation
    if data_validation_samples > 0:
        train_set = ChunkedCSVDataset(
            data_dir,
            file_types,
            split='train',
            data_train_samples=data_train_samples,
            data_validation_samples=data_validation_samples,
            batch_size=batch_size
        )

        val_set = ChunkedCSVDataset(
            data_dir,
            file_types,
            split='val',
            data_train_samples=data_train_samples,
            data_validation_samples=data_validation_samples,
            batch_size=batch_size
        )

        folds.append((train_set, val_set))


        assert non_sparse_key in train_set.column_counts, f"Key '{non_sparse_key}' not found in dataset."
        assert sparse_key in train_set.column_counts, f"Key '{sparse_key}' not found in dataset."
        dense_columns = train_set.column_counts[non_sparse_key]
        sparse_columns = train_set.column_counts[sparse_key]

        print_dataset_summary(train_set, val_set, dense_columns, sparse_columns)

    # K-Fold cross-validation
    else:
        if kfolds <= 1:
            raise ValueError("kfolds must be greater than 1 if data_validation_samples <= 0")

        # Prepare first fold to calculate stats
        first_train_set = ChunkedCSVDataset(
            data_dir,
            file_types,
            split='train',
            data_train_samples=data_train_samples,
            kfolds=kfolds,
            current_fold=0,
            batch_size=batch_size
        )

        first_val_set = ChunkedCSVDataset(
            data_dir,
            file_types,
            split='val',
            data_train_samples=data_train_samples,
            kfolds=kfolds,
            current_fold=0,
            batch_size=batch_size
        )

        total_available_samples = first_train_set.data_samples
        print_dataset_summary(first_train_set, first_val_set, total_available_samples)

        folds.append((first_train_set, first_val_set))

        # Generate remaining folds
        for fold_idx in range(1, kfolds):
            train_set = ChunkedCSVDataset(
                data_dir,
                file_types,
                split='train',
                data_train_samples=data_train_samples,
                kfolds=kfolds,
                current_fold=fold_idx,
                batch_size=batch_size
            )

            val_set = ChunkedCSVDataset(
                data_dir,
                file_types,
                split='val',
                data_train_samples=data_train_samples,
                kfolds=kfolds,
                current_fold=fold_idx,
                batch_size=batch_size
            )

            folds.append((train_set, val_set))

    return folds, dense_columns, sparse_columns


