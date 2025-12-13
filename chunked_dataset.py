import os
import random
import numpy as np
import torch
from torch.utils.data import IterableDataset
import pandas as pd
import glob


DK_BINARY = "binary"
DK_IDS = "ids-sparse"
DK_X = "x0"
DK_XSPARSE = "x0-sparse"
DK_Y = "y"
DK_YSPARSE = "y-sparse"


import os
import random
import numpy as np
import torch
from torch.utils.data import IterableDataset


class ChunkedCSVDataset(IterableDataset):
    def __init__(self, 
                 data_dir, 
                 file_types,  # e.g., {'x': 'x0', 'x_sparse': 'x0-sparse', 'y': 'random-1-gLV_y'}
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
        self.dirichlet_alpha = 0.0 # No Dirichlet noise by default, can be set externally

        # Which key gets Dirichlet-perturbed x values
        # Adjust this if your x key is named differently.
        self.dirichlet_x_keys = ["x0", "x0-sparse"]

        # Use first key as reference for row counts
        self.ref_key = next(iter(self.file_types))

        # Map key -> list of .npy chunk files
        self.chunk_files = self._organize_chunks()

        # Per-chunk row counts (for ref_key) and per-key column counts
        self.chunk_sizes = self._get_chunk_sizes()
        self.total_size = sum(self.chunk_sizes)

        # torch dtypes AND associated numpy dtypes (for casting)
        self.dtypes, self.np_dtypes = self._infer_dtypes()

        # How many samples we actually want to use
        self.data_samples = (
            data_train_samples + data_validation_samples
            if data_train_samples > 0 else self.total_size
        )

        if self.data_samples > self.total_size:
            print(
                f"Warning: Requested data_samples={self.data_samples} "
                f"exceeds total dataset size of {self.total_size}."
            )
            self.data_samples = self.total_size

        # Build global index list, then split into train/val or folds
        self._split_indices()
        self._init_epoch()

    # -----------------------
    # File discovery
    # -----------------------

    def _is_chunk_file(self, filename, ftype):
        """
        Check if filename matches the pattern:
        <prefix>_<integer>.npy

        where prefix is derived from ftype (last path component).
        """
        basename = os.path.basename(filename)
        prefix = ftype.split('/')[-1]

        if not (basename.startswith(prefix + '_') and basename.endswith('.npy')):
            return False

        # strip prefix_ and .npy and ensure numeric suffix
        suffix = basename[len(prefix) + 1 : -4]
        return suffix.isdigit()

    def _organize_chunks(self):
        """
        Walk data_dir and collect .npy files per key in file_types,
        sorted by their numeric suffix.
        """
        chunk_map = {}
        for key, ftype in self.file_types.items():
            files = []
            for root, _, filenames in os.walk(self.data_dir):
                for f in filenames:
                    if self._is_chunk_file(f, ftype):
                        files.append(os.path.join(root, f))

            if not files:
                raise ValueError(f"No .npy chunk files found for key '{key}' in {self.data_dir}")

            files = sorted(
                files,
                key=lambda f: int(os.path.basename(f).split('_')[-1].split('.')[0])
            )
            chunk_map[key] = files

        # sanity check: all keys should have same number of chunks
        num_chunks = {key: len(v) for key, v in chunk_map.items()}
        if len(set(num_chunks.values())) != 1:
            raise ValueError(f"Inconsistent number of chunks across keys: {num_chunks}")

        return chunk_map

    # -----------------------
    # Dtype inference
    # -----------------------

    def _infer_dtypes(self):
        """
        Infer torch and numpy dtypes from the first .npy chunk per key.
        """
        torch_dtypes = {}
        np_dtypes = {}

        for key, files in self.chunk_files.items():
            first_npy = files[0]
            arr = np.load(first_npy, mmap_mode='r')
            inferred_dtype = arr.dtype

            if np.issubdtype(inferred_dtype, np.integer):
                torch_dtypes[key] = torch.int64
                np_dtypes[key] = np.int64
            elif np.issubdtype(inferred_dtype, np.floating):
                # normalize to float32 on the torch side
                torch_dtypes[key] = torch.float32
                np_dtypes[key] = np.float32
            else:
                raise ValueError(
                    f"Unsupported data type {inferred_dtype} in file type {key}"
                )

        return torch_dtypes, np_dtypes

    # -----------------------
    # Sizes & indices
    # -----------------------

    def _get_chunk_sizes(self):
        """
        - Determine column counts per key.
        - Determine row count per chunk for the reference key.
        All from .npy files.
        """
        self.column_counts = {}

        # column counts per key (from first chunk)
        for key, files in self.chunk_files.items():
            first_npy = files[0]
            arr = np.load(first_npy, mmap_mode='r')
            if arr.ndim != 2:
                raise ValueError(f"Expected 2D arrays in {first_npy}, got shape {arr.shape}")
            self.column_counts[key] = arr.shape[1]

        # row counts for reference key
        ref_files = self.chunk_files[self.ref_key]
        sizes = []
        for file in ref_files:
            arr = np.load(file, mmap_mode='r')
            sizes.append(arr.shape[0])

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
                raise ValueError(
                    "kfolds must be greater than 1 if data_validation_samples <= 0"
                )

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
        self.chunk_to_indices = {
            chunk_idx: [] for chunk_idx in range(len(self.chunk_sizes))
        }
        for chunk_idx, row_idx in self.indices:
            self.chunk_to_indices[chunk_idx].append(row_idx)

        self.chunk_order = [
            c for c in self.chunk_to_indices if self.chunk_to_indices[c]
        ]

        if self.split == 'train':
            random.shuffle(self.chunk_order)
            for c in self.chunk_to_indices:
                random.shuffle(self.chunk_to_indices[c])

        self.current_chunk_idx = 0
        self.current_row_pointer = 0
        # lazily set self.current_chunk_data / self.loaded_chunk_id in __next__

    # -----------------------
    # Chunk loading (.npy only)
    # -----------------------

    def _load_chunk(self, chunk_idx):
        """
        Load one chunk for all keys as numpy arrays from .npy only.
        """
        chunk_data = {}
        for key, files in self.chunk_files.items():
            npy_path = files[chunk_idx]
            arr = np.load(npy_path)

            # ensure dtype matches our desired numpy dtype (may cast from float64->float32)
            desired_np_dtype = self.np_dtypes[key]
            if arr.dtype != desired_np_dtype:
                arr = arr.astype(desired_np_dtype, copy=False)

            if arr.ndim != 2:
                raise ValueError(f"Expected 2D array in {npy_path}, got shape {arr.shape}")

            chunk_data[key] = arr

        return chunk_data

    # -----------------------
    # Dirichlet helper
    # -----------------------

    def _apply_dirichlet_to_x_batch(self, batch_np):
        """
        Apply symmetric Dirichlet(alpha) to each row of x on its active support.

        - batch_np: (B, D) numpy array (float)
        - Zeros stay zero.
        - For each row, we take indices where x != 0, draw Dirichlet over those
          positions, and write the resulting composition back into those slots.
        """
        # print("DEBUG: DIRICHLET VALUE ", self.dirichlet_alpha)
        if self.dirichlet_alpha <= 0:
            return batch_np

        # Work on a copy so we never mutate the underlying chunk data
        x = batch_np.copy()
        mask = x != 0

        B, D = x.shape
        for i in range(B):
            active_idx = np.nonzero(mask[i])[0]
            k = active_idx.size
            if k == 0:
                continue  # all zeros, nothing to do

            alpha_vec = np.full(k, self.dirichlet_alpha, dtype=np.float64)
            # Dirichlet returns a vector summing to 1
            sample = np.random.dirichlet(alpha_vec)
            x[i, active_idx] = sample.astype(x.dtype, copy=False)

        # # Debug: print L1 loss between original and perturbed, per sample
        # diff = np.abs(x - batch_np)
        # l1_losses = diff.sum(axis=1).mean()
        # print("DEBUG: DIRICHLET L1 losses per sample: ", l1_losses)

        return x

    # -----------------------
    # IterableDataset API
    # -----------------------

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

            # Load chunk if not already loaded / changed
            if not hasattr(self, 'current_chunk_data') or self.loaded_chunk_id != chunk_id:
                self.current_chunk_data = self._load_chunk(chunk_id)
                self.loaded_chunk_id = chunk_id

            # Determine batch slice once
            start = self.current_row_pointer
            end = min(start + self.batch_size, len(sample_indices))
            batch_indices = sample_indices[start:end]
            self.current_row_pointer = end

            if len(batch_indices) == 0:
                continue

            batch_indices = np.asarray(batch_indices, dtype=np.int64)

            # Vectorized indexing for all keys
            batch_tensors = {}
            for key, arr in self.current_chunk_data.items():
                batch_np = arr[batch_indices]     # shape (B, D)

                # Apply Dirichlet to x only (float)
                if (
                    self.dirichlet_alpha > 0
                    and key in self.dirichlet_x_keys
                    and np.issubdtype(batch_np.dtype, np.floating)
                ):
                    batch_np = self._apply_dirichlet_to_x_batch(batch_np)

                tensor = torch.from_numpy(batch_np)
                if tensor.dtype != self.dtypes[key]:
                    tensor = tensor.to(self.dtypes[key])

                batch_tensors[key] = tensor

            return batch_tensors

        raise StopIteration

    # -----------------------
    # Optional: streaming by chunk
    # -----------------------

    def stream_by_chunk(self, device=None):
        for chunk_idx in self.chunk_order:
            row_indices = self.chunk_to_indices[chunk_idx]
            if not row_indices:
                continue  # skip empty chunks

            chunk_data = self._load_chunk(chunk_idx)
            row_indices = np.asarray(row_indices, dtype=np.int64)
            filtered_chunk = {}

            for key, arr in chunk_data.items():
                rows = arr[row_indices]

                if (
                    self.dirichlet_alpha > 0
                    and key in self.dirichlet_x_keys
                    and np.issubdtype(rows.dtype, np.floating)
                ):
                    rows = self._apply_dirichlet_to_x_batch(rows)

                tensor = torch.from_numpy(rows)
                if tensor.dtype != self.dtypes[key]:
                    tensor = tensor.to(self.dtypes[key])
                if device is not None:
                    tensor = tensor.to(device)
                filtered_chunk[key] = tensor

            yield filtered_chunk


class TestCSVDataset(IterableDataset):
    """
    Test dataset that now loads .npy test files only.

    Expects one .npy per key, matching pattern:
        <data_dir>/**/<ftype>_test.npy

    All arrays must have the same number of rows.
    """
    def __init__(self, data_dir, file_types, batch_size=64):
        super().__init__()
        self.data_dir = data_dir
        self.file_types = file_types
        self.batch_size = batch_size

        # Load all test arrays
        self.arrays = {}
        for key, ftype in file_types.items():
            pattern = os.path.join(self.data_dir, "**", f"{ftype}_test.npy")
            matches = glob.glob(pattern, recursive=True)
            if not matches:
                raise FileNotFoundError(
                    f"Test file for '{key}' with pattern '{pattern}' not found."
                )
            if len(matches) > 1:
                raise ValueError(
                    f"Multiple test files found for '{key}': {matches}"
                )

            npy_path = matches[0]
            arr = np.load(npy_path)

            if arr.ndim == 1:
                # promote 1D to (N, 1) if needed
                arr = arr[:, None]
            elif arr.ndim != 2:
                raise ValueError(
                    f"Expected 1D or 2D array in '{npy_path}', got shape {arr.shape}"
                )

            self.arrays[key] = arr

        # Check row counts match across all keys
        lengths = [arr.shape[0] for arr in self.arrays.values()]
        if not all(length == lengths[0] for length in lengths):
            raise ValueError("Test .npy files have mismatched number of rows!")
        self.total_samples = lengths[0]

        # Infer torch dtypes from numpy dtypes
        self.dtypes = {}
        for key, arr in self.arrays.items():
            inferred_dtype = arr.dtype
            if np.issubdtype(inferred_dtype, np.integer):
                self.dtypes[key] = torch.int64
            elif np.issubdtype(inferred_dtype, np.floating):
                # normalize to float32 on the torch side
                self.dtypes[key] = torch.float32
            else:
                raise ValueError(
                    f"Unsupported data type {inferred_dtype} in test file type '{key}'"
                )

    def __iter__(self):
        self.current_idx = 0
        return self

    def __next__(self):
        if self.current_idx >= self.total_samples:
            raise StopIteration

        end_idx = min(self.current_idx + self.batch_size, self.total_samples)
        idx_slice = slice(self.current_idx, end_idx)

        batch = {}
        for key, arr in self.arrays.items():
            batch_np = arr[idx_slice]  # (B, D)
            tensor = torch.from_numpy(batch_np)
            if tensor.dtype != self.dtypes[key]:
                tensor = tensor.to(self.dtypes[key])
            batch[key] = tensor

        self.current_idx = end_idx
        return batch

    def stream_by_chunk(self, device=None):
        """Yields the full test data as one dictionary-style chunk."""
        chunk = {}
        for key, arr in self.arrays.items():
            tensor = torch.from_numpy(arr)
            if tensor.dtype != self.dtypes[key]:
                tensor = tensor.to(self.dtypes[key])
            if device is not None:
                tensor = tensor.to(device)
            chunk[key] = tensor

        yield chunk



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

    def dataset_summary(train_set, val_set, non_sparse_key, sparse_key):

        num_train_samples = len(train_set.indices)
        
        assert non_sparse_key in train_set.column_counts, f"Key '{non_sparse_key}' not found in dataset."
        assert sparse_key in train_set.column_counts, f"Key '{sparse_key}' not found in dataset."
        
        dense_columns = train_set.column_counts[non_sparse_key]
        sparse_columns = train_set.column_counts[sparse_key]

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
            proportion = 0.0
            print("Proportion of non-validation data used: N/A (no non-val samples)")

        # Feature counts
        print(f"{non_sparse_key} columns: {dense_columns}")
        print(f"{sparse_key} columns: {sparse_columns}")

        return num_train_samples, proportion, dense_columns, sparse_columns

    # Hold-out validation
    if data_validation_samples > 0:
        train_set = ChunkedCSVDataset(
            data_dir,
            file_types,
            split='train',
            data_train_samples=data_train_samples,
            data_validation_samples=data_validation_samples,
            batch_size=batch_size,
        )

        val_set = ChunkedCSVDataset(
            data_dir,
            file_types,
            split='val',
            data_train_samples=data_train_samples,
            data_validation_samples=data_validation_samples,
            batch_size=batch_size, 
        )

        folds.append((train_set, val_set))

        num_train_samples, train_proportion, dense_columns, sparse_columns = dataset_summary(train_set, val_set, non_sparse_key, sparse_key)

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
            batch_size=batch_size,
        )

        first_val_set = ChunkedCSVDataset(
            data_dir,
            file_types,
            split='val',
            data_train_samples=data_train_samples,
            kfolds=kfolds,
            current_fold=0,
            batch_size=batch_size,
        )


        num_train_samples, train_proportion, dense_columns, sparse_columns = dataset_summary(first_train_set, first_val_set, non_sparse_key, sparse_key)

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
                batch_size=batch_size,
            )

            val_set = ChunkedCSVDataset(
                data_dir,
                file_types,
                split='val',
                data_train_samples=data_train_samples,
                kfolds=kfolds,
                current_fold=fold_idx,
                batch_size=batch_size,
            )

            folds.append((train_set, val_set))

    return folds, num_train_samples, train_proportion, dense_columns, sparse_columns


