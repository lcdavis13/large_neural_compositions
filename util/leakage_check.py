import os
import csv
import hashlib

def hash_row(row):
    """Generate a SHA256 hash for a CSV row (as a list of strings)."""
    row_string = ','.join(row).strip()
    return hashlib.sha256(row_string.encode('utf-8')).hexdigest()

def check_duplicates_in_csv_folder(folder_path, filename_prefix):
    seen_hashes = set()
    duplicate_rows = []

    for filename in os.listdir(folder_path):
        if filename.startswith(filename_prefix) and filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            print(f"Processing {filename}...")
            with open(file_path, 'r', newline='', encoding='utf-8') as f:
                reader = csv.reader(f)
                file_hashes = set()
                for row in reader:
                    row_hash = hash_row(row)
                    if row_hash in file_hashes:
                        duplicate_rows.append((filename, "within-file", row))
                    elif row_hash in seen_hashes:
                        duplicate_rows.append((filename, "cross-file", row))
                    else:
                        file_hashes.add(row_hash)
                        seen_hashes.add(row_hash)

    if duplicate_rows:
        print("\nDuplicate rows found:")
        for dup in duplicate_rows:
            print(f"{dup[0]} - {dup[1]} duplicate: {dup[2]}")
    else:
        print("\nNo duplicates found.")


if __name__ == "__main__":
    folder_path = "data/256/"
    filename_prefix = "256_binary_"
    check_duplicates_in_csv_folder(folder_path, filename_prefix)

    folder_path = "data/256/"
    filename_prefix = "256_ids-sparse_"
    check_duplicates_in_csv_folder(folder_path, filename_prefix)