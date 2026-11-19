import pandas as pd
from pathlib import Path
from collections import Counter


# --------------------------------------------------
# Paths
# --------------------------------------------------

DATA_DIR = Path("data/processed")

SPLITS = {
    "train": DATA_DIR / "train",
    "validation": DATA_DIR / "validation",
    "test": DATA_DIR / "test",
}

CHUNK_SIZE = 100_000


# --------------------------------------------------
# Validate one split
# --------------------------------------------------

def validate_split(split_name, split_dir):

    print("\n" + "=" * 60)
    print(f"{split_name.upper()} DATA")
    print("=" * 60)

    csv_files = list(split_dir.glob("*.csv"))

    if not csv_files:
        print(f"No CSV files found in {split_dir}")
        return

    total_rows = 0
    target_counts = Counter()
    missing_counts = Counter()

    for file in csv_files:

        print(f"\nChecking: {file.name}")

        file_rows = 0
        file_target_counts = Counter()

        for chunk in pd.read_csv(
            file,
            chunksize=CHUNK_SIZE
        ):

            # -------------------------------
            # Row count
            # -------------------------------

            file_rows += len(chunk)
            total_rows += len(chunk)

            # -------------------------------
            # Target distribution
            # -------------------------------

            counts = chunk["target"].value_counts()

            for target, count in counts.items():
                target_counts[target] += count
                file_target_counts[target] += count

            # -------------------------------
            # Missing values
            # -------------------------------

            missing = chunk.isna().sum()

            for column, count in missing.items():

                if count > 0:
                    missing_counts[column] += count

        print(f"Rows: {file_rows:,}")
        print(f"Target distribution: {dict(file_target_counts)}")

    # --------------------------------------------------
    # Summary
    # --------------------------------------------------

    print("\n--- SUMMARY ---")

    print(f"Total rows: {total_rows:,}")
    print("\nTarget distribution:")

    benign = target_counts.get(0, 0)
    attack = target_counts.get(1, 0)

    print(f"  Benign (0): {benign:,}")
    print(f"  Attack (1): {attack:,}")

    if total_rows > 0:
        print(f"  Attack rate: {attack / total_rows:.2%}")

    # --------------------------------------------------
    # Missing values
    # --------------------------------------------------

    print("\nMissing values:")

    if missing_counts:
        for column, count in missing_counts.items():
            print(f"  {column}: {count:,}")
    else:
        print("  No missing values found.")



def get_dtype_category(dtype):
    if pd.api.types.is_numeric_dtype(dtype):
        return 'numeric'
    return 'non_numeric'


def validate_dtypes(files):
    reference_file = files[0]
    reference_df = pd.read_csv(reference_file, nrows=5)

    reference_categories = {
        column: get_dtype_category(dtype)
        for column, dtype in reference_df.dtypes.items()
    }

    issues_found = False

    for file in files[1:]:
        current_df = pd.read_csv(file, nrows=5)

        for column in reference_df.columns:
            reference_category = reference_categories[column]
            current_category = get_dtype_category(current_df[column].dtype)

            if not reference_category == current_category:
                issues_found = True
                print("\nDatatype mismatch:")
                print(f"  Column: {column}")
                print(f"  Reference file: {reference_file.name}")
                print(f"  Reference category: {reference_category}")
                print(f"  Current file: {file.name}")
                print(f"  Current category: {current_category}")

    if not issues_found:
        print("\n\n Datatype consistency check passed ✓")

    return not issues_found


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    for split_name, split_dir in SPLITS.items():

        validate_split(
            split_name,
            split_dir
        )

    all_files = []
    for split_dir in SPLITS.values():
        all_files.extend(split_dir.glob("*.csv"))

    validate_dtypes(all_files)

    print("\n" + "=" * 60)
    print("VALIDATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()