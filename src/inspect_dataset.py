from pathlib import Path
import pandas as pd


DATA_DIR = Path("data/raw/cicids2017")


def main() -> None:
    files = list(DATA_DIR.glob("*.csv"))

    if not files:
        print(f"No CSV files found in {DATA_DIR}")
        return

    print(f"Found {len(files)} CSV files\n")

    reference_columns = None

    for file in files:
        print("=" * 80)
        print(f"FILE: {file.name}")

        # Read only a small sample first
        sample = pd.read_csv(file, nrows=5)

        print(f"Number of columns: {len(sample.columns)}")
        print("Columns:")
        print(list(sample.columns))

        # Check schema consistency
        columns = list(sample.columns)

        if reference_columns is None:
            reference_columns = columns
            print("\nThis file establishes the reference schema.")
        elif columns == reference_columns:
            print("\nSchema: SAME as first file")
        else:
            print("\nSchema: DIFFERENT from first file")

        # Look for a likely label column
        label_candidates = [
            col for col in sample.columns
            if "label" in col.lower()
        ]

        print(f"Possible label columns: {label_candidates}")

        print()


if __name__ == "__main__":
    main()