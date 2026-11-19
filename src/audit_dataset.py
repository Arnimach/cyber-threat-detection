import pandas as pd
from pathlib import Path
from collections import Counter
from IPython.display import display

DATA_DIR = Path("data/raw/cicids2017")
CHUNK_SIZE = 100_000


def find_label_column(file):
    """Find the Label column in the CSV header."""

    header = pd.read_csv(file, nrows=0)

    for column in header.columns:
        if column.strip().lower() == "label":
            return column

    raise ValueError(f"No Label column found in {file.name}")


def audit_file(file):
    """Audit one CSV file in chunks."""

    label_column = find_label_column(file)

    label_counts = Counter()
    total_rows = 0

    for chunk in pd.read_csv(
        file,
        usecols=[label_column],
        chunksize=CHUNK_SIZE
    ):

        labels = chunk[label_column].astype("string").str.strip()

        label_counts.update(labels)

        total_rows += len(labels)

    benign_count = label_counts.get("BENIGN", 0)

    attack_count = total_rows - benign_count

    attack_rate = attack_count / total_rows if total_rows else 0

    return {
        "file": file.name,
        "rows": total_rows,
        "benign": benign_count,
        "attack": attack_count,
        "attack_rate": attack_rate,
        "unique_labels": len(label_counts),
        "label_counts": dict(label_counts)
    }


def main():

    csv_files = list(DATA_DIR.glob("*.csv"))

    print(f"Found {len(csv_files)} CSV files.\n")

    results = []

    for file in csv_files:

        print(f"Auditing: {file.name}")

        result = audit_file(file)

        results.append(result)

    summary_df = pd.DataFrame(results)

    print("\n===== AUDIT SUMMARY =====\n")

    print(display(summary_df))

    output_file = "data/raw/audit_summary.csv"

    summary_df.to_csv(output_file)

    print(f"\nSaved audit summary to: {output_file}")


if __name__ == "__main__":
    main()
