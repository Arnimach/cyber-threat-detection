import pandas as pd
from pathlib import Path

# --------------------------------------------------
# Paths
# --------------------------------------------------

RAW_DIR = Path("data/raw/cicids2017")
PROCESSED_DIR = Path("data/processed")

TRAIN_DIR = PROCESSED_DIR / "train"
VAL_DIR = PROCESSED_DIR / "validation"
TEST_DIR = PROCESSED_DIR / "test"

CHUNK_SIZE = 100_000

# --------------------------------------------------
# CICIDS2017 chronological file order
# --------------------------------------------------

TRAIN_FILES = [
    "Monday-WorkingHours.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv",
]

VALIDATION_FILES = [
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
]

TEST_FILES = [
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
]


# --------------------------------------------------
# Create output directories
# --------------------------------------------------

for directory in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# Process one file
# --------------------------------------------------

def process_file(input_file, output_file):
    """
    Read one CICIDS2017 CSV in chunks,
    convert Label to binary target,
    and write the processed data.
    """

    print(f"\nProcessing: {input_file.name}")

    first_chunk = True
    total_rows = 0

    for chunk in pd.read_csv(
        input_file,
        chunksize=CHUNK_SIZE
    ):

        # Clean column names
        chunk.columns = chunk.columns.str.strip()

        # Clean label values
        chunk["Label"] = (
            chunk["Label"]
            .astype("string")
            .str.strip()
        )

        # Binary target:
        # BENIGN = 0
        # Anything else = 1
        chunk["target"] = (
            chunk["Label"]
            .ne("BENIGN")
            .astype("int8")
        )

        # Write first chunk with header
        # Append subsequent chunks without header
        chunk.to_csv(
            output_file,
            mode="w" if first_chunk else "a",
            header=first_chunk,
            index=False
        )

        first_chunk = False
        total_rows += len(chunk)

        print(f"  Processed {total_rows:,} rows")

    print(f"Saved: {output_file}")
    print(f"Total rows: {total_rows:,}")


# --------------------------------------------------
# Process a group of files
# --------------------------------------------------

def process_group(files, output_dir, group_name):

    print("\n" + "=" * 60)
    print(f"{group_name.upper()} DATA")
    print("=" * 60)

    for filename in files:

        input_file = RAW_DIR / filename

        if not input_file.exists():
            raise FileNotFoundError(
                f"File not found: {input_file}"
            )

        output_file = output_dir / filename

        process_file(
            input_file,
            output_file
        )


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    process_group(
        TRAIN_FILES,
        TRAIN_DIR,
        "training"
    )

    process_group(
        VALIDATION_FILES,
        VAL_DIR,
        "validation"
    )

    process_group(
        TEST_FILES,
        TEST_DIR,
        "test"
    )

    print("\n" + "=" * 60)
    print("INGESTION COMPLETE")
    print("=" * 60)

    print(f"\nTraining data:   {TRAIN_DIR}")
    print(f"Validation data: {VAL_DIR}")
    print(f"Test data:       {TEST_DIR}")


if __name__ == "__main__":
    main()