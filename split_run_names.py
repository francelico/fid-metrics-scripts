#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path


def write_names(input_csv: Path, output_csv: Path, pattern: str, prefix: str) -> int:
    """
    Read the input CSV, keep only rows whose Name contains `pattern`,
    prepend `prefix` to the Name value, and write a one-column CSV.

    Returns the number of rows written.
    """
    count = 0

    with input_csv.open("r", newline="", encoding="utf-8") as fin, \
         output_csv.open("w", newline="", encoding="utf-8") as fout:

        reader = csv.DictReader(fin)

        if "Name" not in reader.fieldnames:
            raise ValueError("Input CSV does not contain a 'Name' column.")

        writer = csv.writer(fout)
        writer.writerow(["Name"])

        for row in reader:
            name = row.get("Name", "")
            if pattern in name:
                writer.writerow([f"{prefix}{name}"])
                count += 1

    return count


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Split the Name column of a CSV into two files based on substring patterns, "
            "and prepend a prefix to each matched Name."
        )
    )
    parser.add_argument("input_csv", help="Path to input CSV file")
    parser.add_argument("--pattern_a", required=True, help="Substring for file_a.csv")
    parser.add_argument("--pattern_b", required=True, help="Substring for file_b.csv")
    parser.add_argument("--prefix", default="", help="String to prepend to each Name")
    parser.add_argument("--output_a", default="file_a.csv", help="Output CSV for pattern_a")
    parser.add_argument("--output_b", default="file_b.csv", help="Output CSV for pattern_b")

    args = parser.parse_args()

    input_csv = Path(args.input_csv)
    output_a = Path(args.output_a)
    output_b = Path(args.output_b)

    count_a = write_names(input_csv, output_a, args.pattern_a, args.prefix)
    count_b = write_names(input_csv, output_b, args.pattern_b, args.prefix)

    print(f"Wrote {count_a} rows to {output_a}")
    print(f"Wrote {count_b} rows to {output_b}")


if __name__ == "__main__":
    main()