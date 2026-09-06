#!/usr/bin/env python3

import csv
import argparse


def parse_centerpoint(centerpoint: str):
    """
    Examples:
      12as-assembly1_B -> ("12as", "B")
      155c-assembly1   -> ("155c", None)
    """
    centerpoint = centerpoint.strip()

    # Extract chain if present
    if "_" in centerpoint:
        base, chain = centerpoint.rsplit("_", 1)
    else:
        base, chain = centerpoint, None

    # Remove "-assembly..." part
    if "-assembly" in base:
        pdb_id = base.split("-assembly", 1)[0]
    else:
        pdb_id = base

    return pdb_id, chain


def convert_csv(input_file, output_file):
    seen = set()
    output_rows = []

    with open(input_file, "r", newline="") as f:
        reader = csv.DictReader(f)

        if "centerpoint" not in reader.fieldnames or "category" not in reader.fieldnames:
            raise ValueError("Input CSV must contain 'centerpoint' and 'category' columns")

        for row in reader:
            centerpoint = row["centerpoint"]
            stoichiometry = row["category"]

            pdb_id, parsed_chain = parse_centerpoint(centerpoint)

            if stoichiometry == "heterodimer":
                chain = parsed_chain if parsed_chain else "N/A"
            else:
                chain = "N/A"

            new_row = (pdb_id, chain, stoichiometry)

            # Avoid duplicates
            if new_row not in seen:
                seen.add(new_row)
                output_rows.append(new_row)

    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "chain", "stoichiometry"])
        writer.writerows(output_rows)


def main():
    parser = argparse.ArgumentParser(description="Convert CSV to id, chain, stoichiometry format")
    parser.add_argument("input_csv", help="Input CSV file")
    parser.add_argument("output_csv", help="Output CSV file")

    args = parser.parse_args()

    convert_csv(args.input_csv, args.output_csv)


if __name__ == "__main__":
    main()