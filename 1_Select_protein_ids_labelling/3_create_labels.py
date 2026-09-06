#!/usr/bin/env python3
"""
create_labels.py
================
Add a 'label' column to an ids CSV file.
 
    label: 1 = homodimer
           0 = monomer or heterodimer
 
Input columns : id, chain, stoichiometry
Output columns: id, chain, stoichiometry, label
 
Usage:
    python create_labels.py \
        --input  ids_after_training_cutoff.csv \
        --output ids_after_training_cutoff_labelled.csv
"""
 
import argparse
from pathlib import Path
import pandas as pd
 
 
def main():
    ap = argparse.ArgumentParser(
        description="Add label column (1=homodimer, 0=else) to ids CSV."
    )
    ap.add_argument("--input",  type=Path, required=True,
                    help="ids_after/before_training_cutoff.csv")
    ap.add_argument("--output", type=Path, required=True,
                    help="Output CSV with added label column")
    args = ap.parse_args()
 
    df = pd.read_csv(args.input)
    df["label"] = (df["stoichiometry"] == "homodimer").astype(int)
 
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False, keep_default_na=False)
 
    print(f"Input  : {args.input}  ({len(df)} rows)")
    print(f"Output : {args.output}")
    print(f"\nLabel distribution:")
    print(df["label"].value_counts()
          .rename({1: "1  (homodimer)", 0: "0  (negative)"})
          .to_string())
    print(f"\nStoichiometry distribution:")
    print(df["stoichiometry"].value_counts().to_string())
 
 
if __name__ == "__main__":
    main()