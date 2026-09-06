#!/usr/bin/env python3
"""
Filter non-protein sequences out of a combined FASTA file.

The challenge: many protein names contain "DNA" or "RNA" (e.g. "DNA ligase",
"RNA polymerase"), so naive keyword matching produces many false positives.

This filter uses the RCSB header's molecule-name field (field 3) to distinguish:
  - nucleic acid molecules: "DNA (5'-D(...)...)", "tRNA", "crRNA", "28-MER", etc.
  - proteins that work on nucleic acids: "DNA ligase", "RNA polymerase", etc.

Two independent checks; either triggers removal:
  1. Molecule-name patterns that indicate the entry IS a nucleic acid.
  2. Sequence composition: >=90% nucleotide characters (A,T,G,C,U,N).

Outputs:
  --clean    FASTA with protein sequences only
  --removed  FASTA with the removed non-protein sequences

Usage:
  python filter_nonprotein.py \\
      --input       raw_sequences/sequences_before_training_cutoff.fasta \\
      --clean-dir   clean_sequences/ \\
      --removed-dir removed_sequences/
"""

from __future__ import annotations

import re
import sys
import argparse
from pathlib import Path


# ---------------------------------------------------------------------------
# Header-based detection
# ---------------------------------------------------------------------------

# Patterns in the molecule name that mean the entry IS a nucleic acid molecule.
# Applied to the 3rd pipe-delimited field of the RCSB FASTA header.
_NUCLEIC_MOLECULE_PATTERNS = re.compile(
    r"|".join([
        # Explicit nucleotide-strand notation
        r"5'-[DR]\(",                   # 5'-D(...) or 5'-R(...)
        r"5'-\*",                       # 5'-*...
        r"\b\d+'-D\(",                  # 3'-D(...), 5'-D(...)
        r"\b\d+'-R\(",                  # 3'-R(...), 5'-R(...)
        r"\(\*[ATGCU]P\*",             # (*AP*TP*...) nucleotide notation

        # Molecule name starts with "DNA" or "RNA" — almost always means
        # the entry IS a nucleic acid molecule, not a protein. Protein names
        # that contain "DNA"/"RNA" typically use them as adjectives in the
        # middle ("DNA ligase", "RNA polymerase"), not at the start.
        r"^DNA\b",
        r"^RNA\b",
        r"^5'-",
        r"^3'-",

        # Specific RNA molecule types (the molecule itself, not a protein)
        r"\btRNA\b",
        r"\brRNA\b",
        r"\bsnRNA\b",
        r"\bsnoRNA\b",
        r"\bmiRNA\b",
        r"\bsiRNA\b",
        r"\bcrRNA\b",
        r"\bsgRNA\b",
        r"\bgRNA\b",
        r"\bncRNA\b",
        r"\bpri-miRNA\b",
        r"\bmRNA antiterminator\b",

        # Descriptors that mean it's a nucleic acid molecule
        r"\b\d+-MER\b",                # "26-MER", "55-MER" — oligo descriptor
        r"\bDNA target strand\b",
        r"\bDNA non-target strand\b",
        r"\btarget strand\b",
        r"\bnon-target strand\b",
        r"\bnone transferred strand\b",
        r"\btransferred strand\b",
        r"\bDNA substrate\b",
        r"\bDNA duplex\b",
        r"\bDNA forward\b",
        r"\bDNA reverse\b",
        r"\bDNA motif\b",
        r"\bNTS-DNA\b",
        r"\bTS-DNA\b",
        r"^NTS$",
        r"^TS$",
        r"^NTS\b",
        r"\bpGpG\b",
        r"\bCENP-B BOX DNA\b",
        r"\btelomere single-strand\b",
        r"\bRNA hairpin\b",
        r"\bRNA pseudoknot\b",
        r"\bGCC-box motif\b",
        r"\banti-tag target RNA\b",
        r"\bDNA fragment\b",
        r"\boligonucleotide\b",
        r"\bDNA promoter\b",
        r"^peptide$",                   # short peptides like "GAAG"
    ]),
    re.IGNORECASE,
)

# Patterns that indicate it's a PROTEIN even if "DNA"/"RNA" appears in the name.
# If both nucleic and protein patterns match, keep as protein.
_PROTEIN_OVERRIDE_PATTERNS = re.compile(
    r"|".join([
        r"\bprotein\b",
        r"\bsynthetase\b",
        r"\bsynthase\b",
        r"\bkinase\b",
        r"\bligase\b",
        r"\bpolymerase\b",
        r"\bhelicase\b",
        r"transferase\b",              # suffix: isopentenyltransferase, etc.
        r"glycosylase\b",
        r"methyltransferase\b",
        r"\bendonuclease\b",
        r"\bexonuclease\b",
        r"\bexoribonuclease\b",
        r"\bnuclease\b",
        r"\bgyrase\b",
        r"\btopoisomerase\b",
        r"\bprimase\b",
        r"\brepair\b",
        r"\breplication\b",
        r"\bbinding\b",
        r"\bregulator\b",
        r"\bsigma\b",
        r"\bfactor\b",
        r"\bsubunit\b",
        r"\binhibitor\b",
        r"\bhydrolase\b",
        r"\bphosphodiesterase\b",
        r"\bdeaminase\b",
        r"\bchaperone\b",
        r"\bsilencing\b",
        r"\bsuppressor\b",
        r"\beffector\b",
        r"\bmediator\b",
        r"\bhistone\b",
        r"\bchromodomain\b",
        r"\bdomain\b",
        r"\blyase\b",
        r"\bdebranching\b",
        r"\boxidase\b",
        r"\bATPase\b",
        r"\btranslocase\b",
        r"\brecombinase\b",
        r"\bresponse\b",
        r"\bactivator\b",
        r"\brepressor\b",
        r"\btranscription\b",
        r"\bisoform\b",
        r"\bhomolog\b",
        r"\bglycohydrolase\b",
        r"reductase\b",
        r"deacylase\b",
        r"transglycosylase\b",
        r"\bGTPase\b",
        r"isomerase\b",
        r"amidotransferase\b",
        r"dehydratase\b",
        r"phosphotransferase\b",
        r"methylase\b",
        r"sulfurtransferase\b",
        r"carbamoyltransferase\b",
        r"\bRNase\b",
        r"\benzyme\b",
    ]),
    re.IGNORECASE,
)


def _get_molecule_name(header: str) -> str:
    """Extract the molecule-name field (3rd pipe-delimited field) from an RCSB header."""
    fields = header.split("|")
    if len(fields) >= 3:
        return fields[2].strip()
    return ""


def is_nonprotein_header(header: str) -> bool:
    """Check if the FASTA header indicates a non-protein molecule.

    Strategy:
      1. Extract the molecule name (field 3 of the pipe-delimited header).
      2. If the molecule name matches a nucleic-acid pattern AND does NOT
         match a protein-override pattern, flag it as non-protein.
      3. This avoids false positives like "DNA ligase" (matches nucleic
         pattern via "^DNA" but protein override via "ligase" → kept).
    """
    mol_name = _get_molecule_name(header)
    if not mol_name:
        return False

    if _NUCLEIC_MOLECULE_PATTERNS.search(mol_name):
        if _PROTEIN_OVERRIDE_PATTERNS.search(mol_name):
            return False  # protein that works on nucleic acids
        return True       # nucleic acid molecule

    return False


# ---------------------------------------------------------------------------
# Sequence-based detection
# ---------------------------------------------------------------------------

_NUCLEIC_CHARS = set("ATGCUNatgcun")


def is_nucleic_acid_sequence(seq: str, threshold: float = 0.90) -> bool:
    """Check if a sequence looks like nucleic acid based on composition.

    Returns True if >= threshold fraction of characters are standard
    nucleotide letters. Protein sequences contain some of these amino acids
    (Ala=A, Thr=T, Gly=G, Cys=C) but rarely at >= 90%.
    """
    if not seq or len(seq) < 3:
        return False
    nucleic_count = sum(1 for c in seq if c in _NUCLEIC_CHARS)
    return nucleic_count / len(seq) >= threshold


# ---------------------------------------------------------------------------
# Classification and file processing
# ---------------------------------------------------------------------------

def classify_record(header: str, seq: str) -> tuple[str, str]:
    """Classify a FASTA record. Returns (label, reason)."""
    if is_nonprotein_header(header):
        return "nonprotein", "header"
    if is_nucleic_acid_sequence(seq):
        return "nonprotein", "sequence_composition"
    return "protein", ""


def parse_fasta(text: str) -> list[tuple[str, str]]:
    """Parse multi-FASTA text into [(header, sequence), ...]."""
    records = []
    for block in text.strip().split(">"):
        block = block.strip()
        if not block:
            continue
        lines = block.split("\n")
        records.append((lines[0], "".join(lines[1:])))
    return records


def process_file(in_path, out_clean, out_removed) -> dict[str, int]:
    """Filter one FASTA file. Returns count statistics."""
    text = fasta_path.read_text()
    records = parse_fasta(text)

    proteins = []
    nonproteins = []

    for header, seq in records:
        label, reason = classify_record(header, seq)
        if label == "protein":
            proteins.append((header, seq))
        else:
            nonproteins.append((header, seq, reason))

    out_clean.parent.mkdir(parents=True, exist_ok=True)
    out_removed.parent.mkdir(parents=True, exist_ok=True)

    # Write protein-only back to the original file
    with open(out_clean, "w") as f:
        for header, seq in proteins:
            f.write(f">{header}\n{seq}\n")

    # Write non-proteins to a separate file
    with open(out_removed, "w") as f:
        for header, seq, _ in nonproteins:
            f.write(f">{header}\n{seq}\n")

    stats = {
        "total": len(records),
        "protein": len(proteins),
        "nonprotein": len(nonproteins),
    }

    by_header = sum(1 for _, _, r in nonproteins if r == "header")
    by_seq = sum(1 for _, _, r in nonproteins if r == "sequence_composition")

    print(f"\n{'='*60}")
    print(f"Input:    {in_path}")
    print(f"Clean:    {out_clean}")
    print(f"Removed:  {out_removed}")
    print(f"  Total records:        {stats['total']}")
    print(f"  Proteins kept:        {stats['protein']}")
    print(f"  Non-proteins removed: {stats['nonprotein']}")
    if nonproteins:
        print(f"    by header pattern:       {by_header}")
        print(f"    by sequence composition: {by_seq}")
        print(f"\n  Removed entries:")
        for header, seq, reason in nonproteins:
            short = header[:80] + "..." if len(header) > 80 else header
            print(f"    [{reason:20s}] >{short}")
    print(f"{'='*60}")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Filter non-protein sequences from a FASTA file.")
    parser.add_argument("--input", required=True, type=Path, help="Input FASTA")
    parser.add_argument("--clean-dir", required=True, type=Path, help="Output dir for protein FASTA")
    parser.add_argument("--removed-dir", required=True, type=Path, help="Output dir for non-protein FASTA")
    args = parser.parse_args()

    if not args.input.exists():
        sys.exit(f"ERROR: {args.input} not found")
    
    # Derive filenames from input stem
    stem = args.input.stem  # e.g. "sequences_before_training_cutoff"
    out_clean = args.clean_dir / f"{stem}_clean.fasta"
    out_removed = args.removed_dir / f"{stem}_removed.fasta"

    process_file(args.input, out_clean, out_removed)


if __name__ == "__main__":
    main()
