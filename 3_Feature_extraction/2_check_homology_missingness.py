#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_homology_missingness.py
Pre-check: is homology missingness clean (all-or-nothing) or are there PARTIAL rows?

For each split it sorts every row into one of four states:
  complete       : no homology column is NaN
  no_homolog     : ALL homology columns NaN  (legit "Foldseek found nothing")
  partial_typed  : NaNs ONLY in type-specific cols (benign: "no hit of that type")
  partial_core   : NaNs in core cols          (BROKEN: extraction half-failed -> fix)

Usage:
  python check_homology_missingness.py features_before.tsv features_after.tsv
"""
import sys
import pandas as pd

HOM_NUM = ([f"multimer_frac_tm{t/10:.1f}" for t in range(10)]
           + [f"hm_frac_tm{t/10:.1f}" for t in range(10)]
           + ["highest_tm_all_hits", "highest_tm_multimers", "highest_tm_homomultimers"])
HOM_CAT = ["stoich_all_hits", "stoich_multimers"]
HOM_ALL = HOM_NUM + HOM_CAT
# columns that MUST be present whenever any homolog hit exists:
HOM_TYPED = ["highest_tm_multimers", "highest_tm_homomultimers", "stoich_multimers"]
HOM_CORE = [c for c in HOM_ALL if c not in HOM_TYPED]


def read_tsv(path):
    df = pd.read_csv(path, sep="\t")
    keep = [c for c in df.columns if str(c).strip() and not str(c).startswith("Unnamed")]
    return df[keep]


def check(path):
    df = read_tsv(path)
    hom  = [c for c in HOM_ALL  if c in df.columns]
    core = [c for c in HOM_CORE if c in df.columns]
    typed = [c for c in HOM_TYPED if c in df.columns]
    miss_expected = sorted(set(HOM_ALL) - set(hom))

    na = df[hom].isna()
    all_na  = na.all(axis=1)
    any_na  = na.any(axis=1)
    core_na = df[core].isna().any(axis=1) if core else pd.Series(False, index=df.index)

    complete      = ~any_na
    no_homolog    = all_na
    partial_core  = any_na & ~all_na & core_na          # BROKEN
    partial_typed = any_na & ~all_na & ~core_na         # benign

    print(f"\n=== {path}  (rows={len(df)}) ===")
    if miss_expected:
        print(f"  NOTE: expected homology columns absent from file: {miss_expected}")
    print(f"  complete       : {complete.sum():5d}")
    print(f"  no_homolog     : {no_homolog.sum():5d}")
    print(f"  partial_typed  : {partial_typed.sum():5d}   (benign -> fill typed cols with 0)")
    print(f"  partial_core   : {partial_core.sum():5d}   <-- BROKEN if > 0")

    if partial_core.any():
        bad = df.loc[partial_core, core].isna().sum()
        ids = df.loc[partial_core, "ID"].head(8).tolist() if "ID" in df.columns else []
        print(f"  >>> ERROR: {partial_core.sum()} rows have NaNs in CORE homology columns.")
        print(f"      core columns with NaNs: {bad[bad > 0].to_dict()}")
        print(f"      example IDs: {ids}")
        print(f"      suggestion: re-run Foldseek/MMseqs extraction for these IDs; do NOT zero-fill.")
    if partial_typed.any():
        bad = df.loc[partial_typed, typed].isna().sum()
        print(f"  partial_typed NaNs by column: {bad[bad > 0].to_dict()}")
    return int(partial_core.sum())


if __name__ == "__main__":
    paths = sys.argv[1:] or ["features_before.tsv", "features_after.tsv"]
    broken = sum(check(p) for p in paths)
    print(f"\nTOTAL broken (partial_core) rows across files: {broken}")
    sys.exit(1 if broken else 0)
