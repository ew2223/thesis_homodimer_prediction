#!/usr/bin/env python3
"""
extract_features.py – Self-contained AlphaFold multimer feature extractor.

INPUT
  --input_dir : a directory whose immediate subdirectories are per-complex
                AlphaFold output folders (e.g. 1a17_6-1a17_6/).
                Each subfolder must contain:
                  result_model_<M>_multimer_v3_pred_<N>.pkl   (25 files: 5 models × 5 seeds)
                  unrelaxed_model_<M>_multimer_v3_pred_<N>.pdb (matching PDB files)

OUTPUT
  TSV with one row per complex.  Columns produced:

  1. AlphaFold confidence (PKL — always)
       max_iptm  min_iptm  avg_iptm
       max_ptm   min_ptm   avg_ptm
       max_rc    min_rc    avg_rc         (ranking_confidence)

  2. Foldseek homology (only when --foldseek_db is given and foldseek+mmseqs are in PATH)
       multimer_frac_tm{0.0..0.9}    fraction of hits above TM threshold that are non-monomer
       hm_frac_tm{0.0..0.9}          fraction above threshold that are homomultimer
       highest_match_all_hits  highest_tm_all_hits  stoich_all_hits
       highest_match_multimers  highest_tm_multimers  stoich_multimers
       highest_match_homomultimers  highest_tm_homomultimers

  3. Interface / SPOC features (PDB + PKL — always)
       Based on: Schmid & Walter, Mol Cell 2025 (https://github.com/walterlab-HMS/SPOC)
       num_contacts_with_max_n_models   num_unique_contacts
       mean_contacts_across_predictions  min_contacts_across_predictions
       best_num_residue_contacts  best_if_residues
       best_plddt_max  best_pae_min  best_contact_score_max

  4. FreeSASA burial (PDB — only when `freesasa` binary is in PATH)
       buried_apolar_area  buried_polar_area  total_interaction_area
       fraction_buried_apolar_area  fraction_buried_polar_area

  5. Structural consensus (only when `USalign` binary is in PATH)
       structural_consensus   (mean pairwise TM-score across all 25 structures)

USAGE
  python extract_features.py \\
      --input_dir /path/to/complexes/ \\
      --output    features.tsv \\
      [--foldseek_db /path/to/foldseek_database/entirepdb040626] \\
      [--fident_threshold 0.5] \\
      [--workers 4] [--usalign_workers 4]

"""

import argparse
import concurrent.futures as cf
import gzip
import itertools
import lzma
import math
import os
import pickle
import re
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# File-name patterns
# ─────────────────────────────────────────────────────────────────────────────

_PKL_RE = re.compile(r"result_model_(\d+)_multimer_v3_pred_(\d+)\.pkl$")
_PDB_RE = re.compile(r"unrelaxed_model_(\d+)_multimer_v3_pred_(\d+)\.pdb$")

# ─────────────────────────────────────────────────────────────────────────────
# Amino-acid lookup
# ─────────────────────────────────────────────────────────────────────────────

_AA3 = {
    "ALA": "A",
    "CYS": "C",
    "ASP": "D",
    "GLU": "E",
    "PHE": "F",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LYS": "K",
    "LEU": "L",
    "MET": "M",
    "ASN": "N",
    "PRO": "P",
    "GLN": "Q",
    "ARG": "R",
    "SER": "S",
    "THR": "T",
    "VAL": "V",
    "TRP": "W",
    "TYR": "Y",
}


# ─────────────────────────────────────────────────────────────────────────────
# Section 1 – AlphaFold PKL scalars
# ─────────────────────────────────────────────────────────────────────────────


def _read_pkl_scalars(pkl_path: str) -> dict:
    """Return iptm, ptm, ranking_confidence from one result PKL."""
    with open(pkl_path, "rb") as fh:
        data = pickle.load(fh)
    return {
        "iptm": float(data["iptm"]),
        "ptm": float(data["ptm"]),
        "ranking_confidence": float(data["ranking_confidence"]),
    }


def _read_pkl_for_spoc(pkl_path: str):
    """Return (pae_flat_str_list, iptm) for use in SPOC contact analysis."""
    with open(pkl_path, "rb") as fh:
        data = pickle.load(fh)
    pae_matrix = data["predicted_aligned_error"]
    pae_flat = [str(v) for row in pae_matrix for v in row]
    n = int(math.sqrt(len(pae_flat)))
    if n * n != len(pae_flat):
        raise ValueError(f"Non-square PAE matrix in {pkl_path}")
    return pae_flat, float(data["iptm"])


def aggregate_af_scalars(pkl_paths: list) -> dict:
    """Compute max / min / avg of iptm, ptm, ranking_confidence across all PKL files."""
    iptm_vals, ptm_vals, rc_vals = [], [], []
    for pkl_path in pkl_paths:
        try:
            s = _read_pkl_scalars(pkl_path)
            iptm_vals.append(s["iptm"])
            ptm_vals.append(s["ptm"])
            rc_vals.append(s["ranking_confidence"])
        except Exception as exc:
            print(f"  WARNING: could not read {pkl_path}: {exc}")

    if not iptm_vals:
        nan = float("nan")
        return {
            k: nan
            for k in (
                "max_iptm",
                "min_iptm",
                "avg_iptm",
                "max_ptm",
                "min_ptm",
                "avg_ptm",
                "max_rc",
                "min_rc",
                "avg_rc",
            )
        }
    return {
        "max_iptm": max(iptm_vals),
        "min_iptm": min(iptm_vals),
        "avg_iptm": float(np.mean(iptm_vals)),
        "max_ptm": max(ptm_vals),
        "min_ptm": min(ptm_vals),
        "avg_ptm": float(np.mean(ptm_vals)),
        "max_rc": max(rc_vals),
        "min_rc": min(rc_vals),
        "avg_rc": float(np.mean(rc_vals)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Section 2 – SPOC interface analysis
# Adapted from src/minimized_code_snippets_spoc.py and
# src/retrieving_spoc_features.py
# Citation: Schmid & Walter, Mol Cell 2025, doi:10.1016/j.molcel.2025.01.034
# ─────────────────────────────────────────────────────────────────────────────

_BASIC_ATOMS = {"NH2", "NZ", "ND1", "NE", "NH1"}
_ACIDIC_ATOMS = {"OE2", "OD2", "OXT"}
_HB_DONORS = {
    "OG",
    "OG1",
    "OH",
    "OE2",
    "OD2",
    "NE1",
    "ND2",
    "NE2",
    "NZ",
    "NE",
    "NH1",
    "NH2",
    "ND1",
    "N",
    "OXT",
}
_HB_ACCEPTORS = {"OG", "OG1", "OH", "OE1", "OD1", "OE2", "OD2", "O", "NE1"}
_BACKBONE_ATOMS = {"C", "CA", "O", "N"}


def _dist2(v1, v2) -> float:
    return (v1[0] - v2[0]) ** 2 + (v1[1] - v2[1]) ** 2 + (v1[2] - v2[2]) ** 2


def _atom_from_line(line: str) -> dict:
    return {
        "type": line[13:16].strip(),
        "xyz": np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])]),
    }


def _get_ca(res):
    for a in res["atoms"]:
        if a["type"] == "CA":
            return a
    return None


def _contact_type(
    res1_type: str, a1_type: str, res2_type: str, a2_type: str, d: float
) -> str:
    if d < 1:
        return "C"
    a1b = a1_type in _BASIC_ATOMS or (res1_type == "H" and a1_type == "NE2")
    a2b = a2_type in _BASIC_ATOMS or (res2_type == "H" and a2_type == "NE2")
    if (a1b and a2_type in _ACIDIC_ATOMS) or (a2b and a1_type in _ACIDIC_ATOMS):
        if d <= 5:
            return "S"
    if (a1_type in _HB_DONORS and a2_type in _HB_ACCEPTORS) or (
        a2_type in _HB_DONORS and a1_type in _HB_ACCEPTORS
    ):
        if d <= 3:
            return "H"
    if (a1_type in _ACIDIC_ATOMS and a2_type in _ACIDIC_ATOMS) or (
        a1_type in _BASIC_ATOMS and a2_type in _BASIC_ATOMS
    ):
        if d <= 5:
            return "R"
    return "V"


def _atom_contacts_between(r1: dict, r2: dict, max_d: float = 5):
    """All atom pairs within max_d between two residues."""
    contacts, min_dist = [], 1e6
    max_d2 = max_d**2
    for a1 in r1["atoms"]:
        for a2 in r2["atoms"]:
            d2 = _dist2(a1["xyz"], a2["xyz"])
            if d2 < max_d2:
                d = d2**0.5
                min_dist = min(min_dist, d)
                ct = _contact_type(r1["type"], a1["type"], r2["type"], a2["type"], d)
                contacts.append(
                    [
                        a1["type"] in _BACKBONE_ATOMS and a2["type"] in _BACKBONE_ATOMS,
                        ct,
                        d,
                    ]
                )
    return contacts, min_dist


def _pdb_lines(path: str) -> list:
    if path.endswith(".xz"):
        fh = lzma.open(path, "rt")
    elif path.endswith(".gz"):
        fh = gzip.open(path, "rt")
    else:
        fh = open(path, "rt")
    with fh:
        return fh.read().splitlines()


def get_sequences(pdb_path: str) -> dict:
    seqs, last_chain = {}, None
    for line in _pdb_lines(pdb_path):
        if line[:4] != "ATOM" or line[13:16].strip() != "N":
            continue
        chain = line[20:22].strip()
        aa = _AA3.get(line[17:20])
        if aa is None:
            return None
        if chain != last_chain:
            seqs[chain] = ""
            last_chain = chain
        seqs[chain] += aa
    return seqs


def _parse_pdb_contacts(pdb_path: str, max_dist: float = 5, min_plddt: float = 50):
    """
    Parse a PDB file and return inter-chain contacts that pass the pLDDT threshold.

    Returns (raw_contact_list, pdb_sequence_str, all_plddts_list).
    all_plddts_list contains pLDDT for every residue regardless of cutoff
    (used for whole-protein statistics).
    """
    broad_d2 = (max_dist + 20) ** 2
    chains, residues, N_coords = [], [], []
    all_plddts = []
    pdb_seq = ""
    last_chain = last_chain2 = None
    chain_idx = -1
    abs_res = 0
    current_res = None

    for line in _pdb_lines(pdb_path):
        if line[:4] != "ATOM":
            continue
        atom_type = line[13:16].strip()
        chain = line[20:22].strip()
        aa1 = _AA3.get(line[17:20])
        if aa1 is None:
            continue
        is_N = atom_type == "N"
        bfac = float(line[60:66])

        if is_N:
            abs_res += 1
            if chain != last_chain2:
                if last_chain2 is not None:
                    pdb_seq += ":"
                last_chain2 = chain
            pdb_seq += aa1
            all_plddts.append(bfac)  # always track, regardless of cutoff

        if bfac < min_plddt:
            continue  # skip low-confidence atoms for contact search

        atom = _atom_from_line(line)

        if is_N:
            if chain != last_chain:
                chain_idx += 1
                last_chain = chain
                N_coords.append([])
                residues.append([])
                chains.append(chain)

            current_res = {
                "chain": chain,
                "atoms": [],
                "c_ix": int(line[22:26]),
                "a_ix": abs_res,
                "type": aa1,
                "plddt": bfac,
            }
            residues[chain_idx].append(current_res)
            N_coords[chain_idx].append(atom["xyz"])

        if current_res is not None:
            current_res["atoms"].append(atom)

    contacts = []
    for i in range(len(chains)):
        for j in range(i + 1, len(chains)):
            c1_coords = N_coords[i]
            c2_coords = N_coords[j]
            n1, n2 = len(c1_coords), len(c2_coords)
            if n1 == 0 or n2 == 0:
                continue
            # fast broad distance filter using N-N distances
            c1m = np.tile(c1_coords, (1, n2)).reshape(n1, n2, 3)
            c2m = np.tile(c2_coords, (n1, 1)).reshape(n1, n2, 3)
            d2s = np.sum((c1m - c2m) ** 2, axis=2)
            for ri, rj in zip(*np.where(d2s < broad_d2)):
                r1, r2 = residues[i][ri], residues[j][rj]
                atom_cts, min_d = _atom_contacts_between(r1, r2, max_dist)
                if not atom_cts:
                    continue
                ca1, ca2 = _get_ca(r1), _get_ca(r2)
                if ca1 is None or ca2 is None:
                    continue
                contacts.append(
                    {
                        "distance": min_d,
                        "atom_contacts": atom_cts,
                        "clashing": any(ac[1] == "C" for ac in atom_cts),
                        "aa1": {
                            "chain": r1["chain"],
                            "ca": ca1,
                            "type": r1["type"],
                            "c_ix": r1["c_ix"],
                            "a_ix": r1["a_ix"],
                            "plddt": r1["plddt"],
                        },
                        "aa2": {
                            "chain": r2["chain"],
                            "ca": ca2,
                            "type": r2["type"],
                            "c_ix": r2["c_ix"],
                            "a_ix": r2["a_ix"],
                            "plddt": r2["plddt"],
                        },
                    }
                )

    return contacts, pdb_seq, all_plddts


def _apply_pae_filter(
    contacts: list, pae_flat: list, total_len: int, max_pae: float = 15
) -> dict:
    """
    Add PAE values to contacts, filter by max_pae, and return a nested dict.

    Structure: {chain_pair_str: {contact_id_str: contact_dict}}
    """
    filtered = {}
    for c in contacts:
        ix1, ix2 = c["aa1"]["a_ix"], c["aa2"]["a_ix"]
        pi1 = total_len * (ix1 - 1) + ix2 - 1
        pi2 = total_len * (ix2 - 1) + ix1 - 1
        if pi1 >= len(pae_flat) or pi2 >= len(pae_flat):
            continue
        pae_vals = [float(pae_flat[pi1]), float(pae_flat[pi2])]
        pae_val = min(pae_vals)
        if pae_val > max_pae:
            continue
        chain_key = c["aa1"]["chain"] + ":" + c["aa2"]["chain"]
        contact_id = f"{ix1}&{ix2}"
        if chain_key not in filtered:
            filtered[chain_key] = {}
        filtered[chain_key][contact_id] = {
            "pae": pae_val,
            "paes": pae_vals,
            "plddts": [c["aa1"]["plddt"], c["aa2"]["plddt"]],
            "distance": c["distance"],
            "atom_contacts": c["atom_contacts"],
        }
    return filtered


def _interface_stats(filtered_contacts: dict) -> dict:
    """Summarise confidence metrics across all contacts in one prediction's interface."""
    plddt_sum = pae_sum = num = 0
    plddt_min, plddt_max = 100.0, 0.0
    pae_min = 30.0
    contact_scores = []
    unique_res = {}

    for contacts in filtered_contacts.values():
        for cid, c in contacts.items():
            r1, r2 = cid.split("&")
            unique_res[r1] = unique_res[r2] = 1
            avg_plddt = float(np.mean(c["plddts"]))
            plddt_sum += avg_plddt
            plddt_max = max(plddt_max, avg_plddt)
            plddt_min = min(plddt_min, avg_plddt)
            pae_sum += c["pae"]
            pae_min = min(pae_min, c["pae"])
            # contact score from SPOC: atom_contacts * 0.5 * sum(plddts) / (1 + 0.5 * sum(paes))
            score = (
                len(c["atom_contacts"])
                * 0.5
                * sum(c["plddts"])
                / (1 + 0.5 * sum(c["paes"]))
            )
            contact_scores.append(score)
            num += 1

    if num == 0:
        plddt_min = pae_min = 0.0

    scores_arr = np.array(contact_scores) if contact_scores else np.array([0.0])
    pae_avg = pae_sum / num if num > 0 else 0.0

    return {
        "num_residue_contacts": num,
        "num_residues": len(unique_res),
        "plddt_min": plddt_min,
        "plddt_max": plddt_max,
        "pae_min": pae_min,
        "contact_score_max": float(round(np.max(scores_arr), 2)),
        # used only internally to select the "best" model
        "contacts_per_pae": round(num / (pae_avg + 1), 3),
    }


def _summarize_contact_consistency(all_contacts: dict) -> dict:
    """
    Given {model_idx: filtered_contacts_dict}, return consensus contact counts.
    """
    contact_counts: dict[str, int] = {}
    for model_contacts in all_contacts.values():
        for contacts in model_contacts.values():
            for cid in contacts:
                contact_counts[cid] = contact_counts.get(cid, 0) + 1

    if not contact_counts:
        return {"num_contacts_with_max_n_models": 0, "num_unique_contacts": 0}

    max_n = max(contact_counts.values())
    return {
        "num_contacts_with_max_n_models": sum(
            1 for c in contact_counts.values() if c == max_n
        ),
        "num_unique_contacts": len(contact_counts),
    }


def analyze_spoc(pdb_pkl_pairs: list, complex_name: str) -> dict | None:
    """
    Run SPOC-style interface analysis across all (pdb_path, pkl_path) pairs.

    Returns a flat feature dict, or None if analysis cannot proceed.
    """
    if len(pdb_pkl_pairs) < 3:
        print(f"  {complex_name}: only {len(pdb_pkl_pairs)} prediction(s), need ≥ 3")
        return None

    seqs = get_sequences(pdb_pkl_pairs[0][0])
    if seqs is None or len(seqs) != 2:
        n = len(seqs) if seqs else 0
        print(f"  {complex_name}: expected 2 chains, found {n} — skipping SPOC")
        return None

    all_contacts: dict[int, dict] = {}
    contact_counts_per_model = []
    iptm_vals = []
    best_if_stats = None

    for model_idx, (pdb_path, pkl_path) in enumerate(pdb_pkl_pairs):
        try:
            pae_flat, iptm = _read_pkl_for_spoc(pkl_path)
        except Exception as exc:
            print(f"  {complex_name}: PKL error ({pkl_path}): {exc}")
            continue

        total_len = int(math.sqrt(len(pae_flat)))

        try:
            raw_contacts, _, _ = _parse_pdb_contacts(pdb_path, max_dist=5, min_plddt=50)
        except Exception as exc:
            print(f"  {complex_name}: PDB error ({pdb_path}): {exc}")
            continue

        # remove residues involved in steric clashes (distance < 1 Å)
        clashing_residues = set()
        for c in raw_contacts:
            if c["clashing"]:
                clashing_residues.add(c["aa1"]["a_ix"])
                clashing_residues.add(c["aa2"]["a_ix"])
        clean_contacts = [
            c
            for c in raw_contacts
            if c["aa1"]["a_ix"] not in clashing_residues
            and c["aa2"]["a_ix"] not in clashing_residues
        ]

        filtered = _apply_pae_filter(clean_contacts, pae_flat, total_len, max_pae=15)
        all_contacts[model_idx] = filtered

        if_stats = _interface_stats(filtered)
        if_stats["iptm"] = iptm
        iptm_vals.append(iptm)
        contact_counts_per_model.append(if_stats["num_residue_contacts"])

        if (
            best_if_stats is None
            or if_stats["contacts_per_pae"] > best_if_stats["contacts_per_pae"]
        ):
            best_if_stats = if_stats

    if best_if_stats is None or not contact_counts_per_model:
        return None

    counts = np.array(contact_counts_per_model, dtype=float)
    summary = _summarize_contact_consistency(all_contacts)
    summary.update(
        {
            "mean_contacts_across_predictions": float(round(np.mean(counts), 0)),
            "min_contacts_across_predictions": float(np.min(counts)),
            "best_num_residue_contacts": best_if_stats["num_residue_contacts"],
            "best_if_residues": best_if_stats["num_residues"],
            "best_plddt_max": float(round(best_if_stats["plddt_max"], 0)),
            "best_pae_min": best_if_stats["pae_min"],
            "best_contact_score_max": best_if_stats["contact_score_max"],
        }
    )
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Section 3 – FreeSASA burial features
# Adapted from src/code_for_getting_freesasa_features.py
# ─────────────────────────────────────────────────────────────────────────────

_FSASA_TOTAL = re.compile(r"Total\s*:\s*([\d.]+)")
_FSASA_APOLAR = re.compile(r"Apolar\s*:\s*([\d.]+)")
_FSASA_POLAR = re.compile(r"Polar\s*:\s*([\d.]+)")
_FSASA_CHAIN = re.compile(r"CHAIN\s+(.+?)\s*:\s*([\d.]+)")


def _freesasa_available() -> bool:
    return shutil.which("freesasa") is not None


def _run_freesasa(pdb_path: str) -> dict | None:
    try:
        out = subprocess.run(
            ["freesasa", str(pdb_path)],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None

    result: dict = {"chains": {}}
    for m in _FSASA_CHAIN.finditer(out):
        result["chains"][m.group(1)] = float(m.group(2))
    for pat, key in [
        (_FSASA_TOTAL, "total"),
        (_FSASA_APOLAR, "apolar"),
        (_FSASA_POLAR, "polar"),
    ]:
        m = pat.search(out)
        if m:
            result[key] = float(m.group(1))
    return result


def _write_chain(pdb_path: str, chain: str, out_path: str):
    with open(pdb_path) as fi, open(out_path, "w") as fo:
        for ln in fi:
            if ln.startswith(("ATOM", "HETATM")) and ln[21:22].strip() == chain:
                fo.write(ln)
        fo.write("END\n")


def _freesasa_one_prediction(pdb_path: str, pid: str) -> dict:
    """Compute burial metrics for one PDB file. Returns NaN dict on failure."""
    _nan_row = {k: float("nan") for k in (
        "buried_apolar_area", "buried_polar_area", "total_interaction_area",
        "fraction_buried_apolar_area", "fraction_buried_polar_area",
    )}

    full = _run_freesasa(pdb_path)
    if not full:
        return _nan_row

    chains = list(full.get("chains", {}).keys())
    if len(chains) < 2:
        return _nan_row

    ch1, ch2 = chains[:2]
    t1 = f"/tmp/{pid}_{ch1}_{uuid.uuid4().hex}.pdb"
    t2 = f"/tmp/{pid}_{ch2}_{uuid.uuid4().hex}.pdb"
    try:
        _write_chain(pdb_path, ch1, t1)
        _write_chain(pdb_path, ch2, t2)
        s1 = _run_freesasa(t1)
        s2 = _run_freesasa(t2)
    finally:
        Path(t1).unlink(missing_ok=True)
        Path(t2).unlink(missing_ok=True)

    if not (s1 and s2):
        return _nan_row

    apol  = s1.get("apolar", float("nan")) + s2.get("apolar", float("nan")) - full.get("apolar", float("nan"))
    pol   = s1.get("polar",  float("nan")) + s2.get("polar",  float("nan")) - full.get("polar",  float("nan"))
    total = s1.get("total",  float("nan")) + s2.get("total",  float("nan")) - full.get("total",  float("nan"))

    # Fraction is NaN when total ≤ 0 (no interface for this model).
    # The complex-level aggregation uses nanmean/nanmin/nanmax to exclude those models.
    def _frac(num, denom):
        if math.isnan(num) or math.isnan(denom) or denom <= 0:
            return float("nan")
        return num / denom

    return {
        "buried_apolar_area":          apol,
        "buried_polar_area":           pol,
        "total_interaction_area":      total,
        "fraction_buried_apolar_area": _frac(apol, total),
        "fraction_buried_polar_area":  _frac(pol,  total),
    }


def compute_freesasa_complex(pdb_paths: list, protein_id: str) -> dict:
    """
    Aggregate FreeSASA burial features across all predictions for a complex.

    Raw areas use regular mean/min/max — zero interface area is meaningful data.
    Fractions use nan-aware aggregation: models with zero interface (fraction=NaN)
    are excluded. If ALL models had zero interface, the fraction stays NaN.
    Returns NaN for all columns if `freesasa` is not in PATH.
    """
    nan = float("nan")
    nan_result = {
        "buried_apolar_area_mean": nan, "buried_polar_area_mean": nan, "total_interaction_area_mean": nan,
        "fraction_buried_apolar_area_mean": nan, "fraction_buried_polar_area_mean": nan,
        "buried_apolar_area_min": nan, "buried_polar_area_min": nan, "total_interaction_area_min": nan,
        "fraction_buried_apolar_area_min": nan, "fraction_buried_polar_area_min": nan,
        "buried_apolar_area_max": nan, "buried_polar_area_max": nan, "total_interaction_area_max": nan,
        "fraction_buried_apolar_area_max": nan, "fraction_buried_polar_area_max": nan,
    }

    if not _freesasa_available():
        return nan_result

    rows = [_freesasa_one_prediction(p, protein_id) for p in pdb_paths]
    df = pd.DataFrame(rows).apply(pd.to_numeric, errors="coerce")

    def _nanstat(col, fn):
        """Apply fn to non-NaN values; return NaN if all values are NaN."""
        v = df[col].dropna().values
        return float(fn(v)) if len(v) > 0 else nan

    return {
        # Raw areas: regular aggregation (zeros are meaningful)
        "buried_apolar_area_mean":          float(df["buried_apolar_area"].mean()),
        "buried_polar_area_mean":           float(df["buried_polar_area"].mean()),
        "total_interaction_area_mean":      float(df["total_interaction_area"].mean()),
        "buried_apolar_area_min":           float(df["buried_apolar_area"].min()),
        "buried_polar_area_min":            float(df["buried_polar_area"].min()),
        "total_interaction_area_min":       float(df["total_interaction_area"].min()),
        "buried_apolar_area_max":           float(df["buried_apolar_area"].max()),
        "buried_polar_area_max":            float(df["buried_polar_area"].max()),
        "total_interaction_area_max":       float(df["total_interaction_area"].max()),
        # Fractions: nan-aware — models with zero interface excluded from aggregate
        "fraction_buried_apolar_area_mean": _nanstat("fraction_buried_apolar_area", np.mean),
        "fraction_buried_polar_area_mean":  _nanstat("fraction_buried_polar_area",  np.mean),
        "fraction_buried_apolar_area_min":  _nanstat("fraction_buried_apolar_area", np.min),
        "fraction_buried_polar_area_min":   _nanstat("fraction_buried_polar_area",  np.min),
        "fraction_buried_apolar_area_max":  _nanstat("fraction_buried_apolar_area", np.max),
        "fraction_buried_polar_area_max":   _nanstat("fraction_buried_polar_area",  np.max),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Section 4 – Structural consensus (USalign pairwise TM-score)
# ─────────────────────────────────────────────────────────────────────────────

_TM_SCORE_RE = re.compile(r"TM-score=\s*([\d.]+)")


def _usalign_available() -> bool:
    return shutil.which("USalign") is not None


def _tm_score_pair(pdb1: str, pdb2: str) -> float | None:
    """Run USalign on two PDB files; return average of the two reported TM-scores."""
    try:
        out = subprocess.run(
            ["USalign", pdb1, pdb2, "-mm", "1"],
            capture_output=True,
            text=True,
            timeout=120,
        ).stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None

    scores = [float(m.group(1)) for m in _TM_SCORE_RE.finditer(out)]
    if len(scores) >= 2:
        if abs(scores[0] - scores[1]) < 1e-6:
            return scores[0]
        elif scores[0] < scores[1]:
            return scores[0]
        else:
            return scores[1]
    return scores[0] if scores else None


def compute_structural_consensus(pdb_paths: list, n_workers: int = 4) -> float:
    """
    Mean pairwise TM-score across all unique pairs of PDB structures.
    Returns NaN if USalign is not in PATH or no pairs succeed.

    Note: 25 structures → 300 pairs.  Use n_workers to parallelise.
    """
    if not _usalign_available():
        return (float("nan"), float("nan"), float("nan"))

    pairs = list(itertools.combinations(pdb_paths, 2))
    scores = []

    with cf.ThreadPoolExecutor(max_workers=n_workers) as pool:
        futs = {pool.submit(_tm_score_pair, p1, p2): (p1, p2) for p1, p2 in pairs}
        for fut in cf.as_completed(futs):
            s = fut.result()
            if s is not None:
                scores.append(s)
    if not scores:
        print("  WARNING: no successful USalign pairs")
        return (float("nan"), float("nan"), float("nan"))

    return float(np.mean(scores)), float(np.min(scores)), float(np.max(scores))


# ─────────────────────────────────────────────────────────────────────────────
# Section 5 – Foldseek homology features
# Mirrors run.py and make_logreg_features_df.py logic.
# Requires: foldseek + mmseqs binaries in PATH, --foldseek_db path,
#           and entire_pdb_cache.pkl in the same directory as the database.
# ─────────────────────────────────────────────────────────────────────────────

_FRAC_THRESHOLDS = [round(x * 0.1, 1) for x in range(10)]  # [0.0, 0.1, … 0.9]


def _foldseek_available() -> bool:
    return shutil.which("foldseek") is not None


def _mmseqs_available() -> bool:
    return shutil.which("mmseqs") is not None


def _nan_foldseek_row() -> dict:
    nan = float("nan")
    out: dict = {}
    for t in _FRAC_THRESHOLDS:
        out[f"multimer_frac_tm{t}"] = nan
        out[f"hm_frac_tm{t}"] = nan
    out.update(
        {
            "highest_match_all_hits": nan,
            "highest_tm_all_hits": nan,
            "stoich_all_hits": nan,
            "highest_match_multimers": nan,
            "highest_tm_multimers": nan,
            "stoich_multimers": nan,
            "highest_match_homomultimers": nan,
            "highest_tm_homomultimers": nan,
        }
    )
    return out


def _get_sequence_fasta(pdb_path: str, query_id: str, out_path: str) -> bool:
    """Write chain-A sequence from pdb_path as FASTA to out_path."""
    seqs = get_sequences(pdb_path)
    if not seqs:
        return False
    seq = list(seqs.values())[0]
    with open(out_path, "w") as fh:
        fh.write(f">{query_id}\n{seq}\n")
    return True


def _load_stoich_cache(foldseek_db: str) -> dict:
    cache = Path(foldseek_db).parent / "entire_pdb_cache.pkl"
    if not cache.exists():
        raise FileNotFoundError(f"Stoichiometry cache not found: {cache}")
    with open(cache, "rb") as fh:
        return pickle.load(fh)


def compute_foldseek_features(
    query_pdb: str,
    protein_id: str,
    foldseek_db: str,
    fident_threshold: float = 0.5,
) -> dict:
    """
    Run Foldseek + MMseqs2 and return homology fraction features.
    Returns NaN for all columns if foldseek/mmseqs are not in PATH,
    the database is missing, or search produces no usable hits.
    """
    if not (_foldseek_available() and _mmseqs_available()):
        return _nan_foldseek_row()

    uid = uuid.uuid4().hex
    tmp = Path(f"/tmp/fseek_{protein_id}_{uid}")
    tmp.mkdir(parents=True, exist_ok=True)
    aln_path = str(tmp / "fseek_aln")
    mmseqs_aln = str(tmp / "mmseqs_aln")
    fasta_path = str(tmp / f"{protein_id}.fasta")
    

    try:
        # Run Foldseek (TM-score alignment, alignment-type 1)
        subprocess.run(
            [
                "foldseek", "easy-search",
                query_pdb, foldseek_db, aln_path, str(tmp / "tmp"),
                "--alignment-type", "1",
                "--format-output", "query,target,evalue",
            ],
            check=True,
            capture_output=True,
        )

        if not os.path.exists(aln_path) or os.stat(aln_path).st_size == 0:
            return _nan_foldseek_row()

        fseek_df = pd.read_table(
            aln_path, header=None, keep_default_na=False,
            names=["query", "target", "evalue"],
        )
        fseek_df["evalue"] = pd.to_numeric(fseek_df["evalue"], errors="coerce")

        # Run MMseqs2 to find sequence-similar hits → filter them out
        if _get_sequence_fasta(query_pdb, protein_id, fasta_path):
            subprocess.run(
                [
                    "mmseqs", "easy-search",
                    fasta_path, foldseek_db, mmseqs_aln, str(tmp / "tmp"),
                    "--format-output", "query,target,fident",
                ],
                check=True,
                capture_output=True,
            )
            if os.path.exists(mmseqs_aln) and os.stat(mmseqs_aln).st_size > 0:
                mm_df = pd.read_table(
                    mmseqs_aln, header=None, keep_default_na=False,
                    names=["query", "target", "fident"],
                )
                mm_df["fident"] = pd.to_numeric(mm_df["fident"], errors="coerce")
                seq_similar = set(mm_df.loc[mm_df["fident"] > fident_threshold, "target"])
                fseek_df = fseek_df[~fseek_df["target"].isin(seq_similar)]

        # Load stoichiometry cache and annotate hits
        stoich_cache = _load_stoich_cache(foldseek_db)
        query_pdbid = protein_id[:4].lower()

        records = []
        for _, row in fseek_df.iterrows():
            raw_target = str(row["target"])
            assembly = raw_target.split("_")[0]  # "7u2s-assembly1_B" → "7u2s-assembly1"
            target_pdbid = assembly.split("-")[0].lower()

            # Skip self-hits (target PDB matches query protein)
            if target_pdbid == query_pdbid:
                continue

            stoich = stoich_cache.get(assembly, "unknown") or "unknown"
            if stoich == "unknown":
                print(f"  NOTE: unknown stoichiometry for assembly '{assembly}' ({protein_id})")
            if "homo" in stoich:
                stoich = "homomultimer"
            elif "hetero" in stoich:
                stoich = "heteromultimer"

            records.append(
                {"target": assembly, "evalue": float(row["evalue"]), "stoichiometry": stoich}
            )

        if not records:
            return _nan_foldseek_row()

        hits = (
            pd.DataFrame(records)
            .drop_duplicates(subset=["target"])
            .reset_index(drop=True)
        )

        # Fraction features at each TM-score threshold
        # Exclude "unknown" stoichiometry hits from fraction calculations.
        known = hits[hits["stoichiometry"] != "unknown"]
        out: dict = {}
        for t in _FRAC_THRESHOLDS:
            above = known[known["evalue"] > t]
            n = len(above)
            if n == 0:
                out[f"multimer_frac_tm{t}"] = 0.0
                out[f"hm_frac_tm{t}"] = 0.0
            else:
                out[f"multimer_frac_tm{t}"] = round(
                    len(above[above["stoichiometry"] != "monomer"]) / n, 4
                )
                out[f"hm_frac_tm{t}"] = round(
                    len(above[above["stoichiometry"] == "homomultimer"]) / n, 4
                )

        # Best hits (highest TM-score) per category
        best_all = hits.loc[hits["evalue"].idxmax()]
        out["highest_match_all_hits"] = best_all["target"]
        out["highest_tm_all_hits"] = best_all["evalue"]
        out["stoich_all_hits"] = best_all["stoichiometry"]

        multimers = known[known["stoichiometry"] != "monomer"]
        if not multimers.empty:
            bm = multimers.loc[multimers["evalue"].idxmax()]
            out["highest_match_multimers"] = bm["target"]
            out["highest_tm_multimers"] = bm["evalue"]
            out["stoich_multimers"] = bm["stoichiometry"]
        else:
            out["highest_match_multimers"] = "N/A"
            out["highest_tm_multimers"] = float("nan")
            out["stoich_multimers"] = "N/A"

        homomulti = hits[hits["stoichiometry"] == "homomultimer"]
        if not homomulti.empty:
            bh = homomulti.loc[homomulti["evalue"].idxmax()]
            out["highest_match_homomultimers"] = bh["target"]
            out["highest_tm_homomultimers"] = bh["evalue"]
        else:
            out["highest_match_homomultimers"] = "N/A"
            out["highest_tm_homomultimers"] = float("nan")

        return out

    except Exception as exc:
        print(f"  WARNING: Foldseek features failed for {protein_id}: {exc}")
        print()
        return _nan_foldseek_row()
    finally:
        shutil.rmtree(str(tmp), ignore_errors=True)


# ─────────────────────────────────────────────────────────────────────────────
# Section 6 – ipSAE interface scoring
# Wraps ipsae.py; runs per model, aggregates to min/max/avg across predictions.
# ─────────────────────────────────────────────────────────────────────────────


def _pae_dist_strings(pae_cutoff: float, dist_cutoff: float):
    return str(int(pae_cutoff)).zfill(2), str(int(dist_cutoff)).zfill(2)


def _parse_ipsae_from_txt(txt_path: Path) -> float:
    """Extract ipSAE from the 'max' row of an ipsae .txt output file."""
    with open(txt_path) as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 6 and parts[4] == "max":
                return float(parts[5])
    raise ValueError(f"No 'max' row found in {txt_path}")


def _run_ipsae_for_model(
    ipsae_script: Path,
    pdb: Path,
    pkl: Path,
    pae_cutoff: float,
    dist_cutoff: float,
    python_exe: str = None,
) -> float:
    """Run ipsae.py in a temp directory; side-files are discarded automatically."""
    interpreter = python_exe if python_exe else sys.executable
    tmpdir = Path(tempfile.mkdtemp())
    try:
        tmp_pdb = tmpdir / pdb.name
        tmp_pkl = tmpdir / pkl.name
        tmp_pdb.symlink_to(pdb.resolve())
        tmp_pkl.symlink_to(pkl.resolve())

        result = subprocess.run(
            [interpreter, str(ipsae_script), str(tmp_pkl), str(tmp_pdb),
             str(pae_cutoff), str(dist_cutoff)],
            capture_output=True,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.decode().strip())

        pae_str, dist_str = _pae_dist_strings(pae_cutoff, dist_cutoff)
        stem = str(tmp_pdb).replace(".pdb", f"_{pae_str}_{dist_str}")
        return _parse_ipsae_from_txt(Path(stem + ".txt"))
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def compute_ipsae_complex(
    pdb_pkl_pairs: list,
    ipsae_script: Path | None,
    pae_cutoff: float,
    dist_cutoff: float,
    python_exe: str | None = None,
) -> dict:
    """
    Run ipsae.py for each (pdb, pkl) pair and return min/max/avg ipSAE.
    Returns NaN for all columns if ipsae_script is None.
    """
    nan = {"min_ipsae": float("nan"), "max_ipsae": float("nan"), "avg_ipsae": float("nan")}
    if ipsae_script is None:
        return nan

    vals = []
    for pdb_path, pkl_path in pdb_pkl_pairs:
        try:
            val = _run_ipsae_for_model(
                ipsae_script, Path(pdb_path), Path(pkl_path),
                pae_cutoff, dist_cutoff, python_exe,
            )
            vals.append(val)
        except Exception as exc:
            print(f"  WARNING: ipsae failed for {Path(pdb_path).name}: {exc}")

    if not vals:
        return nan
    return {
        "min_ipsae": round(min(vals), 6),
        "max_ipsae": round(max(vals), 6),
        "avg_ipsae": round(float(np.mean(vals)), 6),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Orchestration – per-complex
# ─────────────────────────────────────────────────────────────────────────────


def _discover_pairs(complex_dir: Path) -> list[tuple[str, str]]:
    """
    Find all matched (pdb_path, pkl_path) pairs in complex_dir.
    Matched by (model_number, prediction_number).
    Returns list sorted by (model_num, pred_num).
    """
    pkl_map: dict[tuple, str] = {}
    pdb_map: dict[tuple, str] = {}

    for f in complex_dir.iterdir():
        m = _PKL_RE.match(f.name)
        if m:
            pkl_map[(int(m.group(1)), int(m.group(2)))] = str(f)
            continue
        m = _PDB_RE.match(f.name)
        if m:
            pdb_map[(int(m.group(1)), int(m.group(2)))] = str(f)

    common = sorted(set(pkl_map) & set(pdb_map))
    return [(pdb_map[k], pkl_map[k]) for k in common]


def _find_search_pdb(complex_dir: Path, pdb_paths: list) -> str:
    """Return ranked_0.pdb if present, else the first unrelaxed PDB."""
    ranked_0 = complex_dir / "ranked_0.pdb"
    if ranked_0.exists():
        return str(ranked_0)
    return pdb_paths[0]


def process_complex(
    complex_dir: Path,
    usalign_workers: int = 4,
    foldseek_db: str | None = None,
    fident_threshold: float = 0.5,
    ipsae_script: Path | None = None,
    ipsae_python: str | None = None,
    pae_cutoff: float = 10.0,
    dist_cutoff: float = 10.0,
) -> dict:
    """Extract all features for one complex directory. Returns one TSV row as a dict."""
    name = complex_dir.name
        # For homodimers named "1a17_6-1a17_6" extract "1a17_6"; fall back to full name.
    protein_id = name.split("-")[0] if "-" in name else name

    row: dict = {"ID": protein_id}

    pairs = _discover_pairs(complex_dir)
    if not pairs:
        print(f"  {name}: no matched PDB/PKL pairs found — skipping")
        return row

    pdb_paths = [p for p, _ in pairs]
    pkl_paths = [k for _, k in pairs]

    # 1. AlphaFold confidence scalars
    row.update(aggregate_af_scalars(pkl_paths))

    # 2. Foldseek homology features
    if foldseek_db is not None:
        search_pdb = _find_search_pdb(complex_dir, pdb_paths)
        row.update(
            compute_foldseek_features(search_pdb, protein_id, foldseek_db, fident_threshold)
        )
    else:
        row.update(_nan_foldseek_row())

    # 3. SPOC interface features
    spoc_cols = [
        "num_contacts_with_max_n_models",
        "num_unique_contacts",
        "mean_contacts_across_predictions",
        "min_contacts_across_predictions",
        "best_num_residue_contacts",
        "best_if_residues",
        "best_plddt_max",
        "best_pae_min",
        "best_contact_score_max",
    ]
    spoc = analyze_spoc(pairs, name)
    if spoc:
        row.update({c: spoc.get(c, float("nan")) for c in spoc_cols})
    else:
        row.update({c: float("nan") for c in spoc_cols})

    # 4. FreeSASA burial
    row.update(compute_freesasa_complex(pdb_paths, protein_id))

    # 5. Structural consensus
    structural_consensus_mean, structural_consensus_min, structural_consensus_max = compute_structural_consensus(
        pdb_paths, usalign_workers
    )
    row["structural_consensus_mean"] = structural_consensus_mean
    row["structural_consensus_min"] = structural_consensus_min
    row["structural_consensus_max"] = structural_consensus_max

    # 6. ipSAE interface scoring
    row.update(compute_ipsae_complex(pairs, ipsae_script, pae_cutoff, dist_cutoff, ipsae_python))

    return row

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

_COLUMN_ORDER = (
    ["ID"]
    # AF scalars
    + ["max_iptm", "min_iptm", "avg_iptm", "max_ptm", "min_ptm", "avg_ptm", "max_rc", "min_rc", "avg_rc"]
    # Foldseek fraction features
    + [f"multimer_frac_tm{t}" for t in _FRAC_THRESHOLDS]
    + [f"hm_frac_tm{t}" for t in _FRAC_THRESHOLDS]
    # Foldseek best-hit features
    + [
        "highest_match_all_hits", "highest_tm_all_hits", "stoich_all_hits",
        "highest_match_multimers", "highest_tm_multimers", "stoich_multimers",
        "highest_match_homomultimers", "highest_tm_homomultimers",
    ]
    # SPOC
    + [
        "num_contacts_with_max_n_models", "num_unique_contacts",
        "mean_contacts_across_predictions", "min_contacts_across_predictions",
        "best_num_residue_contacts", "best_if_residues",
        "best_plddt_max", "best_pae_min", "best_contact_score_max",
    ]
    # FreeSASA
    + [
        "buried_apolar_area_mean", "buried_polar_area_mean", "total_interaction_area_mean",
        "fraction_buried_apolar_area_mean", "fraction_buried_polar_area_mean",
        "buried_apolar_area_min", "buried_polar_area_min", "total_interaction_area_min",
        "fraction_buried_apolar_area_min", "fraction_buried_polar_area_min",
        "buried_apolar_area_max", "buried_polar_area_max", "total_interaction_area_max",
        "fraction_buried_apolar_area_max", "fraction_buried_polar_area_max",
    ]
    # Structural consensus
    + ["structural_consensus_mean", "structural_consensus_min", "structural_consensus_max"]
    # ipSAE
    + ["min_ipsae", "max_ipsae", "avg_ipsae"]
)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Directory whose immediate subdirectories are per-complex AF output folders.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output TSV path (parent directories are created as needed).",
    )
    parser.add_argument(
        "--foldseek_db",
        default=None,
        help=(
            "Path to Foldseek database prefix (e.g. /path/to/foldseek_database/entirepdb260625). "
            "The directory must also contain entire_pdb_cache.pkl. "
            "If omitted, Foldseek columns are filled with NaN."
        ),
    )
    parser.add_argument(
        "--fident_threshold",
        type=float,
        default=0.5,
        help=(
            "MMseqs2 sequence-identity threshold for filtering Foldseek hits (default: 0.5). "
            "Hits with sequence identity > this value are removed before computing fractions."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel worker processes for complex-level processing (default: 1).",
    )
    parser.add_argument(
        "--usalign_workers",
        type=int,
        default=4,
        help="Threads for pairwise USalign comparisons within one complex (default: 4).",
    )
    parser.add_argument(
        "--ipsae_script",
        default=None,
        help="Path to ipsae.py. If omitted, ipSAE columns are filled with NaN.",
    )
    parser.add_argument(
        "--ipsae_python",
        default=None,
        help="Python interpreter to use for ipsae.py (default: same as this script).",
    )
    parser.add_argument(
        "--pae_cutoff",
        type=float,
        default=10.0,
        help="PAE cutoff for ipSAE scoring (default: 10.0).",
    )
    parser.add_argument(
        "--dist_cutoff",
        type=float,
        default=10.0,
        help="Distance cutoff for ipSAE scoring (default: 10.0).",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    complex_dirs = sorted(d for d in input_dir.iterdir() if d.is_dir())

    if not complex_dirs:
        print(f"No subdirectories found in {input_dir}")
        return

    foldseek_db = args.foldseek_db
    ipsae_script = Path(args.ipsae_script) if args.ipsae_script else None
    print(f"Found {len(complex_dirs)} complex director(ies) in {input_dir}")
    print(f"  foldseek available : {_foldseek_available()}")
    print(f"  mmseqs available   : {_mmseqs_available()}")
    print(f"  freesasa available : {_freesasa_available()}")
    print(f"  USalign available  : {_usalign_available()}")
    if foldseek_db:
        print(f"  foldseek_db        : {foldseek_db}")
        print(f"  fident_threshold   : {args.fident_threshold}")
    else:
        print("  → Foldseek columns will be NaN  (provide --foldseek_db to enable)")
    if not _freesasa_available():
        print("  → FreeSASA columns will be NaN  (install: https://freesasa.github.io/)")
    if not _usalign_available():
        print("  → structural_consensus will be NaN  (install: https://zhanggroup.org/US-align/)")
    print()

    rows = []

    if args.workers <= 1:
        for d in tqdm(complex_dirs, desc="Complexes"):
            rows.append(
                process_complex(
                    d, args.usalign_workers, foldseek_db, args.fident_threshold,
                    ipsae_script, args.ipsae_python, args.pae_cutoff, args.dist_cutoff,
                )
            )
    else:
        with cf.ProcessPoolExecutor(max_workers=args.workers) as pool:
            futs = {
                pool.submit(
                    process_complex, d, args.usalign_workers, foldseek_db, args.fident_threshold,
                    ipsae_script, args.ipsae_python, args.pae_cutoff, args.dist_cutoff,
                ): d
                for d in complex_dirs
            }
            for fut in tqdm(cf.as_completed(futs), total=len(futs), desc="Complexes"):
                try:
                    rows.append(fut.result())
                except Exception as exc:
                    print(f"  ERROR processing {futs[fut].name}: {exc}")

    df = pd.DataFrame(rows)
    # enforce column order; keep any extra columns at the end
    ordered = [c for c in _COLUMN_ORDER if c in df.columns]
    extra = [c for c in df.columns if c not in ordered]
    df = df[ordered + extra]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, sep="\t", index=False)
    print(f"Wrote {len(df)} row(s) to {out_path}")


if __name__ == "__main__":
    main()