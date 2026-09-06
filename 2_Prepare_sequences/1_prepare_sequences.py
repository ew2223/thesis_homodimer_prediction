#!/usr/bin/env python3
"""
Download and filter PDB FASTA sequences listed in a CSV.

Input CSV format (with header):
    id,chain,stoichiometry

Selection logic:
    chain specified                -> keep only that chain (matches label IDs
                                      and [auth X] author IDs in the RCSB
                                      "Chains ..." header field)
    chain=N/A, monomer/homodimer   -> deduplicate identical sequences
    chain=N/A, heterodimer/other   -> keep all chains

Downloads are parallelized with a thread pool (I/O-bound workload) and cached
per-PDB-ID so re-runs and cross-CSV overlaps are free.

Example:
    python prepare_sequences.py \\
        --input ids_before_training_cutoff.csv \\
        --cache-dir sequences/_entry_cache \\
        --output sequences/sequences_before_training_cutoff.fasta \\
        --failed-log sequences/failed_before_training_cutoff.log \\
        --run-log sequences/download_before_training_cutoff.log \\
        --workers 10
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

RCSB_URL_TEMPLATE = "https://www.rcsb.org/fasta/entry/{pdb}/display"


# ---------------------------------------------------------------------------
# Pure functions for parsing and selection logic, independent of I/O and network.
# ---------------------------------------------------------------------------
def chains_in_header(header: str) -> set[str]:
    """Return the set of chain IDs (label + auth, uppercase) named in an
    RCSB FASTA header.

    Handles all three common header shapes:
        >7ACW_1|Chains A, C|...                    (label == auth)
        >1XYZ_2|Chains A, B[auth D], C[auth E]|... (label != auth)
        >1ABC_1|Chain A|...                        (single chain)
    """
    fields = header.split("|")
    if len(fields) < 2:
        return set()
    chain_field = re.sub(r"^Chains?\s+", "", fields[1].strip())
    ids: set[str] = set()
    for tok in chain_field.split(","):
        tok = tok.strip()
        if not tok:
            continue
        m = re.match(r"(\S+)\s*\[auth\s+([^\]]+)\]", tok)
        if m:
            ids.add(m.group(1).upper())          # mmCIF label ID
            ids.add(m.group(2).strip().upper())  # author ID
        else:
            ids.add(tok.upper())
    return ids


def parse_fasta(text: str) -> list[tuple[str, str]]:
    """Parse multi-FASTA text into a list of (header, sequence) tuples."""
    out: list[tuple[str, str]] = []
    for block in text.strip().split(">"):
        block = block.strip()
        if not block:
            continue
        lines = block.split("\n")
        out.append((lines[0], "".join(lines[1:])))
    return out


def read_csv_rows(path: Path) -> list[tuple[str, str, str]]:
    """Read (pdb_id, chain, stoichiometry) rows from the input CSV.

    Skips the header row, blank lines, and rows with an empty PDB ID column.
    PDB IDs are upper-cased for consistency with RCSB URLs and cache paths.
    """
    rows: list[tuple[str, str, str]] = []
    with open(path, newline="") as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header
        for r in reader:
            if not r or not r[0].strip():
                continue
            pdb_id = r[0].strip().upper()
            chain = r[1].strip() if len(r) > 1 else ""
            stoich = r[2].strip() if len(r) > 2 else ""
            rows.append((pdb_id, chain, stoich))
    return rows


def select_records(
    entries: list[tuple[str, str]],
    chain: str,
    stoichiometry: str,
) -> tuple[list[tuple[str, str]], str | None]:
    """Apply selection logic to parsed FASTA entries for one CSV row.

    Returns (selected_records, error). On success `error` is None; on failure
    (e.g. requested chain not present) `selected_records` is empty and
    `error` is a short human-readable reason.
    """
    if not entries:
        return [], "empty_fasta"

    if chain and chain.upper() != "N/A":
        want = chain.upper()
        selected = [(h, s) for h, s in entries if want in chains_in_header(h)]
        if not selected:
            return [], "chain_not_found"
        return selected, None

    if stoichiometry in ("monomer", "homodimer"):
        seen: set[str] = set()
        selected = []
        for h, s in entries:
            if s not in seen:
                seen.add(s)
                selected.append((h, s))
        return selected, None

    # heterodimer or any other stoichiometry: keep all chains
    return list(entries), None


# ---------------------------------------------------------------------------
# Network-touching code
# ---------------------------------------------------------------------------
def _cache_hit(cache_path: Path) -> bool:
    return cache_path.exists() and cache_path.stat().st_size > 0


def fetch_one(
    pdb_id: str,
    cache_dir: Path,
    max_retries: int,
    timeout: int,
) -> tuple[str, bool, str]:
    """Download one PDB entry's FASTA into the cache directory.

    Returns (pdb_id, ok, message). Safe to call concurrently from multiple
    threads: writes go to a per-entry temp file that is atomically renamed
    to the final cache path, so partial writes never pollute the cache.
    """
    cache_path = cache_dir / f"{pdb_id}.fasta"
    if _cache_hit(cache_path):
        return (pdb_id, True, "cached")

    url = RCSB_URL_TEMPLATE.format(pdb=pdb_id)
    last_err = "unknown"
    for attempt in range(1, max_retries + 1):
        try:
            req = urllib.request.Request(url, headers={"Accept": "text/plain"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = resp.read()
            if not data:
                last_err = "empty response"
            else:
                tmp = cache_path.with_suffix(".fasta.tmp")
                tmp.write_bytes(data)
                os.replace(tmp, cache_path)       # atomic
                return (pdb_id, True, "downloaded")
        except urllib.error.HTTPError as e:
            last_err = f"HTTP {e.code}"
            if e.code in (404, 410):              # permanent — don't retry
                break
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
        time.sleep(attempt)                       # 1s, 2s, 3s backoff
    return (pdb_id, False, last_err)


def download_all(
    pdb_ids: Iterable[str],
    cache_dir: Path,
    workers: int,
    max_retries: int,
    timeout: int,
    progress_every: int = 100,
) -> dict[str, tuple[bool, str]]:
    """Download every listed PDB ID that isn't already cached, in parallel.

    Returns {pdb_id: (ok, message)} for every ID in the input (cached IDs
    are marked as ok=True, msg='cached').
    """
    ids = list(pdb_ids)
    need = [p for p in ids if not _cache_hit(cache_dir / f"{p}.fasta")]
    already = len(ids) - len(need)
    print(f"Cache hits: {already}   To download: {len(need)}", flush=True)

    status: dict[str, tuple[bool, str]] = {}
    if need:
        t0 = time.time()
        done = 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(fetch_one, p, cache_dir, max_retries, timeout): p
                for p in need
            }
            for fut in as_completed(futures):
                pdb_id, ok, msg = fut.result()
                status[pdb_id] = (ok, msg)
                done += 1
                if done % progress_every == 0 or done == len(need):
                    elapsed = time.time() - t0
                    rate = done / elapsed if elapsed > 0 else 0
                    eta = (len(need) - done) / rate if rate > 0 else 0
                    print(
                        f"  [{done}/{len(need)}] {rate:.1f} req/s, "
                        f"ETA {eta:.0f}s",
                        flush=True,
                    )

    for p in ids:
        status.setdefault(p, (True, "cached"))
    return status


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def write_outputs(
    rows: list[tuple[str, str, str]],
    download_status: dict[str, tuple[bool, str]],
    cache_dir: Path,
    combined_fasta: Path,
    failed_log: Path,
) -> tuple[int, int, int]:
    """Walk rows, select records, write to the combined FASTA.

    Returns (n_success_rows, n_failed_rows, n_records_written).
    """
    success = 0
    fail = 0
    records = 0
    with open(combined_fasta, "a") as fasta_out, open(failed_log, "a") as fail_out:
        for pdb_id, chain, stoich in rows:
            ok, msg = download_status[pdb_id]
            if not ok:
                fail_out.write(f"{pdb_id},{chain},{stoich},download:{msg}\n")
                fail += 1
                continue

            cache_path = cache_dir / f"{pdb_id}.fasta"
            try:
                entries = parse_fasta(cache_path.read_text())
            except Exception as e:
                fail_out.write(f"{pdb_id},{chain},{stoich},parse:{e}\n")
                fail += 1
                continue

            selected, err = select_records(entries, chain, stoich)
            if err is not None:
                fail_out.write(f"{pdb_id},{chain},{stoich},{err}\n")
                fail += 1
                continue

            for h, s in selected:
                fasta_out.write(f">{h}\n{s}\n")
                records += 1
            success += 1
    return success, fail, records


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--input", required=True, type=Path,
                   help="Input CSV with columns: id,chain,stoichiometry")
    p.add_argument("--cache-dir", required=True, type=Path,
                   help="Per-PDB FASTA cache directory")
    p.add_argument("--output", required=True, type=Path,
                   help="Combined output FASTA (will be truncated)")
    p.add_argument("--failed-log", required=True, type=Path,
                   help="Per-row failure log (will be truncated)")
    p.add_argument("--run-log", required=True, type=Path,
                   help="Summary log (appended to)")
    p.add_argument("--workers", type=int, default=10,
                   help="Concurrent HTTP requests (default: 10)")
    p.add_argument("--max-retries", type=int, default=3,
                   help="Retries per PDB entry (default: 3)")
    p.add_argument("--timeout", type=int, default=30,
                   help="HTTP request timeout in seconds (default: 30)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Truncate per-run outputs; the cache is preserved across runs.
    args.output.write_text("")
    args.failed_log.write_text("")

    rows = read_csv_rows(args.input)
    unique_ids = sorted({pdb for pdb, _, _ in rows})
    print(f"Rows: {len(rows)}   Unique PDB IDs: {len(unique_ids)}", flush=True)

    t0 = time.time()
    status = download_all(
        unique_ids,
        cache_dir=args.cache_dir,
        workers=args.workers,
        max_retries=args.max_retries,
        timeout=args.timeout,
    )

    success, fail, records = write_outputs(
        rows,
        download_status=status,
        cache_dir=args.cache_dir,
        combined_fasta=args.output,
        failed_log=args.failed_log,
    )
    elapsed = time.time() - t0

    new_downloads = sum(1 for ok, msg in status.values() if ok and msg != "cached")
    cached = sum(1 for ok, msg in status.values() if ok and msg == "cached")

    summary = (
        "\n=== Done ===\n"
        f"Rows processed:   {len(rows)}\n"
        f"Successful:       {success}\n"
        f"Failed:           {fail}\n"
        f"Cache hits:       {cached} (downloads avoided)\n"
        f"New downloads:    {new_downloads}\n"
        f"Records written:  {records}\n"
        f"Wall time:        {elapsed:.1f}s\n"
        f"Combined FASTA:   {args.output}\n"
    )
    if fail > 0:
        summary += f"Failed entries:   {args.failed_log}\n"
    print(summary)
    with open(args.run_log, "a") as rl:
        rl.write(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
