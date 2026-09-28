#!/usr/bin/env python3
"""
Analyze dimension sweep study results.

Loads scalability_results.json from each n_dimensions config,
computes speedup from sequential vs ensemble runs, and compares
across dimension counts.

Expected outcome:
  - Speedup increases with n_dimensions (higher compute density)
  - This shows the threading architecture scales when compute/sync ratio improves
"""

import json
import os
import sys
import glob
import csv
from collections import defaultdict

import numpy as np

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results_dim_study")


def load_and_compute(config_dir, name):
    """Load JSON, compute speedup from sequential vs ensemble runs."""
    json_file = os.path.join(config_dir, "scalability_results.json")
    if not os.path.exists(json_file):
        return []

    with open(json_file) as f:
        data = json.load(f)

    seq_times = defaultdict(list)
    ens_times = defaultdict(list)
    ens_tps = defaultdict(list)

    for entry in data:
        mode = entry.get("mode", "")
        k = entry.get("n_detectors", 0)
        if k == 0:
            continue
        if mode == "sequential":
            seq_times[k].append(entry["elapsed_sec"])
        elif mode == "ensemble":
            ens_times[k].append(entry["elapsed_sec"])
            ens_tps[k].append(entry.get("throughput_sps", 0))

    # Parse config name: dimX
    n_dim = int(name.replace("dim", ""))

    rows = []
    for k in sorted(ens_times.keys()):
        mean_ens_time = np.mean(ens_times[k])
        mean_ens_tps = np.mean(ens_tps[k])

        if k in seq_times and seq_times[k]:
            seq_mean = np.mean(seq_times[k])
            speedup = seq_mean / mean_ens_time if mean_ens_time > 0 else 0
        else:
            speedup = 0

        rows.append({
            "config": name,
            "n_dimensions": n_dim,
            "K": k,
            "speedup": speedup,
            "throughput_sps": mean_ens_tps,
            "wall_time_sec": mean_ens_time,
            "n_reps": len(ens_times[k]),
        })

    return rows


def main():
    if not os.path.exists(RESULTS_DIR):
        print(f"Results directory not found: {RESULTS_DIR}")
        sys.exit(1)

    all_rows = []
    for config_dir in sorted(glob.glob(os.path.join(RESULTS_DIR, "*"))):
        if not os.path.isdir(config_dir):
            continue
        name = os.path.basename(config_dir)
        rows = load_and_compute(config_dir, name)
        if not rows:
            print(f"  [WARN] No results in {name}")
        all_rows.extend(rows)

    if not all_rows:
        print("No results found. Are the jobs complete?")
        sys.exit(1)

    # Group by K
    all_k = sorted(set(r["K"] for r in all_rows))

    # Main table: speedup vs n_dimensions for each K
    print(f"\n{'='*90}")
    print("Speedup vs n_dimensions (high frequency, 2 NUMA domains, pinned)")
    print(f"{'='*90}")

    # Header
    dims = sorted(set(r["n_dimensions"] for r in all_rows))
    header = f"{'K':>6}"
    for d in dims:
        header += f" {f'dim={d}':>12}"
    print(header)
    print(f"{'-'*6}" + f" {'-'*12}" * len(dims))

    for k in all_k:
        row_str = f"{k:>6}"
        for d in dims:
            matching = [r for r in all_rows if r["K"] == k and r["n_dimensions"] == d]
            if matching:
                sp = np.mean([r["speedup"] for r in matching])
                row_str += f" {sp:>12.2f}"
            else:
                row_str += f" {'N/A':>12}"
        print(row_str)

    # Throughput table
    print(f"\n{'='*90}")
    print("Throughput (sps) vs n_dimensions")
    print(f"{'='*90}")

    print(header)
    print(f"{'-'*6}" + f" {'-'*12}" * len(dims))

    for k in all_k:
        row_str = f"{k:>6}"
        for d in dims:
            matching = [r for r in all_rows if r["K"] == k and r["n_dimensions"] == d]
            if matching:
                tps = np.mean([r["throughput_sps"] for r in matching])
                row_str += f" {tps:>12.1f}"
            else:
                row_str += f" {'N/A':>12}"
        print(row_str)

    # Speedup improvement from min dim to max dim
    min_dim = min(dims)
    max_dim = max(dims)
    print(f"\n{'='*90}")
    print(f"Speedup improvement: dim={min_dim} -> dim={max_dim}")
    print(f"{'='*90}")
    print(f"{'K':>6} {f'dim={min_dim}':>10} {f'dim={max_dim}':>10} {'Delta':>10} {'Delta%':>10}")
    print(f"{'-'*6} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

    for k in all_k:
        low_dim = [r for r in all_rows if r["K"] == k and r["n_dimensions"] == min_dim]
        high_dim = [r for r in all_rows if r["K"] == k and r["n_dimensions"] == max_dim]
        if not low_dim or not high_dim:
            continue
        sp_low = np.mean([r["speedup"] for r in low_dim])
        sp_high = np.mean([r["speedup"] for r in high_dim])
        delta = sp_high - sp_low
        pct = (delta / sp_low * 100) if sp_low > 0 else 0
        print(f"{k:>6} {sp_low:>10.2f} {sp_high:>10.2f} {delta:>10.2f} {pct:>9.1f}%")

    # Interpretation
    print(f"\n{'='*90}")
    print("Interpretation Guide")
    print(f"{'='*90}")
    print("""
If speedup INCREASES with n_dimensions:
  -> Higher compute density reduces sync overhead proportionally
  -> The threading architecture scales well when workload is compute-heavy
  -> The current bottleneck (at dim=32) is synchronization, not parallelization

If speedup is FLAT across n_dimensions:
  -> The bottleneck is purely synchronization (compute density doesn't help)
  -> Would need a different communication model (MPI, multiprocessing)

If speedup DECREASES with n_dimensions:
  -> Larger data exceeds L3 cache capacity, increasing memory stalls
  -> The workload becomes more memory-bound, not less
""")

    # Save CSV
    out_csv = os.path.join(os.path.dirname(__file__), "dim_study_summary.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["config", "n_dimensions", "K", "speedup",
                                          "throughput_sps", "wall_time_sec", "n_reps"])
        w.writeheader()
        for r in sorted(all_rows, key=lambda x: (x["n_dimensions"], x["K"])):
            w.writerow(r)
    print(f"Saved summary to: {out_csv}")


if __name__ == "__main__":
    main()
